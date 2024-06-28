import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter
from gt_sp.layer import DistributedAttentionNoMerge
from gt_sp.initialize import (
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_last_batch_flag,
)
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
    
    
class CoreAttention(nn.Module):
    """
    Core attention
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(CoreAttention, self).__init__()

        # SP group: Per attention head and per partition values.
        seq_parallel_world_size = 1
        if sequence_parallel_is_initialized():
            seq_parallel_world_size = get_sequence_parallel_world_size()
        world_size = seq_parallel_world_size 

        self.hidden_size_per_partition = hidden_size // world_size
        self.hidden_size_per_attention_head = hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // world_size

        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_dropout_rate = attention_dropout_rate


    def reset_parameters(self):
        torch.nn.init.constant_(self.b, 0.1)


    def sparse_attention_bias(self, k, q, v, edge_index, attn_bias):
        # kqv: [b, s+p, np, hn], edge_index: [2, n_edges], attn_bias: [b, s+p, s+p, np]
        batch_size, node_num = k.size(0), k.size(1)
        if self.training:
            num_heads = self.num_attention_heads_per_partition
        else:
            num_heads = self.num_heads
   
        # Reshaping into [total_s, np, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, np, hn]
        q = q.view(-1, num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, num_heads, self.hidden_size_per_attention_head)
        # q = q.half()
        # k = k.half()
        # v = v.half()

        # -> [n_edges, np, hn]
        src = k[edge_index[0].to(torch.long)] 
        dest = q[edge_index[1].to(torch.long)] 
        score = torch.mul(src, dest)  # element-wise multiplication
            
        # Scale scores by sqrt(d)
        score = score / self.scale

        # Use available edge features to modify the scores for edges
        # -> [total_edges, np, 1] 
        score = score.sum(-1, keepdim=True).clamp(-5, 5)

        # [b, s+p, s+p, np] -> [b, s+p, s+p, np] -> [b, s+p, b, s+p, np]
        if attn_bias is not None:
            # attn_bias = attn_bias.repeat(1, 1, 1, num_heads)
            attn_bias = attn_bias.unsqueeze(2).repeat(1, 1, batch_size, 1, 1)  
            attn_bias = attn_bias.view(batch_size*node_num, batch_size*node_num, -1)
    
            score = score + \
                    attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :] 

        # softmax -> [total_edges, np, 1]
        score = torch.exp(score) 

        # Apply attention score to each source node to create edge messages
        # -> [total_edges, np, hn]
        msg = v[edge_index[0].to(torch.long)] * score
        
        # Add-up real msgs in destination nodes as given by edge_index[1]
        # -> [total_s, np, hn]
        wV = torch.zeros_like(v)  
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        # Compute attention normalization coefficient
        # -> [total_s, np, 1]
        Z = score.new_zeros(v.size(0), num_heads, 1)    
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        x = wV / (Z + 1e-6)
        
        return x


    def full_attention(self, k, q, v, attn_bias):
        # [b, np, sq+1, sk+1]
        output_size = (q.size(0),
                       q.size(2),
                       q.size(1),
                       k.size(1))
        if self.training:
            num_heads = self.num_attention_heads_per_partition
        else:
            num_heads = self.num_heads

        
        # [b, sq+1, np, hn] -> [sq+1, b * np, hn]
        q = q.view(output_size[2], output_size[0] * output_size[1], -1)
        # [b, sk+1, np, hn] -> [sk+1, b * np, hn]
        k = k.view(output_size[3], output_size[0] * output_size[1], -1)
        
        # Scaled Dot-Product Attention.
        q = q * self.scale
        
        # Raw attention scores. [b * np, sq+1, sk+1]
        x = torch.matmul(q.transpose(0, 1), # [b * np, sq+1, hn]
                         k.transpose(0, 1).transpose(1, 2))  # [b * np, hn, sk+1]

        # change view to [b, np, sq+1, sk+1]
        x = x.view(*output_size)

        if attn_bias is not None:
            # attn_bias: [b, s+1, s+1, 1] -> [b, s+1, s+1, np]
            # attn_bias = attn_bias.repeat(1, 1, 1, num_heads) 
            attn_bias = attn_bias.view(*output_size) # [b, s+1, s+1, np] -> [b, np, s+1, s+1]
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)

        # =========================
        # Context layer. [s, b, hp]
        # =========================

        # value_layer -> context layer.
        # [b, sk+1, np, hn] --> [b, np, sq+1, hn]
  
        # [b, np, sq+1, hn]
        output_size = (v.size(0),
                       v.size(2),
                       q.size(0),
                       v.size(3))

        # -> [sk+1, b * np, hn]
        v = v.view(v.size(1), output_size[0] * output_size[1], -1)     

        # change view [b * np, sq+1, sk+1]
        x = x.view(output_size[0] * output_size[1], output_size[2], -1)  

        # matmul: [b * np, sq+1, hn]
        x = torch.bmm(x, v.transpose(0, 1)) # [b * np, sk+1, hn]
        
        # [b, np, sq+1, hn]
        x = x.view(*output_size)

        # [b, np, sq+1, hn] --> [b, sq+1, np, hn]
        x = x.permute(0, 2, 1, 3).contiguous()

        # [b, sq+1, np, hn] --> [b, sq+1, hp]
        x = x.view(output_size[0], output_size[2], -1)

        return x


    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        # q, k, v: [b, s+1, np, hn]
        batch_size, s_len = q.size(0), q.size(1)
        
        if attn_type == "full":
            x = self.full_attention(k, q, v, attn_bias)
        elif attn_type == "sparse":
            x = self.sparse_attention_bias(k, q, v, edge_index, attn_bias)
            # x = x.float()
        elif attn_type == "flash":
            q = q.half()
            k = k.half()
            v = v.half()
            x = flash_attn_func(q, k, v, self.attention_dropout_rate)
            x = x.float()

        # [b, s+1, hp]
        x = x.view(batch_size, s_len, -1)

        return x


class MultiHeadAttention(nn.Module):
    """Distributed multi-headed attention.

    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads # hn
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        local_attn = CoreAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_bias_dim)
        self.dist_attn = DistributedAttentionNoMerge(local_attn, get_sequence_parallel_group())

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def reset_parameters(self):
        torch.nn.init.constant_(self.b, 0.1)

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        # x: [b, s/p+1, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # q, k, v: [b, s/p+1, h] -> [b, s/p+1, n_head, hn]
        batch_size = x.size(0) # number of sequences to train a time 
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size) 
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)

        # ==================================
        # core attention computation
        # ==================================

        # [b, s/p+1, h]
        x = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type)

        # =================
        # linear
        # =================

        # [b, s/p+1, h]
        x = self.output_layer(x)  

        return x


class EncoderLayer(nn.Module):
    """A single encoder layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim):
        super(EncoderLayer, self).__init__()
  
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_bias_dim)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):

        # ==================================
        # MHA
        # ==================================     
          
        y = self.self_attention_norm(x) # x: [b, s/p+1, h]
        y = self.self_attention(y, attn_bias, edge_index, attn_type=attn_type)
        y = self.self_attention_dropout(y)
        x = x + y
        
        # ==================================
        # MLP
        # ==================================    
            
        y = self.ffn_norm(x) # x: [b, s/p+1, h]
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
    

class Graphormer(nn.Module):
    """Graphormer for node-level task: one node - one token
        global token index: 0

    """
    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        input_dropout_rate,
        attention_dropout_rate,
        ffn_dim,
        num_global_node,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.n_layers = n_layers
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_global_node = num_global_node
        
        self.graph_token = nn.Embedding(self.num_global_node, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(self.num_global_node, attn_bias_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))


    def forward(self, x, attn_bias, edge_index, perturb=None, attn_type=None):
        # x -> [bs=1, s/p, x_d]
        x = x.unsqueeze(0) 
        n_graph = x.shape[0] 
        
        # [bs, s/p, x_d] -> [bs, s/p, h]
        node_feature = self.node_encoder(x)         
        
        if perturb is not None:
            node_feature += perturb

        # [bs, 1, h]
        global_node_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1) 
        # [b, s/p + 1, h]
        node_feature = torch.cat([global_node_feature, node_feature], dim=1) 

        output = self.input_dropout(node_feature) 

        if attn_bias is not None:
            # attn_bias: [bs, s/p, s, attn_bias_dim]
            attn_bias = attn_bias.unsqueeze(0) 
            n_graph, subseq_len, seq_len = attn_bias.size()[:3]
            graph_attn_bias = attn_bias.clone() 

            # [b, s/p+1, s, attn_bias_dim]
            graph_attn_bias = torch.cat([self.graph_token_virtual_distance.weight.unsqueeze(0).unsqueeze(2).repeat(n_graph, 1, seq_len, 1), 
                                         graph_attn_bias], dim=1) 
            # [b, s/p+1, s+1, attn_bias_dim]
            graph_attn_bias = torch.cat([self.graph_token_virtual_distance.weight.unsqueeze(0).unsqueeze(0).repeat(n_graph, subseq_len+self.num_global_node, 1, 1), graph_attn_bias], dim=2)  
            graph_attn_bias = graph_attn_bias.repeat(1, 1, 1, self.num_heads) # [b, s/p+1, s+1, n]
        else:
            graph_attn_bias = attn_bias
        
        # transfomrer encoder
        for enc_layer in self.layers:
            output = enc_layer(output, attn_bias=graph_attn_bias, edge_index=edge_index, attn_type=attn_type)
        output = self.final_ln(output)

        # output part
        output = self.downstream_out_proj(output[0, 1:, :])
        return F.log_softmax(output, dim=1)


