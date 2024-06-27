import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from gt_sp.gt_layer import DistributedAttention, _SeqGather
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_global_token_indices,
)
from torch_scatter import scatter
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        

class CoreAttention(nn.Module):
    """
    Core attn 
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(CoreAttention, self).__init__()

        # SP group: Per attention head and per partition values.
        seq_parallel_world_size = 1
        if sequence_parallel_is_initialized():
            seq_parallel_world_size = get_sequence_parallel_world_size()
        world_size = seq_parallel_world_size 

        self.hidden_size_per_partition = hidden_size // world_size
        self.hidden_size_per_attention_head =  hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // world_size

        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_dropout_rate = attention_dropout_rate

    # def full_flash_attention(self, q, k, v, attn_bias=None, mask=None):
    #     return flash_attn_func(q, k, v, self.attention_dropout_rate)
    

    def full_attention(self, q, k, v, attn_bias, mask=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        q = q.transpose(1, 2)   # [b, np, s+1, hn]
        v = v.transpose(1, 2)  # [b, np, s+1, hn]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, np, hn, s+1]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            # attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)
            x = x + attn_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask, 0)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        return x
    

    def sparse_attention_bias(self, q, k, v, edge_index, attn_bias):
        # kqv: [total_s, n, hn],  e: [total_edges, n, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+1, s+1]
        batch_size, node_num = k.size(0), k.size(1)
        
        # Reshaping into [total_s, np, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, np, hn],  e: [total_edges, np, hn]
        q = q.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        k = k.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        v = v.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)

        # -> [total_edges, np, hn]
        src = k[edge_index[0].to(torch.long)] 
        dest = q[edge_index[1].to(torch.long)] 
        score = torch.mul(src, dest)  # element-wise multiplication
            
        # Scale scores by sqrt(d)
        score = score / self.scale

        # Use available edge features to modify the scores for edges
        # -> [total_edges, np, 1] 
        score = score.sum(-1, keepdim=True).clamp(-5, 5)

        # [b, np, s+1, s+1] -> [b, s+1, s+1, np] -> [b, s+1, b, s+1, np]
        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 2, 3, 1).contiguous().unsqueeze(2).repeat(1, 1, batch_size, 1, 1)  
            attn_bias = attn_bias.view(batch_size*node_num, batch_size*node_num, self.num_attention_heads_per_partition)

            # NOTE attn_bias里pad的地方是-inf，所以加到score里面pad的边对应的也变为-inf，下面经过exp就为0了
            score = score + \
                    attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :].unsqueeze(2) 

        # softmax -> [total_edges, np, 1]
        # print(score[80:150, :2, 0])
        score = torch.exp(score) 
        # print(score[80:150, :2, 0])

        # Apply attention score to each source node to create edge messages
        # -> [total_edges, np, hn]
        msg = v[edge_index[0].to(torch.long)] * score
        # print(msg[110:150, :2, 0])
        # exit(0)
        
        # Add-up real msgs in destination nodes as given by edge_index[1]
        # -> [total_s, np, hn]
        wV = torch.zeros_like(v)  
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        # Compute attention normalization coefficient
        # -> [total_s, np, 1]
        Z = score.new_zeros(v.size(0), self.num_attention_heads_per_partition, 1)    
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        x = wV / (Z + 1e-6)
        
        return x


    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        # q, k, v: [b, s+p, np, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+p, s+p]
        batch_size, s_len = q.size(0), q.size(1)
         
        if attn_type == "full":
            x = self.full_attention(q, k, v, attn_bias)
        elif attn_type == "sparse":
            x = self.sparse_attention_bias(q, k, v, edge_index, attn_bias)
        elif attn_type == "flash":
            q = q.half()
            k = k.half()
            v = v.half()
            x = flash_attn_func(q, k, v, self.attention_dropout_rate)
            x = x.float()
        
        # [b, s+p, hp]
        x = x.view(batch_size, s_len, -1)

        return x


class MultiHeadAttention(nn.Module):
    """Distributed multi-headed attention.

    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads # hn
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        
        self.dist_attn = CoreAttention(
            hidden_size, attention_dropout_rate, num_heads)
    def forward(self, x, attn_bias=None, mask=None, edge_index=None, attn_type=None):
        # x: [b, s/p+1, h], attn_bias: [b, n_head, s+1, s+1]
        orig_q_size = x.size()
        # =====================
        # Query, Key, and Value
        # =====================

        # q, k, v: [b, s/p+1, h] -> [b, s/p+1, n_head, hn]
        batch_size = x.size(0) # number of sequences to train a time 
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size) 
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)
        # print(f'rank {get_sequence_parallel_rank()} q: {q[:, 0, :, :]}')
        # exit(0)
        

        # ==================================
        # core attention computation
        # ==================================
        x = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type)

        # =================
        # linear
        # =================

        # [b, s/p+1, h]

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.O = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
            
        # FFN
        self.FFN_layer1 = nn.Linear(hidden_size, hidden_size*2)
        self.FFN_layer2 = nn.Linear(hidden_size*2, hidden_size)

        self.layer_norm2 = nn.LayerNorm(hidden_size)
            
            
    def forward(self, x, attn_bias=None, mask=None, edge_index=None, attn_type=None):
        # ==================================
        # MHA
        # ==================================     
        # x: [b, s/p+1, h]
        y = self.self_attention(x, attn_bias, mask=mask, edge_index=edge_index, attn_type=attn_type)
        y = self.self_attention_dropout(y)
        y = self.O(y)
        x = x + y
        x = self.layer_norm1(x)

        # ==================================
        # MLP
        # ==================================    

        y = self.FFN_layer1(y)
        y = F.relu(y)
        y = self.self_attention_dropout(y)
        y = self.FFN_layer2(y)
        x = x + y
        x = self.layer_norm2(x)

        return x
        
        
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y



class GT(nn.Module):
    """GT for graph-level task: one graph - one seq

    """
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        ffn_dim,
        dataset_name,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        output_dim,
    ):
        super().__init__()
        if dataset_name == "ZINC":
            num_atoms = 64
            num_edges = 64
            num_edge_dis = 40
            num_spatial = 40
            num_in_degree = 64
            num_out_degree = 64
        elif dataset_name == "MalNet":
            num_atoms = 560000
            num_edges = 55000
            num_edge_dis = 128
            num_spatial = 512
            num_in_degree = 560000
            num_out_degree = 560000
        else:
            num_atoms = 7000
            num_edges = 7000
            num_edge_dis = 128
            num_spatial = 512
            num_in_degree = 7000
            num_out_degree = 7000
            
        self.embedding_h = nn.Embedding(num_atoms, hidden_dim) # node feat is an integer
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        
        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)

        self.MLP_layer = MLPReadout(hidden_dim, output_dim)   # 1 out dim since regression problem  
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        
        
    def forward(self, batched_data, perturb=None, attn_type=None):
        x = self.embedding_h(batched_data.x).sum(dim=-2) # [bs, s/p + 1, h]
        n_graph, n_node = x.size()[:2]
        # graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        # x = torch.cat([graph_token_feature, x], dim=1)
        output = self.input_dropout(x)
        
        edge_index = batched_data.edge_index
        del batched_data
        
        # Graphormer encoder
        # [b, s/p+1, h]
        for enc_layer in self.layers:
            output = enc_layer(
                output, 
                edge_index=edge_index,
                attn_type=attn_type,
            ) 
            
        # global_token_indices = get_global_token_indices(last_batch=False)
        # output = torch.index_select(output, 1, 
        #                          torch.LongTensor([i for i in range(output.size(1)) if i not in global_token_indices]).to(output.device))
        
        if sequence_parallel_is_initialized():
            output = _SeqGather.apply(output, 1)
        output = output.mean(dim=1)
        
        # Output part
        output = self.MLP_layer(output) 
        # print(output.shape)
        # exit(0)
        return output