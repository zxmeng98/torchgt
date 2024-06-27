import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        

class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    global token at index 0
    """

    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)


    def forward(self, batched_data):
        # x: [n_graph (bs), sp, 1], in_degree: [bs, sp]
        x, in_degree, out_degree = (
            batched_data.x,
            batched_data.in_degree,
            batched_data.out_degree,
        )
        n_graph, n_node = x.size()[:2]

        node_feature = self.atom_encoder(x).sum(dim=-2)  # [bs, sp, h]
        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )
        
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        # graph token + node feauture
        # [bs, sp, h] -> [bs, sp+1, h]
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    global token at index 0
    """

    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
    ):
        super(GraphAttnBias, self).__init__()

        # num_heads = 1
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.edge_type = edge_type

        self.edge_encoder = nn.Embedding(num_edges, num_heads, padding_idx=0)
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)


    def forward(self, batched_data):
        # attn_bias: [bs, s+1, s+1], all zeros。这个batch所有graph的attn_bias都concat在一起了
        attn_bias, spatial_pos = (
            batched_data.attn_bias,
            batched_data.spatial_pos,
        )
        edge_input, attn_edge_type = (
            batched_data.edge_input,
            batched_data.attn_edge_type,
        )
        n_graph, n_node = spatial_pos.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [bs, n_head, s+1, s+1]
        
        # [bs, s, s, n_head] -> [bs, n_head, s, s]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        # [bs, n_head, s+1, s+1]    
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # -> [bs, s, s, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            # -> [bs, s, s, max_dist, n_head]
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            # -> [bs, n_head, s, s]
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # [bs, n_head, s+1, s+1]
        return graph_attn_bias
    

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
    Core attn 
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(CoreAttention, self).__init__()

        self.hidden_size_per_attention_head = att_size = hidden_size // num_heads
        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)


    def full_attention(self, k, q, v, attn_bias, mask=None):
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


    def sparse_attention(self, k, q, v, e, edge_index, attn_bias):
        # kqv: [total_s, n, hn],  e: [total_edges, n, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+1, s+1]
        # kqv: [b, s+1, n, hn],  e: [b, n_edges, n, hn], edge_index: [b, 2, n_edges], attn_bias: [b, n, s+1, s+1]
        
        # -> [total_edges, n, hn]
        # 在这个地方pad score -inf？
        batch_size, edge_num = k.size(0), edge_index.size(2)
        score_full = k.new_zeros(batch_size, edge_num, k.size(2), k.size(3))
        for i in range(batch_size):
            src = k[i, edge_index[i, 0, :].to(torch.long), :, :] 
            dest = q[i, edge_index[i, 1, :].to(torch.long), :, :] 
            score = torch.mul(src, dest)  # element-wise multiplication
            score_full[i, :, :, :] = score
            
        # Scale scores by sqrt(d)
        score_full = score_full / self.scale

        # Use available edge features to modify the scores for edges
        # -> [total_edges, n, hn] -> [total_edges, n, 1]
        # score = torch.mul(score, e)  
        score_full = torch.exp(score_full.sum(-1, keepdim=True).clamp(-5, 5)) 

        # Apply attention score to each source node to create edge messages
        # -> [total_edges, n, hn]
        msg_full = k.new_zeros(batch_size, edge_num, k.size(2), k.size(3))
        for i in range(batch_size):
            msg = v[i, edge_index[i, 0, :].to(torch.long), :, :] * score_full[i, :, :, :] 
            msg_full[i, :, :, :] = msg
        
        # Add-up real msgs in destination nodes as given by edge_index[1]
        # -> [total_s, n, hn]
        wV = torch.zeros_like(v) 
        for i in range(batch_size): 
            scatter(msg_full[i, :, :, :], edge_index[i, 1, :], dim=0, out=wV[i, :, :, :], reduce='add') # TODO pad的index相当于在这里重复加了好多次？

        # Compute attention normalization coefficient
        # -> [total_s, n, 1]
        Z = score_full.new_zeros(batch_size, v.size(1), self.num_heads, 1)  
        for i in range(batch_size):
            scatter(score_full[i, :, :, :], edge_index[i, 1, :], dim=0, out=Z[i, :, :, :], reduce='add')
        
        x = wV / (Z + 1e-6)

        return x
    
    
    def sparse_attention_flatten(self, k, q, v, e, edge_index, attn_bias):
        # kqv: [b, s+1, n, hn],  e: [b, n_edges, n, hn], edge_index: [2, total_edges]

        # Reshaping into [total_s, n, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, n, hn],  e: [total_edges, n, hn]
        q = q.view(-1, self.num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, self.num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, self.num_heads, self.hidden_size_per_attention_head)
        e = e.view(-1, self.num_heads, self.hidden_size_per_attention_head)

        # -> [total_edges, n, hn]
        # 在这个地方pad score -inf？
        src = k[edge_index[0].to(torch.long)] 
        dest = q[edge_index[1].to(torch.long)] 
        score = torch.mul(src, dest)  # element-wise multiplication
            
        # Scale scores by sqrt(d)
        score = score / self.scale

        # Use available edge features to modify the scores for edges
        # -> [total_edges, n, hn] -> [total_edges, n, 1]
        # score = torch.mul(score, e)  
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5)) 

        # Apply attention score to each source node to create edge messages
        # -> [total_edges, n, hn]
        msg = v[edge_index[0].to(torch.long)] * score
        
        # Add-up real msgs in destination nodes as given by edge_index[1]
        # -> wV: [total_s, n, hn]
        wV = torch.zeros_like(v) 
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add') # TODO pad的index相当于在这里重复加了好多次？有些pad的点永远访问不到，值是0

        # Compute attention normalization coefficient
        # -> [total_s, n, 1]
        Z = score.new_zeros(v.size(0), self.num_heads, 1)  
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        x = wV / (Z + 1e-6)

        return x
    

    def sparse_attention_bias(self, k, q, v, edge_index, attn_bias):
        # kqv: [b, s+1, n, hn],  e: [b, n_edges, n, hn], edge_index: [b, 2, n_edges], attn_bias: [b, n, s+1, s+1]
        batch_size, node_num = k.size(0), k.size(1)
        
        # Reshaping into [total_s, n, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, n, hn],  e: [total_edges, n, hn]
        q = q.view(-1, self.num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, self.num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, self.num_heads, self.hidden_size_per_attention_head)
        # e = e.view(-1, self.num_heads, self.hidden_size_per_attention_head)

        # -> [total_edges, n, hn]
        src = k[edge_index[0].to(torch.long)] 
        dest = q[edge_index[1].to(torch.long)] 
        score = torch.mul(src, dest)  # element-wise multiplication
        
        # Scale scores by sqrt(d)
        score = score / self.scale

        # Use available edge features to modify the scores for edges
        # -> [total_edges, n, 1] 
        score = score.sum(-1, keepdim=True).clamp(-5, 5) # sum(-1)后对应full attn的score

        # [b, n, s+1, s+1] -> [b, s+1, s+1, n] -> [b, s+1, b, s+1, n]
        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 2, 3, 1).contiguous().unsqueeze(2).repeat(1, 1, batch_size, 1, 1)  
            attn_bias = attn_bias.view(batch_size*node_num, batch_size*node_num, self.num_heads)

            # NOTE attn_bias里pad的地方是-inf，所以加到score里面pad的边对应的也变为-inf，下面经过exp就为0了
            score = score + \
                attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :].unsqueeze(2) 

        # Softmax 
        # -> [total_edges, n, 1] 
        score = torch.exp(score) 
        # print(score[110:150, :2, 0])
        # exit(0)

        # Apply attention score to each source node to create edge messages
        # -> [total_edges, n, hn]
        msg = v[edge_index[0].to(torch.long)] * score
        # print(msg[110:150, :2, 0])
        # exit(0)
        
        # Add-up real msgs in destination nodes as given by edge_index[1]
        # -> [total_s, n, hn]
        wV = torch.zeros_like(v)  
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        # Compute attention normalization coefficient
        # -> [total_s, n, 1]
        Z = score.new_zeros(v.size(0), self.num_heads, 1)    
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        x = wV / (Z + 1e-6)
        
        return x


    def forward(self, q, k, v, attn_bias=None, edge_index=None, mask=None, attn_type="sparse"):

        # q, k, v: [b, s+1, n, hn], e: [b, n_edges, n, hn]
        batch_size, s_len = q.size(0), q.size(1)
        
        # attn_bias: [b, n, s+1, s+1]
        if attn_type == "full":
            x = self.full_attention(k, q, v, attn_bias)
        elif attn_type == 'sparse':
            x = self.sparse_attention_bias(k, q, v, edge_index, attn_bias)
        elif attn_type == "flash":
            q = q.half()
            k = k.half()
            v = v.half()
            x = flash_attn_func(q, k, v, self.attention_dropout_rate)
            x = x.float()
            
        # -> [bs, s+1, h]
        x = x.view(batch_size, s_len, -1)  
        # print(x[72:74, -10:, :2])
        # exit(0)

        return x


class MultiHeadAttention(nn.Module):
    """multi-headed attention.

    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_type):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads # hn
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        # edge related
        self.edge_encoder = nn.Linear(1, hidden_size)
        self.linear_e = nn.Linear(hidden_size, num_heads * att_size)

        self.dist_attn = CoreAttention(hidden_size, attention_dropout_rate, num_heads)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, x, attn_bias=None, mask=None, edge_attr=None, edge_index=None, attn_type=None):
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
        
        # [b, n_edges, edge_attr_dim] -> [b, n_edges, n_head, hn]
        # edge_attr = self.edge_encoder(edge_attr)
        # e = self.linear_e(edge_attr).view(batch_size, -1, self.num_heads, self.att_size)
        # ==================================
        # core attention computation
        # ==================================
        
        x = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type=attn_type)

        # =================
        # linear
        # =================

        # [b, s+1, h]
        x = self.output_layer(x)  

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_type
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_type
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None, mask=None, edge_attr=None, edge_index=None, attn_type=None):
        # ==================================
        # MHA
        # ==================================     
        y = self.self_attention_norm(x) # x: [b, s/p+1, h]
        y = self.self_attention(y, attn_bias, mask=mask, edge_attr=edge_attr, edge_index=edge_index, attn_type=attn_type)
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
    """Graphormer for graph-level task: one graph - one seq

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
        attn_type
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
            
        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=hidden_dim,
        )
        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=hidden_dim,
        )
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, attn_type
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        if dataset_name == "PCQM4M-LSC":
            self.out_proj = nn.Linear(hidden_dim, 1)
        else:
            self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        
        
    def forward(self, batched_data, perturb=None, attn_type=None):
        x = self.graph_node_feature(batched_data) # [bs, s/p + 1, h]
        if batched_data.attn_bias is not None:
            attn_bias = self.graph_attn_bias(batched_data) # [bs, n_head, s+1, s+1]
        else:
            attn_bias = None
        edge_index = batched_data.edge_index
        del batched_data

        output = self.input_dropout(x)
        
        # Graphormer encoder
        for enc_layer in self.layers:
            output = enc_layer(
                output, 
                attn_bias, 
                mask=None,
                edge_attr=None,
                edge_index=edge_index,
                attn_type=attn_type,
            )  # TODO readd mask as adj
        output = self.final_ln(output)
            
        # Output part
        output = self.downstream_out_proj(output[:, 0, :]) # [b, s/p+1, h]
        return output