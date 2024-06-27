import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul


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
        # x: [bs, sp, 1], in_degree: [bs, sp]
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

    def forward(self, q, k, v, attn_bias=None, mask=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        # q, k, v: [b, s+1, np, hn]
        batch_size, s_len = q.size(0), q.size(1)
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
        x = x.view(batch_size, s_len, -1)

        return x


class MultiHeadAttention(nn.Module):
    """multi-headed attention.

    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads # hn
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        self.dist_attn = CoreAttention(hidden_size, attention_dropout_rate, num_heads)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, x, attn_bias=None, mask=None):
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
        

        # ==================================
        # core attention computation
        # ==================================
        
        x = self.dist_attn(q, k, v, attn_bias)

        # =================
        # linear
        # =================

        # [b, s/p+1, h]
        x = self.output_layer(x)  

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

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):
        # ==================================
        # MHA
        # ==================================     
        y = self.self_attention_norm(x) # x: [b, s/p+1, h]
        y = self.self_attention(y, attn_bias, mask=mask)
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
    ):
        super().__init__()
        if dataset_name == "ZINC":
            num_atoms = 64
            num_edges = 64
            num_edge_dis = 40
            num_spatial = 40
            num_in_degree = 64
            num_out_degree = 64
        else:
            num_atoms = 512 * 9 + 1
            num_edges = 512 * 3 + 1
            num_edge_dis = 128
            num_spatial = 512
            num_in_degree = 512
            num_out_degree = 512  
            
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
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads
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
        
        
    def forward(self, batched_data, perturb=None):
        x = self.graph_node_feature(batched_data) # [bs, s/p + 1, h]
        attn_bias = self.graph_attn_bias(batched_data) # [bs, n_head, s+1, s+1]

        output = self.input_dropout(x)
        
        # Graphormer encoder
        for enc_layer in self.layers:
            output = enc_layer(
                output, 
                attn_bias, 
                mask=None,
            )  # TODO readd mask as adj
        output = self.final_ln(output)
            
        # Output part
        output = self.downstream_out_proj(output[:, 0, :]) # [b, s/p+1, h]
        return output