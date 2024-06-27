# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from gt_sp.initialize import (
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    get_global_token_indices,
    get_global_token_num,
    last_batch_flag,
    get_last_batch_flag,
)
import time
from gt_sp.utils import partition_graph_and_remap


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_bool(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(False)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1 # pad id = 0
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_edge_attr_unsqueeze(x, padlen):
    edge_num, edge_attr_dim = x.size()
    if edge_num < padlen:
        new_x = x.new_zeros([padlen, edge_attr_dim], dtype=x.dtype)
        new_x[:edge_num, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_edge_index_unsqueeze(x, padlen, max_node_index):
    # TODO Pad的值为1
    xlen, edge_num = x.size()
    if edge_num < padlen:
        new_x = x.new_zeros([xlen, padlen], dtype=x.dtype).fill_(int(max_node_index))
        new_x[:, :edge_num] = x
        x = new_x
    return x.unsqueeze(0)


def fix_edge_index(x, num_node):
    # Add new edges of virtual nodes
    virt_edges = []

    num_virtual_tokens = 1
    for idx in range(num_virtual_tokens):
        virt_edge_index = torch.cat([(torch.arange(num_node)+(1+idx)).view(1, -1), # virtual node index = 0
                                        (x.new_zeros([num_node])+idx).view(1, -1)], dim=0)
        virt_edges.append(virt_edge_index)

        virt_edge_index = torch.cat([(x.new_zeros([num_node])+idx).view(1, -1), 
                                    (torch.arange(num_node)+(1+idx)).view(1, -1)], dim=0)
        virt_edges.append(virt_edge_index)
    
    extra_virt_edges = torch.cat(virt_edges, dim=1)
    x = torch.cat([(x + 1), extra_virt_edges], dim=1) # virtual node index = 0, other nodes start from 1
    return x


def flatten_edge_index_nopad(x, batch_node_num):
    # x: list(...tensor(2, fixed_edge_num)...)
    batch_size = len(x)
    num_edges = torch.LongTensor([i.size(1) for i in x])
    edge_index_base = torch.arange(0, batch_node_num*batch_size, batch_node_num).repeat_interleave(num_edges)
    edge_index_base = edge_index_base.unsqueeze(0).repeat(2, 1)
    x = torch.cat(x, dim=1) + edge_index_base
    return x


def adjust_edge_index_nomerge(edge_index, sub_seq_len):
    """
    从第二个点开始，每隔s个点插入一个新点，调整edge_index的值。

    :param edge_index: 原始图的edge_index，2xM的张量。
    :param s: 每隔s个点插入一个新点。
    :return: 调整后的edge_index。
    """
    # 计算新的节点编号
    new_index = edge_index.clone()
    mask = edge_index > 0
    new_index[mask] = edge_index[mask] + ((edge_index[mask].float() - 1) // sub_seq_len).to(torch.int64)

    return new_index


class Batch:
    def __init__(
        self,
        idx,
        attn_bias,
        attn_edge_type,
        spatial_pos,
        in_degree,
        out_degree,
        x,
        edge_input,
        y,
        adj,
        graph_node_num,
        edge_attr,
        edge_index,
        sub_split_seq_lens,
    ):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.spatial_pos = (
            attn_bias,
            attn_edge_type,
            spatial_pos,
        )
        self.edge_input = edge_input
        self.adj = adj
        self.graph_node_num = graph_node_num
        self.edge_attr = edge_attr
        self.edge_index = edge_index
        self.sub_split_seq_lens = sub_split_seq_lens

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = (
            self.in_degree.to(device),
            self.out_degree.to(device),
        )
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.spatial_pos = (
            self.attn_bias.to(device),
            self.attn_edge_type.to(device),
            self.spatial_pos.to(device),
        )
        self.edge_input = self.edge_input.to(device)
        self.adj = self.adj.to(device)
        self.graph_node_num = self.graph_node_num.to(device)
        self.edge_index = self.edge_index.to(device)
        
        return self

    def __len__(self):
        return self.in_degree.size(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, myargs=None):
    # Dataloader会自动取一个batch的数据到items
    num_virtual_tokens = 1
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.adj,
            item.edge_attr,
            item.edge_index,
            item.num_nodes,
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        adjs,
        edge_attrs,
        edge_indexes,
        num_nodes,
    ) = zip(*items)
    
    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][num_virtual_tokens:, num_virtual_tokens:][
            spatial_poses[idx] >= spatial_pos_max
        ] = float("-inf")

    # Maximum number of nodes in the batch.
    graph_node_num = torch.LongTensor([i.size(0) for i in xs])
    max_node_num = max(i.size(0) for i in xs)
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    div = max_node_num // seq_parallel_world_size
    max_node_num = seq_parallel_world_size * div + (seq_parallel_world_size - 1) 

    # # Graph reorder & blockize
    # edge_indexes, xs, edge_inputs, attn_biases, spatial_poses, in_degrees, out_degrees  = [], [], [], [], [], [], []
    # k = 2
    # for i in range(len(edge_indexes_raw)):
    #     edge_index_i, sorted_indices = partition_graph_and_remap(edge_indexes_raw[i], k)
    #     edge_indexes.append(edge_index_i)
    #     xs.append(torch.index_select(xs_raw[i], 0, sorted_indices))
    #     edge_input = torch.index_select(edge_inputs_raw[i], 0, sorted_indices)
    #     edge_input = torch.index_select(edge_input, 1, sorted_indices)
    #     edge_inputs.append(edge_input)
    #     attn_bias = torch.index_select(attn_biases_raw[i], 0, sorted_indices)
    #     attn_bias = torch.index_select(attn_bias, 1, sorted_indices)
    #     attn_biases.append(attn_bias)
    #     spatial_pos = torch.index_select(spatial_poses_raw[i], 0, sorted_indices)
    #     spatial_pos = torch.index_select(spatial_pos, 1, sorted_indices)
    #     spatial_poses.append(spatial_pos)
    #     in_degrees.append(torch.index_select(in_degrees_raw[i], 0, sorted_indices))
    #     out_degrees.append(torch.index_select(out_degrees_raw[i], 0, sorted_indices))

    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    ) # sp下面这一步最慢


    attn_bias = torch.cat(
        [
            pad_attn_bias_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_biases
        ]
    )
    adj = torch.cat([pad_2d_bool(i, max_node_num + num_virtual_tokens) for i in adjs])
    attn_edge_type = torch.cat(
        [
            pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_edge_types
        ]
    )
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])

    ### Edge index related
    # Fix edge index: add new edges of virtual nodes
    fixed_edge_indexes = [fix_edge_index(edge_indexes[idx], num_nodes[idx]) for idx in range(len(edge_indexes))]
    t8 = time.time()

    if sequence_parallel_is_initialized():
        seq_parallel_world_size = get_sequence_parallel_world_size()
        x_i_list = [t for t in torch.tensor_split(x, seq_parallel_world_size, dim=1)]
        sub_split_seq_lens = [t.shape[1] for t in x_i_list] # [9, 9, 9, 8]
        total_real_seq_len = max(sub_split_seq_lens) * seq_parallel_world_size + num_virtual_tokens # 9 * 4 + 1 = 37
        nomerge_real_seq_len = (max(sub_split_seq_lens) + num_virtual_tokens) * seq_parallel_world_size # 10 * 4 = 40

        edge_indexes = [adjust_edge_index_nomerge(i, max(sub_split_seq_lens)) for i in fixed_edge_indexes]
        edge_index = flatten_edge_index_nopad(edge_indexes, nomerge_real_seq_len)


    else:
        sub_split_seq_lens = None
        total_real_seq_len = max_node_num + num_virtual_tokens

        edge_index = flatten_edge_index_nopad(fixed_edge_indexes, total_real_seq_len)


    return Batch(
        idx=torch.LongTensor(idxs), # [bs]
        attn_bias=attn_bias, # [bs, N_max+1, N_max+1] torch.float32
        attn_edge_type=attn_edge_type,  # [bs, N_max, N_max, edge_attr.size(-1)] torch.int64
        spatial_pos=spatial_pos,  # [bs, N_max, N_max] torch.int64
        in_degree=in_degree, # [bs, N_max] torch.int64
        out_degree=out_degree, # [bs, N_max] torch.int64
        x=x, # [bs, N_max, 1] torch.int64
        edge_input=edge_input, # [bs, N_max, N_max, max_dist, edge_attr.size(-1)] torch.int64
        y=y,  # [bs] torch.float32
        adj=adj,
        graph_node_num=graph_node_num, # [bs]
        edge_attr=None, # [bs, edge_num, edge_attr_dim] 
        edge_index=edge_index, # [bs, 2, edge_num] / [2, total_edges]
        sub_split_seq_lens=sub_split_seq_lens, # [seq_world_size]
    )