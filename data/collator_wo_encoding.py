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
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
from gt_sp.utils import partition_graph_and_remap
import torch.distributed as dist


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


def pad_3d_unsqueeze_optimized(x, padlen1, padlen2, padlen3):
    xlen1, xlen2, xlen3, xlen4 = x.size()
    pad_size1 = max(0, padlen1 - xlen1)
    pad_size2 = max(0, padlen2 - xlen2)
    pad_size3 = max(0, padlen3 - xlen3)
    x = F.pad(x, (0, 0, 0, pad_size3, 0, pad_size2, 0, pad_size1))
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


def flatten_edge_index(x, batch_node_num):
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
        # attn_edge_type,
        # spatial_pos,
        in_degree,
        out_degree,
        x,
        # edge_input,
        y,
        graph_node_num,
        # edge_attr,
        edge_index,
        sub_split_seq_lens,
    ):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        # self.attn_bias, self.attn_edge_type, self.spatial_pos = (
        #     attn_bias,
        #     attn_edge_type,
        #     spatial_pos,
        # )
        self.attn_bias = attn_bias
        # self.edge_input = edge_input
        self.graph_node_num = graph_node_num
        # self.edge_attr = edge_attr
        self.edge_index = edge_index
        self.sub_split_seq_lens = sub_split_seq_lens

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = (
            self.in_degree.to(device),
            self.out_degree.to(device),
        )
        self.x, self.y = self.x.to(device), self.y.to(device)
        # self.attn_bias, self.attn_edge_type, self.spatial_pos = (
        #     self.attn_bias.to(device),
        #     self.attn_edge_type.to(device),
        #     self.spatial_pos.to(device),
        # )
        # self.attn_bias = self.attn_bias.to(device)
        # self.edge_input = self.edge_input.to(device)
        self.graph_node_num = self.graph_node_num.to(device)
        # self.edge_attr = self.edge_attr.to(device)
        self.edge_index = self.edge_index.to(device)
        
        return self

    def __len__(self):
        return self.in_degree.size(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, myargs=None):
    t1 = time.time()
    # Dataloader会自动取一个batch的数据到items
    
    items = [item for item in items if item is not None]
    items = [
        (
            item.idx,
            # item.attn_bias,
            # item.attn_edge_type,
            # item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            # item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            # item.edge_attr,
            item.edge_index,
            item.num_nodes,
        )
        for item in items
    ]
    (
        idxs,
        # attn_biases,
        # attn_edge_types,
        # spatial_poses,
        in_degrees_raw,
        out_degrees_raw,
        xs_raw,
        # edge_inputs,
        ys,
        # edge_attrs,
        edge_indexes_raw,
        num_nodes,
    ) = zip(*items)

    # Maximum number of nodes in the batch. 
    graph_node_num = torch.LongTensor([i.size(0) for i in xs_raw])

    max_node_num = max(i.size(0) for i in xs_raw)
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()
        
    if max_node_num % seq_parallel_world_size != 0:
        div = max_node_num // seq_parallel_world_size
        max_node_num = seq_parallel_world_size * div + (seq_parallel_world_size - 1) 

    # Graph reorder & blockize
    if myargs.reorder:
        edge_indexes, xs, in_degrees, out_degrees  = [], [], [], []
        t0 = time.time()
        for i in range(len(edge_indexes_raw)):
            if graph_node_num[i] > 128000:
                k, block_size = 4, 32
            else:
                k, block_size = 4, 4
            edge_index_i, sorted_indices = partition_graph_and_remap(edge_indexes_raw[i], k, block_size)
            
            if myargs.model == "graphormer":
                # 重排x,y, attn_bias张量的第0维度
                sorted_indices = sorted_indices[sorted_indices != 0]
                sorted_indices = sorted_indices - 1
            
            # # Broadcast the reordered edges & sorted indices to all ranks
            # if sequence_parallel_is_initialized():
            #     device = f'cuda:{torch.cuda.current_device()}' # 相对GPU编号
            #     if myargs.rank == 0:
            #         edge_index_i_broad = edge_index_i.to(device)
            #         sorted_indices_broad = sorted_indices.to(device)
            #     else:
            #         edge_index_i_broad = torch.empty_like(edge_index_i,
            #                                 device=device,
            #                                 dtype=torch.int64)
            #         sorted_indices_broad = torch.empty_like(sorted_indices,
            #                                 device=device,
            #                                 dtype=torch.int64)
            #     dist.broadcast(edge_index_i_broad, src_rank, group=group)
            #     dist.broadcast(sorted_indices_broad, src_rank, group=group)
            # edge_index_i = edge_index_i_broad.to("cpu")
            # sorted_indices = sorted_indices_broad.to("cpu")
            # dist.barrier()
            
            # print(f"rank {get_sequence_parallel_rank()}, {edge_index_i} {sorted_indices}")
                   
            edge_indexes.append(edge_index_i)
            xs.append(torch.index_select(xs_raw[i], 0, sorted_indices))
            in_degrees.append(torch.index_select(in_degrees_raw[i], 0, sorted_indices))
            out_degrees.append(torch.index_select(out_degrees_raw[i], 0, sorted_indices))
        
        t1 = time.time()
        # print(f"Graph reorder & blockize: {t1-t0}s")

    else:
        xs, in_degrees, out_degrees, edge_indexes = xs_raw, in_degrees_raw, out_degrees_raw, edge_indexes_raw

    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])
    
    ### Edge index related
    if myargs.model == "graphormer":
        # Fix edge index: add new edges of virtual nodes
        num_virtual_tokens = 1
        edge_indexes = [fix_edge_index(edge_indexes[idx], num_nodes[idx]) for idx in range(len(edge_indexes))]
        
    elif myargs.model == "gt":
        num_virtual_tokens = 0
 
    if sequence_parallel_is_initialized():
        seq_parallel_world_size = get_sequence_parallel_world_size()
        x_i_list = [t for t in torch.tensor_split(x, seq_parallel_world_size, dim=1)]
        sub_split_seq_lens = [t.shape[1] for t in x_i_list] # [9, 9, 9, 8]
        total_real_seq_len = (max(sub_split_seq_lens) + num_virtual_tokens) * seq_parallel_world_size # 10 * 4 = 40
        
        if num_virtual_tokens > 0:
            edge_indexes = [adjust_edge_index_nomerge(i, max(sub_split_seq_lens)) for i in edge_indexes]
        edge_index = flatten_edge_index(edge_indexes, total_real_seq_len)

    else:
        sub_split_seq_lens = None
        total_real_seq_len = max_node_num + num_virtual_tokens
        edge_index = flatten_edge_index(edge_indexes, total_real_seq_len)
    
        
    # print(f"N: {torch.max(graph_node_num)} {t10-t1}s")
    # print(f"N: {torch.max(graph_node_num)} {t2-t1}s, {t3-t2}s, {t4-t3}s, {t5-t4}s, {t6-t5}s, {t7-t6}s, {t8-t7}s, {t10-t1}s")
    # print(f"N: {torch.max(graph_node_num)} {t10-t4}s, {t11-t10}s, {t12-t11}s, {t5-t12}s")

    return Batch(
        idx=torch.LongTensor(idxs), # [bs]
        attn_bias=None, # [bs, N_max+1, N_max+1] torch.float32
        # attn_edge_type=attn_edge_type,  # [bs, N_max, N_max, edge_attr.size(-1)] torch.int64
        # spatial_pos=spatial_pos,  # [bs, N_max, N_max] torch.int64
        in_degree=in_degree, # [bs, N_max] torch.int64
        out_degree=out_degree, # [bs, N_max] torch.int64
        x=x, # [bs, N_max, 1] torch.int64
        # edge_input=edge_input, # [bs, N_max, N_max, max_dist, edge_attr.size(-1)] torch.int64
        y=y,  # [bs] torch.float32
        # adj=adj,
        graph_node_num=graph_node_num, # [bs]
        # edge_attr=edge_attr, # [bs, edge_num, edge_attr_dim] 
        edge_index=edge_index, # [bs, 2, edge_num] / [2, total_edges]
        sub_split_seq_lens=sub_split_seq_lens, # [seq_world_size]
    )