import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import random
from typing import Any, Tuple, List
from functools import partial
from torch import Tensor
import time
import sys
import os
import dgl
import copy
import contextlib
import networkx as nx
import itertools
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph
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


def adjust_edge_index_nomerge(edge_index, sub_seq_len):
    new_index = edge_index.clone()
    mask = edge_index > 0
    new_index[mask] = edge_index[mask] + ((edge_index[mask].float() - 1) // sub_seq_len).to(torch.int64)

    return new_index


def pad_y(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.full((padlen, ), -100, dtype=x.dtype, device=x.device)
        new_x[:xlen] = x
        x = new_x
    # x = torch.cat([x, torch.full((addlen, ), -100, dtype=x.dtype, device=x.device)], dim=0)
    return x


def pad_2d(x, padlen):
    # xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    # ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    # ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    # indexes = torch.stack(indexes, dim=0)

    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x


def pad_attn_bias(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    xlen = x.size(0)
    new_x = x.new_zeros([padlen, padlen, x.size(2)], dtype=x.dtype)
    new_x[:xlen, :xlen, :] = x
    x = new_x
    return x


def pad_2d_bs(x, padlen):
    bs, xlen = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([bs, padlen], dtype=x.dtype)
        new_x[:, :xlen] = x
        x = new_x
    return x


def pad_x_bs(x, padlen):
    bs, xlen2, xlen3 = x.size()
    if xlen2 < padlen:
        new_x = x.new_zeros([bs, padlen, xlen3], dtype=x.dtype)
        new_x[:, :xlen2, :] = x
        x = new_x
    return x


def pad_3d_bs(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3 = x.size()
    new_x = x.new_zeros([bs, padlen, padlen*seq_parallel_world_size], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3] = x
    x = new_x
    return x


def pad_4d_bs(x, padlen):
    bs, xlen2, xlen3, xlen4 = x.size()
    if xlen2 < padlen:
        new_x = x.new_zeros([bs, padlen, xlen3, xlen4], dtype=x.dtype)
        new_x[:, :xlen2, :, :] = x
        x = new_x
    return x


def pad_5d_bs(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3, xlen4, xlen5 = x.size()
    new_x = x.new_zeros([bs, padlen, padlen*seq_parallel_world_size, xlen4, xlen5], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3, :, :] = x
    x = new_x
    return x


def pad_attn_bias_bs(x, padlen): 
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3 = x.size()
    new_x = x.new_zeros([bs, padlen, (padlen-1)*seq_parallel_world_size+1], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3] = x
    x = new_x
    return x


def pad_attn_bias_bs_unsplit(x, padlen, graph_node_num):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3 = x.size()
    if xlen2 < padlen*seq_parallel_world_size+1:
        # new_x = x.new_zeros([bs, padlen*seq_parallel_world_size+1, padlen*seq_parallel_world_size+1], dtype=x.dtype)
        # new_x[:, :xlen2, :xlen3] = x
        # x = new_x

        # Pad "-inf"
        new_x = x.new_zeros([bs, padlen*seq_parallel_world_size+1, padlen*seq_parallel_world_size+1], dtype=x.dtype).fill_(float("-inf"))
        new_x[:, :xlen2, :xlen3] = x
        
        for i in range(graph_node_num.size(0)):
            new_x[i, xlen2:, :graph_node_num[i]] = 0
        x = new_x
        # if get_sequence_parallel_rank() == 0:
        #     print(x[3, :, :])
        # exit(0)
    return x


def pad_spatial_pos_bs_unsplit(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3 = x.size()
    new_x = x.new_zeros([bs, padlen*seq_parallel_world_size, padlen*seq_parallel_world_size], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3] = x
    x = new_x
    return x


def pad_edge_input_bs_unsplit(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3, xlen4, xlen5 = x.size()
    new_x = x.new_zeros([bs, padlen*seq_parallel_world_size, padlen*seq_parallel_world_size, xlen4, xlen5], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3, :, :] = x
    x = new_x
    return x


def random_split_idx(data_y, frac_train, frac_valid, frac_test, seed):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    random.seed(seed)
    all_idx = np.arange(data_y.shape[0])
    random.shuffle(all_idx)
    train_idx = all_idx[:int(frac_train * data_y.shape[0])]
    val_idx = all_idx[int(frac_train * data_y.shape[0]):int((frac_train+frac_valid) * data_y.shape[0])]
    test_idx = all_idx[int((frac_train+frac_valid) * data_y.shape[0]):]
    split_idx = {'train': torch.tensor(train_idx),
                'valid': torch.tensor(val_idx),
                'test': torch.tensor(test_idx)}
    # print(f"Train nodes: {len(train_idx)}, Valid nodes: {len(val_idx)}, Test nodes: {len(test_idx)}")
    return split_idx


def gen_sub_edge_index(edge_index, idx_batch, N):
    """
    Get sub edge_index according to given sequence nodes
        
    Arguments:
        edge_index (Tensor): original edge_index of the whole graph
        idx_batch (Tensor): training node indexes of a batch
        N (Int): number of nodes in the whole graph
    """
    adj, _ = remove_self_loops(edge_index)
    adj, _ = add_self_loops(adj, num_nodes=N)
    edge_index_i, _ = subgraph(idx_batch, adj, num_nodes=N, relabel_nodes=True)

    return edge_index_i


def create_pairs(N, M, off_N, off_M):
    """Create a list of pairs (a, b) where a ranges from 1 to N and b ranges from 1 to M."""
    return [(a, b) for a in range(off_N, off_N + N) for b in range(off_M, off_M + M)]


def generate_new_edges_optimized(edge_index, k, partition_ids, p, blocksize, new_id_mapping):
    # edge_counts = np.zeros((k, k), dtype=np.int64)
    new_edges = []
    src, dst = edge_index
    edge_partiton = (partition_ids.numpy()[src.numpy()], partition_ids.numpy()[dst.numpy()])
    edge_index_np = (src.numpy(), dst.numpy())
    # Combining the first and second rows of a for sorting
    combined_a = list(zip(edge_partiton[0], edge_partiton[1]))
    # Sorting combined_a and also sorting b based on the sorting of combined_a
    sorted_combined_a, sorted_b = zip(*sorted(zip(combined_a, zip(*edge_index_np))))
    # Unzipping the sorted lists
    sorted_a1, sorted_a2 = zip(*sorted_combined_a)
    sorted_b1, sorted_b2 = zip(*sorted_b)

    edge_count_list = np.zeros(k*k, dtype=np.int64)
    change_count = 0
    for i in range(len(sorted_a1)-1):
        if (sorted_a1[i] == sorted_a1[i+1]) and (sorted_a2[i] == sorted_a2[i+1]):
            edge_count_list[change_count] += 1
        else:
            edge_count_list[change_count] += 1
            change_count += 1
    edge_count_list[change_count] += 1
    cumulative_sum = [sum(edge_count_list[:i]) for i in range(len(edge_count_list) + 1)]

    # # Converting the sorted tuples back to lists
    # sorted_a_new = (list(sorted_a1), list(sorted_a2))
    # sorted_b_new = (list(sorted_b1), list(sorted_b2))
    
    # for src, dst in zip(*edge_index):
    #     src_part = partition_ids[src]
    #     dst_part = partition_ids[dst]
    #     edge_counts[src_part, dst_part] += 1

    node_counts = np.zeros(k, dtype=int)
    total_node_offset = np.zeros(k, dtype=int)
    for pid in partition_ids:
        node_counts[pid] += 1

    for i in range(k-1):
        total_node_offset[i+1] = total_node_offset[i] + node_counts[i]
    # print(total_node_offset)
    # node_counts = np.array([np.sum(partition_ids == i) for i in range(k)])
    # print(node_counts)
    keep_cnt = 0
    for i in range(k):
        for j in range(k):
            sparsity = edge_count_list[i*k+j] / (node_counts[i] * node_counts[j])
            # if i == j:
            # print(f"chunk sparsity: {sparsity:.8f}, {sparsity/8.7988461e-05}")

            if sparsity > p:
                raw_edge_src = sorted_b1[cumulative_sum[i*k+j]:cumulative_sum[i*k+j+1]]
                raw_edge_dst = sorted_b2[cumulative_sum[i*k+j]:cumulative_sum[i*k+j+1]]
                # print(len(raw_edge_src))
                for jj in range(len(raw_edge_src)):
                    new_edges.append((new_id_mapping[raw_edge_src[jj]], new_id_mapping[raw_edge_dst[jj]]))
                # for src, dst in zip(*edge_index):
                #     if partition_ids[src] == i and partition_ids[dst] == j:
                #         new_edges.append((new_id_mapping[src], new_id_mapping[dst]))
                #         # new_edges.append((src, dst))
            else:
                # mask = torch.zeros(node_counts[i], node_counts[j])
                num_elements_per_block = blocksize * blocksize

                total_elements = node_counts[i] * node_counts[j]
                num_nonzero_blocks = int(sparsity * total_elements / num_elements_per_block) + 1
                np.random.seed(keep_cnt)
                if (node_counts[i] >= blocksize) and (node_counts[j] >= blocksize):
                    for ii in range(num_nonzero_blocks):
                        block_row = (np.random.randint(0, node_counts[i] // blocksize)) #% (node_counts[i] // blocksize)
                        # print(ii, block_row)
                        block_col = (np.random.randint(0, node_counts[j] // blocksize)) #% (node_counts[j] // blocksize)
                        # print(ii, keep_cnt, block_row, block_col)
                        keep_cnt += 1
                        random_edge = create_pairs(blocksize, blocksize, (block_row * blocksize + total_node_offset[i]), (block_col * blocksize + total_node_offset[j]))
                        new_edges.extend(random_edge)


    # print(keep_cnt)
    # print("Edge raw {} new {}".format(len(edge_index[0]), len(new_edges)))
    # print(new_edges)
    return torch.tensor(new_edges).t()


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def reformat_graph(edge_index, k, block_size, beta_coeffi='1'):

    src, dst = edge_index
    g = dgl.graph((src, dst))
    # logging.getLogger('dgl').setLevel(logging.WARNING)
    t0 = time.time()
    with suppress_stdout():
        partition_ids = dgl.metis_partition_assignment(g, k)
    # exit(0)
    t1 = time.time()
    new_id_mapping = np.empty(g.num_nodes(), dtype=np.int64)
    # print(x.shape)
    current_id = 0
    for part_id in range(k):
        nodes_in_part = np.where(partition_ids == part_id)[0]
        new_id_mapping[nodes_in_part] = np.arange(current_id, current_id + len(nodes_in_part))
        current_id += len(nodes_in_part)
        
    t2 = time.time()

    p_ori = len(edge_index[0]) / (g.num_nodes() * g.num_nodes()) # 1-sparsity
    if beta_coeffi == '1':
        p = 1
    else:
        p = beta_coeffi * p_ori
    p = 1
    
    new_edge_index = generate_new_edges_optimized(edge_index, k, partition_ids, p, blocksize=block_size, new_id_mapping=new_id_mapping)
    # new_edge_index = (new_id_mapping[src.numpy()], new_id_mapping[dst.numpy()])
    # new_edge_index = (new_id_mapping[block_src.numpy()], new_id_mapping[block_dst.numpy()])
    t3 = time.time()
    new_id_mapping_tensor = torch.from_numpy(new_id_mapping)

    sorted_indices = torch.argsort(new_id_mapping_tensor)

    sorted_indices_edge = torch.argsort(new_edge_index[0])

    sorted_edge_index = new_edge_index[:, sorted_indices_edge]
    t4 = time.time()
    # print("Time in reorder {} {} {}".format(t1-t0, t2-t1, t3-t2))
    return sorted_edge_index, sorted_indices


def get_batch(args, x, y, idx_batch, adjs, rest_split_sizes, device):
    """Generate a local subsequence in sequence parallel
    """

    # For sequence parallel
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    seq_length = args.seq_len

    assert seq_length % seq_parallel_world_size == 0
    sub_seq_length = seq_length // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_length
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

    x_i = x[idx_batch]
    y_i = y[idx_batch]
    attn_bias = torch.cat([torch.tensor(i[idx_batch.cpu(), :][:, idx_batch.cpu()].toarray(), dtype=torch.float32).unsqueeze(0) for i in adjs])
    attn_bias = attn_bias.permute(1, 2, 0) # [s, s, d]

    if idx_batch.shape[0] < seq_length:
        
        assert rest_split_sizes is not None, 'split_sizes should not be None'
        x_i_list = [t for t in torch.split(x_i, rest_split_sizes, dim=0)]
        y_i_list = [t for t in torch.split(y_i, rest_split_sizes, dim=0)]
        attn_bias_list = [t for t in torch.split(attn_bias, rest_split_sizes, dim=0)]

        padlen = max(rest_split_sizes)
        x_i_list_pad = []
        y_i_list_pad = []
        attn_bias_list_pad = []
        for i in range(len(x_i_list)):
            x_i_list_pad.append(pad_2d(x_i_list[i], padlen))
            y_i_list_pad.append(pad_y(y_i_list[i], padlen))
            attn_bias_list_pad.append(pad_attn_bias(attn_bias_list[i], padlen))
        last_batch_flag(True)
        
        return x_i_list_pad[seq_parallel_world_rank].to(device), y_i_list_pad[seq_parallel_world_rank].to(device), attn_bias_list_pad[seq_parallel_world_rank].to(device)
    
    else:
        x_i = x_i[sub_seq_start:sub_seq_end, :].to(device)
        y_i = y_i[sub_seq_start:sub_seq_end].to(device)
        attn_bias = attn_bias[sub_seq_start:sub_seq_end, :, :].to(device) 
        last_batch_flag(False)
        
        return x_i, y_i, attn_bias
    

def get_batch_reorder_blockize(args, x, y, idx_batch, rest_split_sizes, device, edge_index, N, k, block_size, beta_coeffi='1'):
    """
    Dummy bias for faster processing time each iteration
    Generate a local subsequence in sequence parallel
    Get sub edge_index according to sequence length
    """
     # For sequence parallel
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()
    seq_length = args.seq_len

    x_i = x[idx_batch] # [s, x_d]
    y_i = y[idx_batch] # [s]

    # Get sub edge_index according to current sequence nodes
    edge_index_i_raw = gen_sub_edge_index(edge_index, idx_batch, N) 
    
    if args.model == "graphormer":
        # Fix edge index: add new edges of virtual nodes
        edge_index_i_raw = fix_edge_index(edge_index_i_raw, idx_batch.shape[0])
    
    # Broadcast the reordered edges & sorted indices to all ranks
    if args.reorder:
        if args.rank == 0:
            edge_index_i, sorted_indices = reformat_graph(edge_index_i_raw, k, block_size, beta_coeffi)
            
            sizes_broad = torch.LongTensor([edge_index_i.shape[1], sorted_indices.shape[0]]).to(device)
        else:
            sizes_broad = torch.empty(2, dtype=torch.int64, device=device)
        dist.barrier()
        dist.broadcast(sizes_broad, src_rank, group=group)
        
        if args.rank == 0:
            edge_index_i_broad = edge_index_i.to(device)
            sorted_indices_broad = sorted_indices.to(device)
        else:
            shape = sizes_broad.tolist()
            edge_index_i_broad = torch.empty((2, shape[0]),
                                    device=device,
                                    dtype=torch.int64)
            sorted_indices_broad = torch.empty((shape[1]),
                                    device=device,
                                    dtype=torch.int64)
        dist.broadcast(edge_index_i_broad, src_rank, group=group)
        dist.broadcast(sorted_indices_broad, src_rank, group=group)
        edge_index_i = edge_index_i_broad.to("cpu")
        sorted_indices = sorted_indices_broad.to("cpu")
        # dist.barrier()
        
        # Remap x, y, attn_bias tensors according to the sorted indices
        if args.model == "graphormer":
            sorted_indices = sorted_indices[sorted_indices != 0]
            sorted_indices = sorted_indices - 1
            attn_bias = None
            # attn_bias = torch.zeros(idx_batch.shape[0], idx_batch.shape[0], args.attn_bias_dim, dtype=torch.float32) # For quicker experiments, use dummy attn bias
            # attn_bias = torch.index_select(attn_bias, 0, sorted_indices)
            # attn_bias = torch.index_select(attn_bias, 1, sorted_indices)
        else:
            attn_bias = None
        x_i = torch.index_select(x_i, 0, sorted_indices)
        y_i = torch.index_select(y_i, 0, sorted_indices)
    else:
        edge_index_i = edge_index_i_raw
    

    if idx_batch.shape[0] < seq_length:
        assert rest_split_sizes is not None, 'Rest split_sizes should not be None'
        seq_length = max(rest_split_sizes) * seq_parallel_world_size # 14 * 4 = 56 
        x_i = pad_2d(x_i, seq_length)
        y_i = pad_y(y_i, seq_length)

        afterpad_split_sizes = [max(rest_split_sizes)] * seq_parallel_world_size
        x_i_list = [t for t in torch.split(x_i, afterpad_split_sizes, dim=0)]
        y_i_list = [t for t in torch.split(y_i, afterpad_split_sizes, dim=0)]

        if args.model == "graphormer":
            # Adjust edge index values w/o merging global token after all2all 
            edge_index_i = adjust_edge_index_nomerge(edge_index_i, max(rest_split_sizes))   

        x_i = x_i_list[seq_parallel_world_rank]
        y_i = y_i_list[seq_parallel_world_rank]
        edge_index_i = edge_index_i
        
        if attn_bias is not None:
            attn_bias = pad_attn_bias(attn_bias, seq_length)
            attn_bias_list = [t for t in torch.split(attn_bias, afterpad_split_sizes, dim=0)]
            attn_bias = attn_bias_list[seq_parallel_world_rank] # [s/p, s, d]
        
        last_batch_flag(True)
        
    else:
        assert seq_length % seq_parallel_world_size == 0
        sub_seq_length = seq_length // seq_parallel_world_size
        sub_seq_start = seq_parallel_world_rank * sub_seq_length
        sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length
        
        x_i = x_i[sub_seq_start:sub_seq_end, :]
        y_i = y_i[sub_seq_start:sub_seq_end]
        
        if attn_bias is not None:
            attn_bias = attn_bias[sub_seq_start:sub_seq_end, :, :] 

        if args.model == "graphormer":
            # Adjust edge index values w/o merging global token after all2all
            edge_index_i = adjust_edge_index_nomerge(edge_index_i, sub_seq_length)
            
        last_batch_flag(False)

    return (x_i, y_i, edge_index_i, attn_bias)


def get_batch_papers100m(args, x, y, idx_batch, attn_bias, rest_split_sizes, device, edge_index, N):
    """
    Dummy bias for faster processing time each iteration
    Generate a local subsequence in sequence parallel
    Get sub edge_index according to sequence length
    """
    # For sequence parallel
    seq_length = args.seq_len
    sub_seq_length = seq_length // args.world_size

    x_i = x[idx_batch] # [s, x_d]
    y_i = y[idx_batch] # [s]

    # Get sub edge_index according to given sequence nodes
    edge_index_i = gen_sub_edge_index(edge_index, idx_batch, N) # NOTE: make sure all rank share the same edge_index_i
    
    if args.model == "graphormer":
        # Fix edge index: add new edges of virtual nodes
        edge_index_i = fix_edge_index(edge_index_i, idx_batch.shape[0])

    if args.reorder:
        k, block_size = 8, 16 # number of partitions
        edge_index_i, sorted_indices = reformat_graph(edge_index_i, k, block_size)
        
        if args.model == "graphormer":
            sorted_indices = sorted_indices[sorted_indices != 0]
            sorted_indices = sorted_indices - 1

        x_i = torch.index_select(x_i, 0, sorted_indices)
        y_i = torch.index_select(y_i, 0, sorted_indices)

    if idx_batch.shape[0] < seq_length:
        # 对于剩余的node，feature用0 pad, label用-100 pad, attn_bias用0填充，
        # 先把总的pad到新长度（为了将pad的点都放在最后一个rank），然后划分
        
        assert rest_split_sizes is not None, 'split_sizes should not be None'
        seq_length = max(rest_split_sizes) * args.world_size # 14 * 4 = 56 
        x_i = pad_2d(x_i, seq_length)
        y_i = pad_y(y_i, seq_length)

        if args.model == "graphormer":
            # Adjust edge index values w/o merging global token after all2all 
            edge_index_i = adjust_edge_index_nomerge(edge_index_i, max(rest_split_sizes))   
    else:
        if args.model == "graphormer":
            edge_index_i = adjust_edge_index_nomerge(edge_index_i, sub_seq_length)

    return (x_i, y_i, attn_bias, edge_index_i)


def get_batch_from_loader(args, batch):
    """Generate a batch of local subsequences from dataloader in sequence parallel.
    cut in seq-level: x, attn_edge_type, spatial_pos, in-degree, out-degree, edge_input.
    attn_bias, edge_index copy in each rank.
    Set global token indices for each batch.
    """  
    #### For sequence parallel
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    
    seq_length = batch.x.size(1)

    #### Split input data to each rank: if attn_bias use all2all, need to split attn_bias, spatial_pos, edge_input
    sub_split_seq_lens = batch.sub_split_seq_lens # [9, 9, 9, 8]
    # if args.rank == 0:
    #     print(f'This batch seq len: {seq_length}, sub seq len: {sub_seq_lengths}')

    x_i_list = [t for t in torch.split(batch.x, sub_split_seq_lens, dim=1)]
    in_degree_list = [t for t in torch.split(batch.in_degree, sub_split_seq_lens, dim=1)]
    out_degree_list = [t for t in torch.split(batch.out_degree, sub_split_seq_lens, dim=1)]
    
    # # Attn_bias all2all need
    # spatial_pos_list = [t for t in torch.split(batch.spatial_pos, sub_split_seq_lens, dim=1)]
    # edge_input_list = [t for t in torch.split(batch.edge_input, sub_split_seq_lens, dim=1)]
    # attn_biases = [t for t in torch.split(batch.attn_bias[:, 1:, :], sub_split_seq_lens, dim=1)]
    # attn_bias_list = [torch.cat([torch.index_select(batch.attn_bias, 1, torch.LongTensor([0])), t], dim=1) for t in attn_biases] # add glotal token
    # attn_edge_type_list = [t for t in torch.split(batch.attn_edge_type, sub_seq_lengths, dim=1)]
    
    #### Pad cut data to the same sub seq length. e.g., [9, 9, 9, 8] -> [9, 9, 9, 9]
    padlen = max(sub_split_seq_lens)
    sub_real_seq_len = padlen + args.num_global_node
    
    x_i_list_pad = [pad_x_bs(t, padlen) for t in x_i_list]
    in_degree_list_pad = [pad_2d_bs(t, padlen) for t in in_degree_list]
    out_degree_list_pad = [pad_2d_bs(t, padlen) for t in out_degree_list]
    # attn_edge_type_list_pad = [pad_4d_bs(t, padlen) for t in attn_edge_type_list]
    # spatial_pos_list_pad = [pad_3d_bs(t, padlen) for t in spatial_pos_list]
    # edge_input_list_pad = [pad_5d_bs(t, padlen) for t in edge_input_list]
    # attn_bias_list_pad = [pad_attn_bias_bs(t, sub_real_seq_len) for t in attn_bias_list]

    batch.x = x_i_list_pad[seq_parallel_world_rank] # [bs, padlen, 1]
    batch.in_degree = in_degree_list_pad[seq_parallel_world_rank]
    batch.out_degree = out_degree_list_pad[seq_parallel_world_rank]
    # batch.attn_edge_type = attn_edge_type_list_pad[seq_parallel_world_rank]
    # batch.spatial_pos = spatial_pos_list_pad[seq_parallel_world_rank]
    # batch.edge_input = edge_input_list_pad[seq_parallel_world_rank]
    # batch.attn_bias = attn_bias_list_pad[seq_parallel_world_rank]
    # print(f"{batch.spatial_pos.size()} {batch}")
    if args.dummy_bias:
        batch.attn_bias = None

    #### Unsplit data
    # batch.attn_bias = pad_attn_bias_bs_unsplit(batch.attn_bias, padlen, batch.graph_node_num)
    # batch.spatial_pos = pad_spatial_pos_bs_unsplit(batch.spatial_pos, padlen)
    # batch.edge_input = pad_edge_input_bs_unsplit(batch.edge_input, padlen)

    # Set global token indices for each batch, in graphormer global token idx: 0
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
    # set_global_token_indices(global_token_indices)
    batch.global_token_indices = global_token_indices

    del batch.sub_split_seq_lens      
    
    
def get_batch_from_loader_malnet(args, batch):
    """Generate a batch of local subsequences from dataloader in sequence parallel.
    cut in seq-level: x, attn_edge_type, spatial_pos, in-degree, out-degree, edge_input.
    attn_bias, edge_index copy in each rank.
    Set global token indices for each batch.
    """  
    #### For sequence parallel
    seq_parallel_world_size = args.world_size
    seq_length = batch.x.size(1)

    sub_split_seq_lens = batch.sub_split_seq_lens # [9, 9, 9, 8]
    x_i_list = [t for t in torch.split(batch.x, sub_split_seq_lens, dim=1)]
    in_degree_list = [t for t in torch.split(batch.in_degree, sub_split_seq_lens, dim=1)]
    out_degree_list = [t for t in torch.split(batch.out_degree, sub_split_seq_lens, dim=1)]
    
    #### Pad cut data to the same sub seq length. e.g., [9, 9, 9, 8] -> [9, 9, 9, 9]
    padlen = max(sub_split_seq_lens)
    sub_real_seq_len = padlen + args.num_global_node
    
    x_i_list_pad = [pad_x_bs(t, padlen) for t in x_i_list]
    in_degree_list_pad = [pad_2d_bs(t, padlen) for t in in_degree_list]
    out_degree_list_pad = [pad_2d_bs(t, padlen) for t in out_degree_list]
    # attn_edge_type_list_pad = [pad_4d_bs(t, padlen) for t in attn_edge_type_list]
    # spatial_pos_list_pad = [pad_3d_bs(t, padlen) for t in spatial_pos_list]
    # edge_input_list_pad = [pad_5d_bs(t, padlen) for t in edge_input_list]
    # attn_bias_list_pad = [pad_attn_bias_bs(t, sub_real_seq_len) for t in attn_bias_list]

    batch.x = x_i_list_pad # [bs, padlen, 1]
    batch.in_degree = in_degree_list_pad
    batch.out_degree = out_degree_list_pad
    # batch.attn_edge_type = attn_edge_type_list_pad[seq_parallel_world_rank]
    # batch.spatial_pos = spatial_pos_list_pad[seq_parallel_world_rank]
    # batch.edge_input = edge_input_list_pad[seq_parallel_world_rank]
    # batch.attn_bias = attn_bias_list_pad[seq_parallel_world_rank]
    # print(f"{batch.spatial_pos.size()} {batch}")
    if args.dummy_bias:
        batch.attn_bias = None

    #### Unsplit data
    # batch.attn_bias = pad_attn_bias_bs_unsplit(batch.attn_bias, padlen, batch.graph_node_num)
    # batch.spatial_pos = pad_spatial_pos_bs_unsplit(batch.spatial_pos, padlen)
    # batch.edge_input = pad_edge_input_bs_unsplit(batch.edge_input, padlen)

    # Set global token indices for each batch, in graphormer global token idx: 0
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
    # set_global_token_indices(global_token_indices)
    batch.global_token_indices = global_token_indices
    
    return (x_i_list_pad, in_degree_list_pad, out_degree_list_pad, global_token_indices)
    

def split_tensor_along_second_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its second dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # global_token_num = get_global_token_num()
    
    # Get the size and dimension.
    split_dim = 1
    split_dim_size = tensor.size()[split_dim] // num_partitions
        
    # Split.
    tensor_list = torch.split(tensor, split_dim_size, dim=split_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def merge_global_token(x_layer: Tensor, merge_dim: int) -> Tensor:
    """Merge each rank's global token embedding into one for k, q, v

    Arguments:
        x_layer (Tensor): input tensor

    Returns:    
        * output (Tensor): merged output tensor
    """
    # TODO consider multiple global tokens case; maybe slow?
    if not get_last_batch_flag():
        global_token_indices = get_global_token_indices(last_batch=False)
    else:
        global_token_indices = get_global_token_indices(last_batch=True)
        
    avg_global_token = torch.mean(torch.index_select(x_layer, merge_dim, 
                                                       torch.LongTensor(global_token_indices).to(x_layer.device)), dim=merge_dim, keepdim=True)
    x_layer = torch.index_select(x_layer, merge_dim, 
                                 torch.LongTensor([i for i in range(x_layer.size(merge_dim)) if i not in global_token_indices]).to(x_layer.device))
    
    x_layer = torch.cat([x_layer, avg_global_token], dim=merge_dim).contiguous()
    return x_layer


def merge_global_token0(x_layer: Tensor, merge_dim: int) -> Tensor:
    """Merge each rank's global token embedding into one for k, q, v

    Arguments:
        x_layer (Tensor): input tensor

    Returns:    
        * output (Tensor): merged output tensor
    """
    # TODO consider multiple global tokens case; maybe slow?
    if not get_last_batch_flag():
        global_token_indices = get_global_token_indices(last_batch=False)
    else:
        global_token_indices = get_global_token_indices(last_batch=True)
        
    # if get_sequence_parallel_rank() == 0:
    #     a = torch.index_select(x_layer, merge_dim, torch.LongTensor(global_token_indices).to(x_layer.device))
    #     print(a.view(4, 4, -1))

    # [bs, 1, np, hn]
    avg_global_token = torch.mean(torch.index_select(x_layer, merge_dim, 
                                                       torch.LongTensor(global_token_indices).to(x_layer.device)), dim=merge_dim, keepdim=True)

    x_layer = torch.index_select(x_layer, merge_dim, 
                                 torch.LongTensor([i for i in range(x_layer.size(merge_dim)) if i not in global_token_indices]).to(x_layer.device))
    x_layer = torch.cat([avg_global_token, x_layer], dim=merge_dim).contiguous()
    return x_layer


def extend_global_token(x_layer: Tensor, extend_dim: int) -> Tensor:
    """Extend global token embedding to each rank

    Arguments:
        x_layer (Tensor): input tensor

    Returns:
        * output (Tensor): output tensor
    """
    # TODO consider multiple global tokens case
    global_token_num = get_global_token_num()
    seq_world_size = get_sequence_parallel_world_size()
    
    x_layer_list = split_tensor_along_second_dim(x_layer, seq_world_size)
        
    # Split.
    output_list = [torch.cat([x_layer_list[i], x_layer_list[-1]], dim=extend_dim).contiguous() for i in range(seq_world_size)]
    
    return torch.cat(output_list, dim=extend_dim).contiguous()


def extend_global_token0(x_layer: Tensor, extend_dim: int) -> Tensor:
    """Extend global token embedding to each rank

    Arguments:
        x_layer (Tensor): input tensor

    Returns:
        * output (Tensor): output tensor
    """
    # TODO consider multiple global tokens case
    # x_layer: [b, s+1, hp] 
    global_token_num = get_global_token_num()
    seq_world_size = get_sequence_parallel_world_size()
    
    assert (x_layer.size(extend_dim) - 1) % seq_world_size == 0
    split_sizes = [1] + [(x_layer.size(extend_dim) - 1) // seq_world_size] * seq_world_size

    # Split.
    tensor_list = torch.split(x_layer, split_sizes, dim=extend_dim)    
    output_list = [torch.cat([tensor_list[0], tensor_list[i]], dim=extend_dim).contiguous() for i in range(1, seq_world_size+1)]
    
    return torch.cat(output_list, dim=extend_dim).contiguous()


def copy_global_token0(x_layer: Tensor, extend_dim: int) -> Tensor:
    """copy global token embedding to each rank

    Arguments:
        x_layer (Tensor): input tensor

    Returns:
        * output (Tensor): output tensor
    """
    # x_layer: [b, s+p, hp] 
    # global_token_num = get_global_token_num()
    seq_world_size = get_sequence_parallel_world_size()

    if not get_last_batch_flag():
        global_token_indices = get_global_token_indices(last_batch=False)
    else:
        global_token_indices = get_global_token_indices(last_batch=True)

    assert x_layer.size(extend_dim) % seq_world_size == 0
    x_layer[:, torch.LongTensor(global_token_indices), :] = x_layer[:, 0, :].unsqueeze(1).repeat(1, len(global_token_indices), 1)
    
    return x_layer.contiguous()


def broadcast_data(args, dataset_train, batch, device):
    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()
    group = get_sequence_parallel_group()
    src_rank = get_sequence_parallel_src_rank()

    # Pack on rank zero
    batch_idx = batch.idx

    if rank == 0:
        flatten_idx = batch_idx.to('cuda')
    else:
        total_numel = batch_idx.numel()
        flatten_idx = torch.empty(total_numel,
                                device=device,
                                dtype=torch.int64)
    # Broadcast
    dist.broadcast(flatten_idx, src_rank, group=group)


    # sampler = SubsetRandomSampler(flatten_idx.tolist())
    # dataloader = DataLoader(
    #             dataset_train,
    #             batch_size=args.batch_size,
    #             num_workers=args.num_workers,
    #             pin_memory=True,
    #             sampler=sampler,
    #             collate_fn=partial(
    #                 collator,
    #                 max_node=get_dataset(args.dataset)["max_node"],
    #                 multi_hop_max_dist=args.multi_hop_max_dist,
    #                 spatial_pos_max=args.spatial_pos_max,
    #             )
    # )
    # # batch = next(iter(dataloader))
    # # print(f"rank {rank} {batch.y}")
    # # for batch in dataloader:
    # #     print(f"rank {rank} {batch.y}")
    # # exit(0)
    # return next(iter(dataloader))


def calc_power_edge_index(edge_index, N, power):
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    adj_matrix = torch.sparse_coo_tensor(edge_index, values, torch.Size([N, N]))

    m_adj = adj_matrix.clone()
    for _ in range(power - 1):
        m_adj = torch.sparse.mm(m_adj, adj_matrix) 

    sparse_rate = calculate_sparsity(m_adj)

    coalesced_matrix = m_adj.coalesce()
    
    m_edge_index = coalesced_matrix.indices()
    return m_edge_index


def calculate_sparsity(sparse_matrix):
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]

    non_zero_elements = sparse_matrix._nnz()

    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity


def calculate_sparsity_csr(csr_matrix):
    non_zero = csr_matrix.nnz

    total_elements = csr_matrix.shape[0] * csr_matrix.shape[1]

    sparsity = 1 - (non_zero / total_elements)
    return sparsity


def make_strongly_connected(graph):
    if nx.is_weakly_connected(graph):
        print("The graph is already strongly connected.")
        return graph
    
    scc = list(nx.strongly_connected_components(graph))
    
    if len(scc) == 1:
        return graph

    for i in range(len(scc) - 1):
        comp_a = scc[i]
        comp_b = scc[i + 1]
        node_a = next(iter(comp_a))
        node_b = next(iter(comp_b))
        graph.add_edge(node_a, node_b)

    scc = list(nx.strongly_connected_components(graph))
    while len(scc) > 1:
        comp_a = scc[0]
        comp_b = scc[1]
        node_a = next(iter(comp_a))
        node_b = next(iter(comp_b))
        graph.add_edge(node_a, node_b)
        scc = list(nx.strongly_connected_components(graph))

    return graph


def check_conditions(edge_index, num_nodes):
    graph = nx.DiGraph()
    
    nodes = range(num_nodes)
    for node in nodes:
        graph.add_node(node)
        
    edges = list(zip(edge_index[0], edge_index[1]))
    graph.add_edges_from(edges)

    for node in graph.nodes:
        if not graph.has_edge(node, node):
            print(f"Condition 1 failed: Node {node} does not attend to itself.")
            return False

    # graph = make_strongly_connected(graph)
    if not nx.is_weakly_connected(graph):
        print("Condition 3 failed: The graph is not strongly connected.")
        return False

    print("All conditions are satisfied.")
    return True