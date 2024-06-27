# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import random
import torch
import numpy as np
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
import torch.distributed as dist
import torch_geometric.transforms as T
import time
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx


pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    num_virtual_tokens = 1
    # edge_attr: [num_edges, edge_attr_dim], edge_index: [2, num_edges], x: [num_nodes, node_attr_dim]
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    
    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)

    N = x.size(0)
    x = convert_to_single_emb(x)  # For ZINC: [n_nodes, 1]

    # node adj matrix [N, N] bool
    adj_orig = torch.zeros([N, N], dtype=torch.bool)
    adj_orig[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]

    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
    #     convert_to_single_emb(edge_attr) + 1
    # )  # [n_nodes, n_nodes, 1] for ZINC

    # shortest_path_result, path = algos.floyd_warshall(
    #     adj_orig.numpy()
    # )  # [n_nodesxn_nodes, n_nodesxn_nodes]

    # max_dist = np.amax(shortest_path_result)
    # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()

    # attn_bias = torch.zeros(
    #     [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.float
    # )  # with graph token

    adj = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.bool
    )
    adj[edge_index[0, :], edge_index[1, :]] = True

    for i in range(num_virtual_tokens):
        adj[N + i, :] = True
        adj[:, N + i] = True

    # for i in range(N + num_virtual_tokens):
    #     for j in range(N + num_virtual_tokens):

    #         val = True if random.random() < 0.3 else False
    #         adj[i, j] = adj[i, j] or val

    # combine
    item.x = x
    item.adj = adj
    # item.attn_bias = attn_bias
    # item.attn_edge_type = attn_edge_type
    # item.spatial_pos = spatial_pos # [N, N]
    item.in_degree = adj_orig.long().sum(dim=1).view(-1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1)
    # item.edge_input = torch.from_numpy(edge_input).long() # [N_max, N_max, max_dist, edge_attr.size(-1)]
    item.edge_attr = edge_attr.float() 

    return item


def preprocess_item_malnet(item):
    """
    Precompute node features(x) as MalNet originally doesn't have any node nor edge features
    """
    num_virtual_tokens = 1
    # edge_attr: [num_edges, edge_attr_dim], edge_index: [2, num_edges], x: [num_nodes, node_attr_dim]
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    
    if hasattr(item, "num_nodes"):
        N = item.num_nodes
    else:
        N = x.size(0)

    # Precompute node features x: [N, 5]
    if x is None:
        transform_func = T.LocalDegreeProfile()
        item = transform_func(item)
        x = item.x.long()
    
    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)
    
    # x = convert_to_single_emb(x)  

    # Node adj matrix: [N, N] bool
    adj_orig = torch.zeros([N, N], dtype=torch.bool)
    adj_orig[edge_index[0, :], edge_index[1, :]] = True

    # Edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )  # [n_nodes, n_nodes, 1] 
    
    t0 = time.time()
    graph: nx.DiGraph = to_networkx(item)
    shortest_paths = nx.shortest_path(graph)
    # t1 = time.time()
    # print(f"{N} node spd done: {t1-t0}s")

    distance = 20
    spatial_pos = torch.empty(N ** 2, dtype=torch.long).fill_(distance)
    shortest_path_types = torch.zeros(N ** 2, distance, dtype=torch.long) 
    edge_attr_cal = torch.zeros(N, N, dtype=torch.long)
    edge_attr_cal[edge_index[0], edge_index[1]] = edge_attr.squeeze(1)
    for i, paths in shortest_paths.items():
        for j, path in paths.items():
            if len(path) > distance:
                path = path[:distance]

            assert len(path) >= 1
            spatial_pos[i * N + j] = len(path) - 1

            if len(path) > 1 and hasattr(item, "edge_attr") and item.edge_attr is not None:
                path_attr = [
                    edge_attr_cal[path[k], path[k + 1]] for k in
                    range(len(path) - 1)  # len(path) * (num_edge_types)
                ]

                # We map each edge-encoding-distance pair to a distinct value
                # and so obtain dist * num_edge_features many encodings
                shortest_path_types[i * N + j, :len(path) - 1] = torch.tensor(
                    path_attr, dtype=torch.long)
    spatial_pos = spatial_pos.view(N, N)
    shortest_path_types = shortest_path_types.view(N, N, distance).unsqueeze(-1)
    # print(f"{N} node done: {time.time()-t0}s")

    attn_bias = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.float
    )  # with graph token

    adj = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.bool
    )
    adj[edge_index[0, :], edge_index[1, :]] = True

    for i in range(num_virtual_tokens):
        adj[N + i, :] = True
        adj[:, N + i] = True

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj_orig.long().sum(dim=1).view(-1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1)
    item.edge_input = shortest_path_types
    item.edge_attr = edge_attr.float()

    return item



class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyZINCDataset(torch_geometric.datasets.ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINCDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINCDataset, self).process()
        if dist.is_initialized():
            dist.barrier()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item) # 每个graph都经历一遍process的过程，也是在for batch in loader时被调用
        else:
            return self.index_select(idx)


class MyCoraDataset(torch_geometric.datasets.Planetoid):
    def download(self):
        super(MyCoraDataset, self).download()

    def process(self):
        super(MyCoraDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)
        

class MyMalNetTiny(torch_geometric.datasets.MalNetTiny):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMalNetTiny, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMalNetTiny, self).process()
        if dist.is_initialized():
            dist.barrier()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            t0 = time.time()
            item = preprocess_item_malnet(item)
            # print(f"N: {item.num_nodes} wrapper t: {time.time()-t0}s")
            
            # print()
            return item
        else:
            return self.index_select(idx)


