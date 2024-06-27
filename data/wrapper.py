# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import os.path as osp
import glob
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
from torch_geometric.utils import to_dense_adj, to_networkx, remove_isolated_nodes, degree
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip)
from typing import Optional, Callable, List


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

    edge_index = edge_index.long()
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
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr.long()) + 1
    )  # [n_nodes, n_nodes, 1] for ZINC

    shortest_path_result, path = algos.floyd_warshall(
        adj_orig.numpy()
    )  # [n_nodesxn_nodes, n_nodesxn_nodes]


    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()

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

    # for i in range(N + num_virtual_tokens):
    #     for j in range(N + num_virtual_tokens):

    #         val = True if random.random() < 0.3 else False
    #         adj[i, j] = adj[i, j] or val

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos # [N, N]
    item.in_degree = adj_orig.long().sum(dim=1).view(-1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long() # [N_max, N_max, max_dist, edge_attr.size(-1)]
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


def preprocess_item_malnet_dummy_encd(item):
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

    in_degree = degree(edge_index[1], num_nodes=N)
    out_degree = degree(edge_index[0], num_nodes=N)


    # combine
    item.x = x
    item.in_degree = in_degree.long()
    item.out_degree = out_degree.long()
    # item.edge_attr = edge_attr.float()

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
            item = preprocess_item_malnet_dummy_encd(item)
            # print(f"N: {item.num_nodes} wrapper t: {time.time()-t0}s")
            
            # print()
            return item
        else:
            return self.index_select(idx)


class MalNet(InMemoryDataset):
    r"""The MalNet Tiny dataset from the
    `"A Large-Scale Database for Graph Representation Learning"
    <https://openreview.net/pdf?id=1xDTDk3XPW>`_ paper.
    :class:`MalNet` contains 5,000 malicious and benign software function
    call graphs across 5 different types. Each graph contains at most 5k nodes.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'http://malnet.cc.gatech.edu/graph-data/malnet-graphs-tiny.tar.gz'
    # 70/10/20 train, val, test split by type
    split_url = 'http://malnet.cc.gatech.edu/split-info/split_info_tiny.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # print("haha")
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self) -> List[str]:
        # Get the current directory
        # current_directory = os.getcwd()
        current_directory = self.root
        print(current_directory)
        # Get all entries in the directory
        folders = os.listdir(osp.join(current_directory, 'raw'))
        # print(directory_entries)
        # # Filter out the files, keeping only directories
        # folders = [entry for entry in directory_entries if os.path.isdir(os.path.join(current_directory, entry))]

        # folders = ['']
        # print(folders)
        return [osp.join('./', folder) for folder in folders]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_dict.pt']

    def download(self):
        return

    def process(self):
        print("start processing")
        data_list = []
        split_dict = {'train': [], 'valid': [], 'test': []}

        parse = lambda f: set([x.split('/')[-1]
                               for x in f.read().split('\n')[:-1]])  # -1 for empty line at EOF
        split_dir = osp.join(self.raw_dir + '/../')
        print(split_dir)
    
        with open(osp.join(split_dir, 'train.txt'), 'r') as f:
            train_names = parse(f)
        with open(osp.join(split_dir, 'val.txt'), 'r') as f:
            val_names = parse(f)
        with open(osp.join(split_dir, 'test.txt'), 'r') as f:
            test_names = parse(f)
        print("number of training samples: ")
        print(len(train_names))
        print("number of validation samples: ")
        print(len(val_names))
        print("number of testing samples: ")
        print(len(test_names))
        index = 0
        graph_count = 0
        for y, raw_path in enumerate(self.raw_paths):
            if index % 1 == 0:
                print("Processing File {} {}".format(index, raw_path))
            index = index + 1
            # raw_path = osp.join(raw_path, os.listdir(raw_path)[0])
            directories = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
            filenames = []
            for directory in directories:
                # 构建每个子文件夹的完整路径
                dir_path = os.path.join(raw_path, directory)

                # 在每个子文件夹中搜索 .edgelist 文件
                temp_filenames = glob.glob(os.path.join(dir_path, '*.edgelist'))

                # 添加找到的文件到总列表
                filenames.extend(temp_filenames)
            # filenames = glob.glob(osp.join(raw_path, '*.edgelist'))
            print(directories)
            for filename in filenames:
                with open(filename, 'r') as f:
                    edges = f.read().split('\n')[5:-1]
                edge_index = [[int(s) for s in edge.split()] for edge in edges]
                edge_index = torch.tensor(edge_index).t().contiguous()
                # Remove isolated nodes, including those with only a self-loop
                edge_index = remove_isolated_nodes(edge_index)[0]
                num_nodes = int(edge_index.max()) + 1
                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)
                graph_count += 1
                # print(graph_count)
                ind = len(data_list) - 1
                graph_id = osp.splitext(osp.basename(filename))[0]
                if graph_id in train_names:
                    split_dict['train'].append(ind)
                elif graph_id in val_names:
                    split_dict['valid'].append(ind)
                elif graph_id in test_names:
                    split_dict['test'].append(ind)
                else:
                   raise ValueError(f'No split assignment for "{graph_id}" {filename}.')
        
        print("start pre_filtering data")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        print("start pre_transforming data")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("start saving data")
        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])
        print("finish saving data")

    def get_idx_split(self):
        return torch.load(self.processed_paths[1])
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            t0 = time.time()
            item = preprocess_item_malnet_dummy_encd(item)
            # print(f"N: {item.num_nodes} wrapper t: {time.time()-t0}s")
            
            # print()
            return item
        else:
            return self.index_select(idx)
    

class MyGNNBenchmarkDataset(torch_geometric.datasets.GNNBenchmarkDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyGNNBenchmarkDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyGNNBenchmarkDataset, self).process()
        if dist.is_initialized():
            dist.barrier()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            t0 = time.time()
            item = preprocess_item(item)
            # print(f"N: {item.num_nodes} wrapper t: {time.time()-t0}s")
            
            # print()
            return item
        else:
            return self.index_select(idx)
        

if __name__ == "__main__":
    dataset = MalNet("/home/mzhang/work/sj/", split="train")
    print(dataset[0])
    print(len(dataset))
    # print(dataset.split_idxs)
    print(dataset[0].num_node_features)
