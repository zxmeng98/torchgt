import numpy as np
import torch
import os.path as osp
import pickle
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
import scipy.sparse as sp
import scipy
from numpy.linalg import inv
from torch_geometric.datasets import Planetoid, Amazon, Actor, CitationFull, Coauthor
from torch.nn.functional import normalize
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce
from tqdm import tqdm
import os
import random
import math
import pickle as pkl
from ogb.nodeproppred import NodePropPredDataset


def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def eigenvector(L):
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.tensor(EigVec[:, 1:11], dtype = torch.float32)


def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def get_dataset(dataset_name):
    print(f'Get dataset {dataset_name}...')

    dataset_dir = './dataset/'
    dataset_path = dataset_dir + dataset_name + '/edge_index.pt'
    split_seed = 0
    if not os.path.exists(f'{dataset_dir}/{dataset_name}'): 
        os.makedirs(f'{dataset_dir}/{dataset_name}')

    # if os.path.exists(dataset_path):
    #     print(f'Already downloaded. Loading {dataset_name}...')
    #     data_x = torch.load(dataset_dir + dataset_name + '/x.pt')
    #     data_y = torch.load(dataset_dir + dataset_name + '/y.pt')
    #     adj = sp.load_npz(dataset_dir + dataset_name + '/adj.npz')
        
    #     normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/normalized_adj.npz')
    #     column_normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz')
    # else: 
    if True:
        if dataset_name in ['cora', 'citeseer', 'pubmed']: 
            dataset = Planetoid(root=dataset_dir, name=dataset_name)       
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in ['dblp']:
            dataset = CitationFull(root=dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)
                
        elif dataset_name in ["CS", "Physics"]:
        # TODO: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Coauthor.html
            dataset = Coauthor(root=dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in ["Photo"]:
            dataset = Amazon(root=dataset_dir, name=dataset_name)
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)
        
        elif dataset_name in ['aminer']:
            adj = pkl.load(open(os.path.join(dataset_dir, dataset_name, "{}.adj.sp.pkl".format(dataset_name)), "rb"))
            data_x = pkl.load(
                open(os.path.join(dataset_dir, dataset_name, "{}.features.pkl".format(dataset_name)), "rb"))
            data_y = pkl.load(
                open(os.path.join(dataset_dir, dataset_name, "{}.labels.pkl".format(dataset_name)), "rb"))
            # random_state = np.random.RandomState(split_seed)
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))
            
        elif dataset_name in ['reddit']:
            adj = sp.load_npz(os.path.join(dataset_dir, dataset_name, '{}_adj.npz'.format(dataset_name)))
            data_x = np.load(os.path.join(dataset_dir, dataset_name, '{}_feat.npy'.format(dataset_name)))
            data_y = np.load(os.path.join(dataset_dir, dataset_name, '{}_labels.npy'.format(dataset_name)))
            # random_state = np.random.RandomState(split_seed)
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))

            
        elif dataset_name in ['Amazon2M']:
            adj = sp.load_npz(os.path.join(dataset_dir, dataset_name, '{}_adj.npz'.format(dataset_name)))
            data_x = np.load(os.path.join(dataset_dir, dataset_name, '{}_feat.npy'.format(dataset_name)))
            data_y = np.load(os.path.join(dataset_dir, dataset_name, '{}_labels.npy'.format(dataset_name)))
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))
            
        elif dataset_name in ['amazon']:
            dataset_dir = "/home/mzhang/data/"
            adj = sp.load_npz(os.path.join(dataset_dir, dataset_name, 'adj_full.npz'))
            data_x = np.load(os.path.join(dataset_dir, dataset_name, 'feats.npy'))
            data_y = np.load(os.path.join(dataset_dir, dataset_name, 'labels.npy'))
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))
            
        elif dataset_name in ['pokec']:
            fulldata = scipy.io.loadmat(f'/home/mzhang/work/GTNodeLevel/dataset/pokec.mat')
            edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
            
            data_x = torch.tensor(fulldata['node_feat']).float()
            label = fulldata['label'].flatten()
            data_y = torch.tensor(label, dtype=torch.long)
            
            num_nodes = data_y.shape[0]
            adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                        shape=(num_nodes, num_nodes), dtype=np.float32)
          
            
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in {"ogbn-papers100M"}:
            file_dir = '/home/mzhang/data/'
            ogb_dataset = NodePropPredDataset(name=dataset_name, root=file_dir)
            split_idx = ogb_dataset.get_idx_split()
            idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
            
            data_y = torch.as_tensor(ogb_dataset.labels).squeeze(1)
            # data_x = torch.as_tensor(ogb_dataset.graph['node_feat'])
            edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
            # num_nodes=ogb_dataset.graph['num_nodes']
            # adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            #                         shape=(num_nodes, num_nodes), dtype=np.float32)
            # normalized_adj = adj_normalize(adj)
            # column_normalized_adj = column_normalize(adj)
        
        elif dataset_name in ["ogbn-arxiv", "ogbn-products"]:
            ogb_dataset = NodePropPredDataset(name=dataset_name, root=dataset_dir)
            split_idx = ogb_dataset.get_idx_split()
            idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
            
            data_y = torch.as_tensor(ogb_dataset.labels).squeeze(1)
            data_x = torch.as_tensor(ogb_dataset.graph['node_feat'])
            edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
            num_nodes=ogb_dataset.graph['num_nodes']
            adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                    shape=(num_nodes, num_nodes), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            # column_normalized_adj = column_normalize(adj)

        # sp.save_npz(dataset_dir + dataset_name + '/adj.npz', adj)
        # sp.save_npz(dataset_dir + dataset_name + '/normalized_adj.npz', normalized_adj)
        dataset_dir = './dataset/'
        torch.save(data_x, dataset_dir + dataset_name + '/x.pt')
        torch.save(data_y, dataset_dir + dataset_name + '/y.pt')
        torch.save(edge_index, dataset_dir + dataset_name + '/edge_index.pt')
        # sp.save_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz', column_normalized_adj)


def process_data(dataset_name, k1):
    """
    Arguments:
        k1: sequence length-1 / number of sampled neighbors
    """
    print(f'Process dataset {dataset_name}...')

    dataset_dir = './dataset/'
    data_x = torch.load(dataset_dir + dataset_name + '/x.pt')
    data_y = torch.load(dataset_dir + dataset_name + '/y.pt')
    adj = sp.load_npz(dataset_dir + dataset_name + '/adj.npz')
    normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/normalized_adj.npz')
    column_normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz')
    
    # # Compute SPD, spacial pos
    # N = adj.shape[0]
    # adj_bool = torch.zeros([N, N], dtype=torch.bool) 
    # adj_bool[data.edge_index[0, :], data.edge_index[1, :]] = True
    # shortest_path_result, path = algos.floyd_warshall(adj_bool.numpy())
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()
    # print(spatial_pos, spatial_pos.shape)
    # exit(0)
    
    c = 0.15
    # k1 = 100 # number of sampled neighbors, sequence length here
    Samples = 1 # sampled subgraphs for each node
    power_adj_list = [normalized_adj]
    for m in range(5): # attn_bias_dim - 1
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])

    sampling_matrix = c * inv((sp.eye(adj.shape[0]) - (1 - c) * normalized_adj).toarray()) # power_adj_list[1].toarray(), [n_node, n_node]
    # sampling_matrix = power_adj_list[4].toarray()

    # Create subgraph samples
    data_list = []
    for id in range(data_y.shape[0]):
        s = sampling_matrix[id]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-k1:]

        s = sampling_matrix[id]
        s[id] = 0
        s = np.maximum(s, 0)
        sample_num1 = np.minimum(k1, (s > 0).sum())
        sub_data_list = []
        for _ in range(Samples):
            if sample_num1 > 0:
                sample_index1 = np.random.choice(a=np.arange(data_y.shape[0]), size=sample_num1, replace=False, p=s/s.sum())
            else:
                sample_index1 = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index1, dtype=int), torch.tensor(top_neighbor_index[: k1-sample_num1], dtype=int)])

            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])
            attn_bias = attn_bias.permute(1, 2, 0)

            sub_data_list.append([attn_bias, node_feature_id, data_y[node_feature_id].long()])
        data_list.append(sub_data_list)

    data_file_path = dataset_dir + dataset_name + '/data_s' + str(k1) + '.pt'
    torch.save(data_list, data_file_path)

    print(f'Process done!')


def rand_nodes_seq(dataset_name, k1, p=None):
    # random nodes in sequence do not overlap
    print('Generate long sequence as input')

    dataset_dir = './dataset/'
    data_x = torch.load(dataset_dir + dataset_name + '/x.pt')
    data_y = torch.load(dataset_dir + dataset_name + '/y.pt')
    adj = sp.load_npz(dataset_dir + dataset_name + '/adj.npz')
    normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/normalized_adj.npz')
    column_normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz')

    # elif args.dataset_name in ['arxiv']:
    #     dataset = DglNodePropPredDataset(name='ogbn-arxiv',
    #                                      root=dataset_dir)
    #     split_idx = dataset.get_idx_split()
    #     train, val, test = split_idx["train"], split_idx["valid"], split_idx["test"]
    #     g, labels = dataset[0]
    #     features = g.ndata['feat']
    #     nclass = 40
    #     labels = labels.squeeze()
    #     g = dgl.to_bidirected(g)

    power_adj_list = [normalized_adj]
    attn_bias_dim = 6
    for m in range(attn_bias_dim - 1): # attn_bias_dim - 1
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])
    
    feature = data_x
    data_list = []

    # Shuffle node ids
    node_idx = np.arange(data_y.shape[0])
    random.shuffle(node_idx)

    # Each group contains random nodes for train
    n_group = math.ceil(data_y.shape[0]/k1)
    for group in range(n_group):
        sub_data_list = []
        for _ in range(1):
            if group == n_group - 1:
                # TODO: Pad length if use bs
                node_feature_id = torch.cat([torch.tensor(node_idx[group * k1: ], dtype=int)])

            else:
                node_feature_id = torch.cat([torch.tensor(node_idx[group * k1: (group+1) * k1], dtype=int)])

            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])
            attn_bias = attn_bias.permute(1, 2, 0)  # [n_node, n_node, attn_bias_dim]
            sub_data_list.append([attn_bias, node_feature_id, data_y[node_feature_id].long()])
            
        data_list.append(sub_data_list)


    data_file_path = dataset_dir + dataset_name + '/data_rand' + str(k1) + '.pt'
    # feature_file_path = dataset_dir + args.dataset_name  + '/feature.pt'
    if not os.path.exists(data_file_path): 
        torch.save(data_list, data_file_path)
    print(f'Process done!')
    # if not os.path.exists(feature_file_path): 
    #     torch.save(feature, feature_file_path)


if __name__ == '__main__':
    get_dataset('ogbn-arxiv')