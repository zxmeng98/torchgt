import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch_sparse import SparseTensor, matmul
import time
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from ogb.nodeproppred import DglNodePropPredDataset, PygNodePropPredDataset,Evaluator
from torch_scatter import scatter

# import dgl
# import torch
import numpy as np
# import psutil
# import os
import dgl
import itertools

from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph
from gt_sp.utils import fix_edge_index, reformat_graph
import random
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

def generate_sparse_edge_index(num_nodes, sparsity):
    max_edges = num_nodes * (num_nodes - 1) // 2  
    num_edges = int(max_edges * sparsity)  

    edges = set()
    while len(edges) < num_edges:
        node1 = random.randint(0, num_nodes - 1)
        node2 = random.randint(0, num_nodes - 1)
        if node1 != node2:
            edge = tuple(sorted([node1, node2]))
            edges.add(edge)

    edge_index = torch.tensor(list(edges), dtype=torch.long).t()
    return edge_index

def sparse_matrix_power(edge_index, num_nodes, power, device):
    edge_index = edge_index.to(device)
    values = torch.ones(edge_index.size(1), dtype=torch.float32).to(device)
    adj_matrix = torch.sparse.FloatTensor(edge_index, values, torch.Size([num_nodes, num_nodes])).to(device)

    result = adj_matrix.clone()
    cnt = 1
    for _ in range(power - 1):
        print("Iter {}".format(cnt))
        result = torch.sparse.mm(result, adj_matrix)
        cnt += 1
    return result.to("cpu")

def calculate_sparsity(sparse_matrix):
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]

    non_zero_elements = sparse_matrix._nnz()

    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity

def extract_edge_index(sparse_matrix):
    coalesced_matrix = sparse_matrix.coalesce()
    
    edge_index = coalesced_matrix.indices()
    return edge_index

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
    # print(edge_index_i, edge_index_i.size(), torch.max(edge_index_i), torch.min(edge_index_i))
    # exit(0)
    
    # # Fix edge index: add new edges of virtual nodes
    # edge_index_i = fix_edge_index(edge_index_i, idx_batch.shape[0])

    return edge_index_i

class CoreAttention(nn.Module):
    """
    Core attn 
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, device):
        super(CoreAttention, self).__init__()
        self.hidden_size_per_attention_head = att_size = hidden_size // num_heads
        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.drop_out_rate = attention_dropout_rate
        self.att_dropout = nn.Dropout(attention_dropout_rate).to(device)

    def full_attention(self, q, k, v, attn_bias=None, mask=None):
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
        
    def full_flash_attention(self, q, k, v, attn_bias=None, mask=None):
        # return None #flash_attn_func(q, k, v, mask=mask)
        return flash_attn_func(q, k, v, self.drop_out_rate)
    
    def sparse_attention(self, q, k, v, edge_index, attn_bias=None):
       # kqv: [total_s, n, hn],  e: [total_edges, n, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+1, s+1]
        batch_size, node_num = k.size(0), k.size(1)
        num_heads = self.num_heads
        
        # Reshaping into [total_s, np, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, np, hn],  e: [total_edges, np, hn]
        q = q.view(-1, num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, num_heads, self.hidden_size_per_attention_head)

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
            attn_bias = attn_bias.view(batch_size*node_num, batch_size*node_num, num_heads)
            attn_bias = attn_bias.repeat(1, 1, 1, num_heads)

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
        Z = score.new_zeros(v.size(0), num_heads, 1)    
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        x = wV / (Z + 1e-6)
        
        return x.view(batch_size, node_num, -1)
    
    def full_attention_hybrid(self, q, k, v, attn_bias=None, mask=None):
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
    
    def forward(self, q, k, v, edge_index, flash_attn=False, full=False, mask=None):
        if flash_attn:
            x = self.full_flash_attention(q, k, v, mask)
        else:
            if full:
                x = self.full_attention(q, k, v)
            else:
                x = self.sparse_attention(q, k, v, edge_index)
        return x


def generate_edge_index(num_nodes, sparsity=0.5):
    all_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]

    num_edges = int(len(all_edges) * sparsity)
    selected_edges = random.sample(all_edges, num_edges)

    edge_index = torch.tensor(selected_edges, dtype=torch.long).t()
    return edge_index
def create_block_sparse_mask(num_nodes, block_size, sparsity):
    mask = torch.zeros(num_nodes, num_nodes)

    num_elements_per_block = block_size * block_size

    total_elements = num_nodes * num_nodes
    num_nonzero_blocks = int(sparsity * total_elements / num_elements_per_block)

    for _ in range(num_nonzero_blocks):
        block_row = np.random.randint(0, num_nodes // block_size)
        block_col = np.random.randint(0, num_nodes // block_size)
        mask[block_row * block_size : (block_row + 1) * block_size,
            block_col * block_size : (block_col + 1) * block_size] = 1

    return mask

def mask_to_edge_index(mask):
    rows, cols = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([rows, cols], dim=0)
    return edge_index


def create_pairs(N, M, off_N, off_M):
    """Create a list of pairs (a, b) where a ranges from 1 to N and b ranges from 1 to M."""
    return [(a, b) for a in range(off_N, off_N + N) for b in range(off_M, off_M + M)]

def fully_connected_edge_index(num_nodes):
    all_edges = list(itertools.combinations(range(num_nodes), 2))
    edge_index_tensor = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    return edge_index_tensor


#     return torch.tensor(new_edge_index)
b = 1
s = 64000
n = 4 # per rank
hn = 16 # 16 32 64
h = n * hn
power = 1
block_size = 16
k = 8
# sparsity = 1-0.9999583207536489
# n = 96
# h = 12288
# hn = int(h/n)

# -------------------------Flash Attention-------------------------
# flash_attn = True
# full = False
# reorder = False

# -------------------------Sparse Attention-------------------------
# flash_attn = False
# full = False
# reorder = False

# -------------------------TorchGT-------------------------
flash_attn = False
full = False
reorder = True

device = "cuda"
    
if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if flash_attn:
        attn_type = "flash"
    else:
        if full:
            attn_type = "full"
        else:
            attn_type = "sparse"
    
    
    
    print("Start loading dataset")
    # dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/mzhang/data/')
    dataset = PygNodePropPredDataset(name='ogbn-products', root='/home/mzhang/data/')
    # edge_index = torch.load('./dataset/ogbn-arxiv' + '/edge_index.pt')

    num_nodes = dataset[0].num_nodes
    total_edge_index = dataset[0].edge_index
    # print(total_edge_index.shape)


    idx_batch_0 = torch.randint(0, num_nodes, [s])
    new_total_edge_index = gen_sub_edge_index(total_edge_index, idx_batch_0, num_nodes)
    
    # num_nodes = s
    # new_total_edge_index = fully_connected_edge_index(num_nodes)
    # print(edge_index)
    # print(f"raw edge shape: {new_total_edge_index.shape}")
    
    
    # Fix edge index: add new edges of virtual nodes
    new_total_edge_index = fix_edge_index(new_total_edge_index, idx_batch_0.shape[0])
    # print(f"fixed edge shape: {new_total_edge_index.shape}")

    # result = sparse_matrix_power(new_total_edge_index, s, power, "cuda")
    # sparsity = calculate_sparsity(result)
    # print("Sparsity of the matrix:", sparsity)
    # new_edge_index = extract_edge_index(result).to(device)

    # new_edge_index = generate_edge_index(s, 0.5).to(device)
    # mask = create_block_sparse_mask(s, block_size, sparsity)
    # new_edge_index = mask_to_edge_index(mask).to(device)
    
    if not reorder:
        new_edge_index = new_total_edge_index.to(device)  
    else:
        new_edge_index, _ = reformat_graph(new_total_edge_index, k, block_size)
        new_edge_index = new_edge_index.to(device)
        # print("New edge index:\n", new_edge_index.shape)

    str_edge_index = '\n'.join([f'{src}, {dst}' for src, dst in new_total_edge_index.t().tolist()])

    # Save graph topology in file
    # with open('./edge_index_full.txt', 'w') as file:
    #     file.write(str_edge_index)
    # exit(0)

    k = torch.randn(b, s+1, n, hn, requires_grad=True).to(device)
    q = torch.randn(b, s+1, n, hn, requires_grad=True).to(device)
    v = torch.randn(b, s+1, n, hn, requires_grad=True).to(device)
    if not flash_attn:
        targets = torch.randn(b, s+1, h, requires_grad=True).to(device)
    else:
        k = k.half()
        q = q.half()
        v = v.half()
        targets = torch.randn(b, s+1, n, hn, requires_grad=True).half().to(device)


    attn = CoreAttention(h, 0.1, n, device)

    criterion = nn.MSELoss()
    
    
    #warm up##
    for i in range(20):
        ret = attn(q, k, v, new_edge_index, flash_attn, full)
        loss = criterion(ret, targets)
        attn.zero_grad()
        # 反向传播
        loss.backward()
        
    torch.cuda.synchronize()

    # Note: the printed attention computation time is not accurate, since there is memorcy copy time which accounts large margin. It is suggested to use profiler tool to observe the preciser time.
    llm_profile = torch.profiler.profile
    with llm_profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            skip_first=1, wait=1, warmup=1, active=1, repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./tensorboard_trace/attn_module{h}_s{s}_{attn_type}_reorder-{reorder}/"
        ),
        with_stack=True,
        with_modules=True,
        profile_memory=True,
    ) as prof:
        start_time = time.time()
        for i in range(6):
            ret = attn(q, k, v, new_edge_index, flash_attn, full)
            loss = criterion(ret, targets)
            attn.zero_grad()
            # 反向传播
            loss.backward()
            prof.step()

        torch.cuda.synchronize()
            
        end_time = time.time()
    
    print("Attn Computation: {:.4f} ms with {} TFLOPS".format((end_time-start_time)/6*1000, (4*s*s*h*10/(end_time-start_time)/(1000*1000*1000*1000))))
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f"Allocated memory: {allocated / 1024 / 1024} MB")
    print(f"Reserved memory: {reserved / 1024 / 1024} MB")        
    # if flash_attn:
    #     print(ret.shape)
    # else:
    #     if full:
    #         print(ret.shape)
    #     else:
    #         print(ret.shape)
    
    # print(ret)