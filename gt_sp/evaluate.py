import torch
import torch.nn.functional as F
import numpy as np
from gt_sp.utils import get_batch, gen_sub_edge_index
from gt_sp.initialize import (
    get_sequence_parallel_world_size,
    set_last_batch_global_token_indices
)

def calc_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    is_labeled = y_true[:] == y_true[:]
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


@torch.no_grad()
def eval(args, model, device, x, y, sub_idx, adjs):
    """
    Evaluate the model on valid/test subset of nodes on GPU.
    If use this, model need to delete self.training.
    TODO loss need allreduce if reuse sp forward pass
    """
    seq_parallel_world_size = get_sequence_parallel_world_size()
    
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    
    num_batch = sub_idx.size(0) // args.seq_len + 1
    
    if sub_idx.shape[0] % args.seq_len != 0:
        x_dummy_list = [t for t in torch.tensor_split(
            torch.randn(sub_idx.shape[0] % args.seq_len, ), seq_parallel_world_size, dim=0)]
        rest_split_sizes = [t.shape[0] for t in x_dummy_list]
        afterpad_split_sizes = [max(rest_split_sizes)] * seq_parallel_world_size
        global_token_indices_last_batch = [afterpad_split_sizes[0]] + [sum(afterpad_split_sizes[:(i+1)]) + args.num_global_node*i for i in range(1, seq_parallel_world_size)]
        set_last_batch_global_token_indices(global_token_indices_last_batch)

    for i in range(num_batch):
        idx_i = sub_idx[i*args.seq_len:(i+1)*args.seq_len]
        
        # subsequence data
        x_i, y_i, attn_bias = get_batch(args, x, y, idx_i, adjs, rest_split_sizes, device)
        
        pred = model(x_i, attn_bias)
        loss = F.nll_loss(pred, y_i)
        loss_list.append(loss.item())
        y_true.append(y_i.view(-1))
        y_pred.append(pred.argmax(1))
        torch.cuda.empty_cache()

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
        
    acc = calc_acc(y_true, y_pred)
    
    return acc, np.mean(loss_list)


@torch.no_grad()
def eval_cpu_fullgraph(args, model, x, y, split_idx, full_attn_bias):
    """
    Evaluate the model on full nodes on CPU. - Very slow -
    pubmed: acc 87.8%
    """
    y_pred = []
    model.eval()
    model.to(torch.device("cpu"))
    
    y_pred = model(x, full_attn_bias)
    loss = F.nll_loss(y_pred, y.view(-1)).item()
    
    y_pred = y_pred.argmax(1)  # size: (N)
    
    train_acc = calc_acc(y[split_idx['train']], y_pred[split_idx['train']])
    valid_acc = calc_acc(y[split_idx['valid']], y_pred[split_idx['valid']])
    test_acc = calc_acc(y[split_idx['test']], y_pred[split_idx['test']])
    
    return train_acc, valid_acc, test_acc


@torch.no_grad()
def eval_cpu_subset_batch(args, model, x, y, sub_idx, adjs):
    """
    Evaluate the model on train/valid/test subset of nodes in a batched way on CPU.
    lager seq_len will be slower 
    pubmed: acc 87.8%
    """
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    model.to(torch.device("cpu"))
    
    # num_batch = sub_idx.size(0) // args.seq_len + 1
    # seq_len = 128
    num_batch = sub_idx.size(0) // args.seq_len + 1

    # for i in tqdm(range(num_batch), desc="Iteration"):
    for i in range(num_batch):
        idx_i = sub_idx[i*args.seq_len:(i+1)*args.seq_len]
        x_i = x[idx_i]
        y_i = y[idx_i]
        
        attn_bias = torch.cat([torch.tensor(i[idx_i, :][:, idx_i].toarray(), dtype=torch.float32).unsqueeze(0) for i in adjs])
        attn_bias = attn_bias.permute(1, 2, 0)
        
        pred = model(x_i, attn_bias)
        loss = F.nll_loss(pred, y_i)
        loss_list.append(loss.item())
        
        y_true.append(y_i.view(-1))
        y_pred.append(pred.argmax(1))
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
        
    acc = calc_acc(y_true, y_pred)
    
    return acc  


@torch.no_grad()
def eval_gpu_subset_batch(args, model, x, y, sub_idx, adjs, device):
    """
    Evaluate the model on train/valid/test subset of nodes in a batched way on GPU.
    """
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    
    # num_batch = sub_idx.size(0) // args.seq_len + 1
    # seq_len = 128
    num_batch = sub_idx.size(0) // args.seq_len + 1

    # for i in tqdm(range(num_batch), desc="Iteration"):
    for i in range(num_batch):
        idx_i = sub_idx[i*args.seq_len:(i+1)*args.seq_len]
        x_i = x[idx_i]
        y_i = y[idx_i]
        
        attn_bias = torch.cat([torch.tensor(i[idx_i, :][:, idx_i].toarray(), dtype=torch.float32).unsqueeze(0) for i in adjs])
        attn_bias = attn_bias.permute(1, 2, 0)
        
        x_i, y_i, attn_bias = x_i.to(device), y_i.to(device), attn_bias.to(device)
        pred = model(x_i, attn_bias)
        loss = F.nll_loss(pred, y_i)
        loss_list.append(loss.item())
        
        y_true.append(y_i.view(-1))
        y_pred.append(pred.argmax(1))
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
        
    acc = calc_acc(y_true, y_pred)
    
    del x_i, y_i, attn_bias
    torch.cuda.empty_cache()
    
    return acc  


@torch.no_grad()
def sparse_eval_cpu_subset_batch(args, model, x, y, sub_idx, adjs, edge_index):
    """
    Evaluate the model on train/valid/test subset of nodes in a batched way on CPU.
    lager seq_len will be slower 
    """
    model.eval()
    model.to(torch.device("cpu"))

    y_true = []
    y_pred = []
    loss_list = []
    N = x.shape[0]
    
    # num_batch = sub_idx.size(0) // args.seq_len + 1
    # seq_len = 128
    num_batch = sub_idx.size(0) // args.seq_len + 1

    if args.attn_type == "full":
        attn_type = "full"
    else:
        attn_type = "sparse"

    # for i in tqdm(range(num_batch), desc="Iteration"):
    for i in range(num_batch):
        idx_i = sub_idx[i*args.seq_len:(i+1)*args.seq_len]
        x_i = x[idx_i]
        y_i = y[idx_i]
        
        attn_bias = torch.cat([torch.tensor(i[idx_i, :][:, idx_i].toarray(), dtype=torch.float32).unsqueeze(0) for i in adjs])
        attn_bias = attn_bias.permute(1, 2, 0)
        edge_index_i = gen_sub_edge_index(edge_index, idx_i, N) # [2, num_edges] index plused global token
        
        pred = model(x_i, attn_bias, edge_index_i, attn_type=attn_type)
        loss = F.nll_loss(pred, y_i)
        loss_list.append(loss.item())
        
        y_true.append(y_i.view(-1))
        y_pred.append(pred.argmax(1))
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
        
    acc = calc_acc(y_true, y_pred)
    
    return acc  


@torch.no_grad()
def sparse_eval_gpu_subset_batch(args, model, x, y, sub_idx, adjs, edge_index, device):
    """
    Evaluate the model on train/valid/test subset of nodes in a batched way on GPU.
    lager seq_len will be slower 
    """
    model.eval()

    y_true = []
    y_pred = []
    loss_list = []
    N = x.shape[0]
    
    # num_batch = sub_idx.size(0) // args.seq_len + 1
    # seq_len = 128
    num_batch = sub_idx.size(0) // args.seq_len + 1

    if args.attn_type == "full":
        attn_type = "full"
    else:
        attn_type = "sparse"

    # for i in tqdm(range(num_batch), desc="Iteration"):
    for i in range(num_batch):
        idx_i = sub_idx[i*args.seq_len:(i+1)*args.seq_len]
        x_i = x[idx_i]
        y_i = y[idx_i]
        
        attn_bias = torch.cat([torch.tensor(i[idx_i, :][:, idx_i].toarray(), dtype=torch.float32).unsqueeze(0) for i in adjs])
        attn_bias = attn_bias.permute(1, 2, 0)
        edge_index_i = gen_sub_edge_index(edge_index, idx_i, N) # [2, num_edges] index plused global token
        
        x_i, y_i, edge_index_i, attn_bias = x_i.to(device), y_i.to(device), edge_index_i.to(device), attn_bias.to(device)
        
        pred = model(x_i, attn_bias, edge_index_i, attn_type=attn_type)
        loss = F.nll_loss(pred, y_i)
        loss_list.append(loss.item())
        
        y_true.append(y_i.view(-1))
        y_pred.append(pred.argmax(1))
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
        
    acc = calc_acc(y_true.to(torch.device("cpu")), y_pred.to(torch.device("cpu")))
    
    del x_i, y_i, edge_index_i, attn_bias
    torch.cuda.empty_cache()
    
    return acc  



@torch.no_grad()
def sparse_eval_cpu_subset_batch_dummy_bias(args, model, x, y, sub_idx, dummy_attn_bias, edge_index):
    """
    Evaluate the model on train/valid/test subset of nodes in a batched way on CPU.
    lager seq_len will be slower 
    """
    model.eval()
    model.to(torch.device("cpu"))

    y_true = []
    y_pred = []
    loss_list = []
    N = x.shape[0]
    
    # num_batch = sub_idx.size(0) // args.seq_len + 1
    # seq_len = 128
    num_batch = sub_idx.size(0) // args.seq_len + 1
    
    if args.attn_type == "full":
        attn_type = "full"
    # elif args.attn_type == "flash":
    #     attn_type = "flash"
    else:
        attn_type = "sparse"
    

    # for i in tqdm(range(num_batch), desc="Iteration"):
    for i in range(num_batch):
        idx_i = sub_idx[i*args.seq_len:(i+1)*args.seq_len]
        x_i = x[idx_i]
        y_i = y[idx_i]
        
        # if idx_i.shape[0] < args.seq_len:
        #     dummy_attn_bias = torch.zeros(idx_i.shape[0], idx_i.shape[0], args.attn_bias_dim, dtype=torch.float32)

        edge_index_i = gen_sub_edge_index(edge_index, idx_i, N) # [2, num_edges] index plused global token
        
        pred = model(x_i, dummy_attn_bias, edge_index_i, attn_type=attn_type)
        loss = F.nll_loss(pred, y_i)
        loss_list.append(loss.item())
        
        y_true.append(y_i.view(-1))
        y_pred.append(pred.argmax(1))
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
        
    acc = calc_acc(y_true, y_pred)
    
    return acc  


@torch.no_grad()
def sparse_eval_gpu(args, model, x, y, sub_idx, attn_bias, edge_index, device):
    """
    Evaluate the model on train/valid/test subset of nodes in a batched way on GPU.
    lager seq_len will be slower 
    """
    model.eval()

    y_true = []
    y_pred = []
    loss_list = []
    N = x.shape[0]
    
    # num_batch = sub_idx.size(0) // args.seq_len + 1
    # seq_len = 128
    num_batch = sub_idx.size(0) // args.seq_len + 1
    
    if args.attn_type == "full":
        attn_type = "full"
    elif args.attn_type == "flash":
        attn_type = "flash"
    else:
        attn_type = "sparse"
    

    # for i in tqdm(range(num_batch), desc="Iteration"):
    for i in range(num_batch):
        idx_i = sub_idx[i*args.seq_len:(i+1)*args.seq_len]
        x_i = x[idx_i]
        y_i = y[idx_i]
        
        # if idx_i.shape[0] < args.seq_len:
        #     dummy_attn_bias = torch.zeros(idx_i.shape[0], idx_i.shape[0], args.attn_bias_dim, dtype=torch.float32)
        edge_index_i = gen_sub_edge_index(edge_index, idx_i, N) # [2, num_edges] index plused global token
        
        x_i, y_i, edge_index_i = x_i.to(device), y_i.to(device), edge_index_i.to(device)

        pred = model(x_i, attn_bias, edge_index_i, attn_type=attn_type)
        loss = F.nll_loss(pred, y_i)
        loss_list.append(loss.item())
        
        y_true.append(y_i.view(-1))
        y_pred.append(pred.argmax(1))
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
        
    acc = calc_acc(y_true, y_pred)
    
    del x_i, y_i, edge_index_i
    torch.cuda.empty_cache()
    
    return acc  


@torch.no_grad()
def eval_cpu_batch(args, model, x, y, split_idx, adjs):
    """
    Evaluate the model on full nodes in a batched way.
    pubmed: acc 88.0%
    """
    y_true = []
    y_pred = []
    model.eval()
    model.to(torch.device("cpu"))

    full_idx = torch.tensor(np.arange(y.shape[0]))
    
    num_batch = full_idx.size(0) // args.seq_len + 1

    # for i in tqdm(range(num_batch), desc="Iteration"):
    for i in range(num_batch):
        idx_i = full_idx[i*args.seq_len:(i+1)*args.seq_len]
        x_i = x[idx_i]
        y_i = y[idx_i]
        
        attn_bias = torch.cat([torch.tensor(i[idx_i, :][:, idx_i].toarray(), dtype=torch.float32).unsqueeze(0) for i in adjs])
        attn_bias = attn_bias.permute(1, 2, 0)
        
        pred = model(x_i, attn_bias)
        loss = F.nll_loss(pred, y_i)
        
        y_true.append(y_i.view(-1))
        y_pred.append(pred.argmax(1))
        
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    
    train_acc = calc_acc(y[split_idx['train']], y_pred[split_idx['train']])
    valid_acc = calc_acc(y[split_idx['valid']], y_pred[split_idx['valid']])
    test_acc = calc_acc(y[split_idx['test']], y_pred[split_idx['test']])
    
    return train_acc, valid_acc, test_acc