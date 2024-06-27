import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from utils.lr import PolynomialDecayLR
import argparse
import math
from tqdm import tqdm
import scipy.sparse as sp
import copy
import os
import time
import random
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from gt_sp.evaluate import calc_acc
from gt_sp.initialize import (
    initialize_distributed,
    initialize_sequence_parallel,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
    get_last_batch_flag,
    last_batch_flag,
)
from gt_sp.reducer import sync_params_and_buffers, Reducer
from gt_sp.evaluate import eval_cpu_subset_batch, eval_cpu_fullgraph, eval_cpu_batch
from gt_sp.utils import get_batch_from_loader, get_batch_from_loader_malnet, pad_x_bs, pad_2d_bs
from data.dataset import GraphormerDataset
from models.graphormer_dist_graph_level_mp_malnet import Graphormer
from models.gt_dist_graph_level_mp_malnet import GT
from utils.parser import parser_add_main_args
import torch.multiprocessing as mp
from datetime import timedelta


# gradient all-reduce context
reducer = Reducer()


def reduce_hook(param, name, grad):
    # reduction hook is only used if overlapping communication
    reducer.reduce(param, name, grad)


def get_sp_rank_data(args, batch, device):
    # For sequence parallel
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0

    sub_split_seq_lens = batch[5] # [9, 9, 9, 8]
    x_i_list = [t for t in torch.split(batch[0], sub_split_seq_lens, dim=1)]
    in_degree_list = [t for t in torch.split(batch[2], sub_split_seq_lens, dim=1)]
    out_degree_list = [t for t in torch.split(batch[3], sub_split_seq_lens, dim=1)]


    #### Pad cut data to the same sub seq length. e.g., [9, 9, 9, 8] -> [9, 9, 9, 9]
    padlen = max(sub_split_seq_lens)
    sub_real_seq_len = padlen + args.num_global_node
    
    x_i_list_pad = [pad_x_bs(t, padlen) for t in x_i_list]
    in_degree_list_pad = [pad_2d_bs(t, padlen) for t in in_degree_list]
    out_degree_list_pad = [pad_2d_bs(t, padlen) for t in out_degree_list]

    # Set global token indices for each batch, in graphormer global token idx: 0
    global_token_indices = list(range(0, args.sequence_parallel_size * sub_real_seq_len, sub_real_seq_len))
    set_global_token_indices(global_token_indices)
    
    x_i = x_i_list_pad[seq_parallel_world_rank].to(device) # [bs, padlen, 1]
    y_i = batch[1].to(device)
    in_degree_i = in_degree_list_pad[seq_parallel_world_rank].to(device)
    out_degree_i = out_degree_list_pad[seq_parallel_world_rank].to(device)
    edge_index = batch[4].to(device)

    
    return x_i, y_i, in_degree_i, out_degree_i, edge_index
    


def train(args, model, device, packed_data, optimizer, criterion, epoch, lr_scheduler):
    model.train()
    model.to(device)

    loss_list, iter_t_list = [], []
    y_pred_list = []
    y_true_list = []
    # for batch in tqdm(loader, desc=f"Epoch {epoch} Rank {get_sequence_parallel_rank()} Iteration"):

    if args.attn_type == "hybrid":
        percent_list  = [(i + 1) / args.switch_freq for i in range(args.switch_freq)]
        switch_points = [int(len(packed_data) * percentage) for percentage in percent_list]
    iter = 1
    for batch in packed_data:
        x_i, y_i, in_degree_i, out_degree_i, edge_index = get_sp_rank_data(args, batch, device)
        # print(edge_index)
        
        t0 = time.time()
        
        # batch = batch.to(device)
        
        if args.attn_type == "hybrid":
            if iter in switch_points:
                attn_type = "full"  
            else:
                attn_type = "sparse"
        elif args.attn_type == "sparse":
            attn_type = "sparse"
        elif args.attn_type == "full":
            attn_type = "full"
        elif args.attn_type == "flash":
            attn_type = "flash"
        
        pred = model(x_i, in_degree_i, out_degree_i, edge_index, attn_type=attn_type) # [bs, num_class]

        if args.dataset in ["ZINC"]:
            pred = pred.view(-1)
            y_true = y_i.view(-1)
        elif args.dataset in ["MalNetTiny","MalNet","CIFAR10"]:
            y_true = y_i.view(-1)
            
        loss = criterion(pred, y_true)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # sync all-reduce gradient 
        # reducer.synchronize()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.div_(get_sequence_parallel_world_size())
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())

        optimizer.step()      

        # if cnt > 1:
        iter_t_list.append(time.time() - t0)
        loss_list.append(loss.item())

        lr_scheduler.step()
        iter += 1

        if args.dataset in ["MalNetTiny","MalNet","CIFAR10"]:
            y_pred_list.append(pred.argmax(1))
            y_true_list.append(y_true)

    if args.dataset in ["MalNetTiny","MalNet","CIFAR10"]:
        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        eval_metric = calc_acc(y_true, y_pred)
        if args.rank == 0:
            print(f'trian acc: {eval_metric}')

    return np.mean(loss_list)

    # # cnt = 0
    # for name, param in model.named_parameters():
    #     if name == 'graph_node_feature.atom_encoder.weight' and (epoch-1) % 10 == 0:
    #     # if args.rank == 0:
    #         print(f'rank {args.rank} {name} {param}')
    # if epoch == 11:
    #     exit(0)


@torch.no_grad()
def eval_gpu(args, model, device, packed_data, criterion, evaluator, metric, str):
    """
    Do graph-level evluation on GPU
    Convergence is normal 
    """
    model.eval()
    model.to(device)

    y_pred_list = []
    y_true_list = []
    loss_list = []
    # for batch in tqdm(loader, desc=f"Epoch {epoch} Rank {get_sequence_parallel_rank()} Iteration"):
    for batch in packed_data:
        x_i, y_i, in_degree_i, out_degree_i, edge_index = get_sp_rank_data(args, batch, device)
        if args.attn_type == "full":
            attn_type = "full"
        elif args.attn_type == "flash":
            attn_type = "flash"
        else:
            attn_type = "sparse"
        
        pred = model(x_i, in_degree_i, out_degree_i, edge_index, attn_type=attn_type)

        if args.dataset in ["ZINC"]:
            pred = pred.view(-1)
            y_true = y_i.view(-1)
        elif args.dataset in ["MalNetTiny","MalNet","CIFAR10"]:
            y_true = y_i.view(-1)
        
            
        loss = criterion(pred, y_true)      
        loss_list.append(loss.item())

        if args.dataset in ["ZINC","ogbg-molhiv","ogbg-molpcba"]:
            y_pred_list.append(pred)
        elif args.dataset in ["MalNetTiny","MalNet","CIFAR10"]:
            y_pred_list.append(pred.argmax(1))
        y_true_list.append(y_true)


    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    
    if metric in ["mae", "rocauc", "ap"]:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        eval_metric = evaluator.eval(input_dict)[metric]    
    elif metric in ["accuracy"]:
        eval_metric = calc_acc(y_true, y_pred)

    if args.rank == 0:
        print(f'{str} {metric}: {eval_metric}')
    
    return eval_metric


def main(rank, args, packed_train_data, packed_val_data, packed_test_data, n_classes):
    # Initialize distributed 
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    args.rank = rank
    
    dist.init_process_group(
    backend=args.distributed_backend,
    world_size=args.world_size, rank=rank,
    timeout=timedelta(minutes=args.distributed_timeout_minutes))

    initialize_sequence_parallel(args.seq_len, 1, 1,
                                        args.sequence_parallel_size)
    
    device = f'cuda:{torch.cuda.current_device()}' # 相对GPU编号  
    
    # Model 
    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.input_dropout_rate,
            ffn_dim=args.ffn_dim,
            dataset_name=args.dataset,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            attention_dropout_rate=args.attention_dropout_rate,     
            output_dim=n_classes,
        ).to(device)
    elif args.model == "gt":
        model = GT(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.input_dropout_rate,
            ffn_dim=args.ffn_dim,
            dataset_name=args.dataset,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            attention_dropout_rate=args.attention_dropout_rate,     
            output_dim=n_classes,
        ).to(device)
    
    if args.rank == 0:
        print('Model params:', sum(p.numel() for p in model.parameters()))

    # Sync params and buffers. Ensures all rank models start off at the same value
    sync_params_and_buffers(model)

    # # Register the grad hooks
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     param.register_hook(partial(reduce_hook, param, name))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_updates,
            tot=args.tot_updates,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)
    criterion = F.cross_entropy
    
    val_acc_list, test_acc_list, epoch_t_list, loss_list = [], [], [], []
    best_model, best_val, best_test = None, float('-inf'), float('-inf')


    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train(args, model, device, packed_train_data, optimizer, criterion, epoch, lr_scheduler) 
        torch.cuda.synchronize()
        t1 = time.time()
        # if epoch > 5:
        epoch_t_list.append(t1 - t0)
        if args.rank == 0:
            if epoch > 2:
                print(f"Epoch: {epoch}, train loss: {train_loss}, epoch time: {np.mean(epoch_t_list):.2f}s")
            else:
                print(f"Epoch: {epoch}, train loss: {train_loss}, warmup epoch time: {t1 - t0:.2f}s")
            # print(train_t)

    #     # # NOTE: Graph-level任务evaluation要复用sp的model forward，收敛才正确！
    #     valid_matric = eval_gpu(args, model, device, packed_val_data, criterion, None, "accuracy", "val")
    #     test_matric = eval_gpu(args, model, device, packed_test_data, criterion, None, "accuracy", "test")
    #     val_acc_list.append(valid_matric)
    #     test_acc_list.append(test_matric)
    #     loss_list.append(train_loss)

    #     if test_matric > best_test:
    #         best_test = test_matric

    # if args.rank == 0:
    #     print("Best test accuracy: {:.2%}".format(best_test))

    #     if not os.path.exists(f'./exps/{args.dataset}'): 
    #         os.makedirs(f'./exps/{args.dataset}')
    #     if args.attn_type != "hybrid":
    #         if args.reorder:
    #             np.save(f'./exps/{args.dataset}/{args.model}_{str(args.attn_type)}_reorder_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_test', np.array(test_acc_list))
    #             # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
    #             np.save(f'./exps/{args.dataset}/{args.model}_{str(args.attn_type)}_reorder_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_loss', np.array(loss_list))
    #         else:
    #             np.save(f'./exps/{args.dataset}/{args.model}_{str(args.attn_type)}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_test', np.array(test_acc_list))
    #             # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
    #             np.save(f'./exps/{args.dataset}/{args.model}_{str(args.attn_type)}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_loss', np.array(loss_list))
    #     else:
    #         np.save(f'./exps/{args.dataset}/{args.model}_{str(args.attn_type)}_{args.switch_freq}_s{args.batch_size}_e{args.epochs}_sp{args.world_size}_test', np.array(test_acc_list))
    #         # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
    #         np.save(f'./exps/{args.dataset}/{args.model}_{str(args.attn_type)}_{args.switch_freq}_s{args.batch_size}_e{args.epochs}_sp{args.world_size}_loss', np.array(loss_list))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Parallel implementation of graphormer')
    parser_add_main_args(parser)
    args = parser.parse_args()
        
    print(args)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)
    
    args.world_size = args.sequence_parallel_size

    # Data stuff.
    dm = GraphormerDataset(dataset_name=args.dataset, dataset_dir=args.dataset_dir,
                        num_workers=args.num_workers, batch_size=args.batch_size, seed=args.seed, multi_hop_max_dist=args.multi_hop_max_dist, spatial_pos_max=args.spatial_pos_max, myargs=args)
    
    shuffle_idx = torch.randperm(len(dm.dataset_train))
    rand_train = Subset(dm.dataset_train, shuffle_idx.tolist())
    dm.dataset_train = rand_train

    train_loader, val_loader, test_loader = dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()
    
    # TODO: 从dataloader里面取得batch不能直接送给子进程，会报错
    # Pack all epochs' data in one step
    packed_train_data, packed_val_data, packed_test_data = [], [], []
    t0 = time.time()
    cnt = 0
    for batch in train_loader:
        # data_i = get_batch_from_loader_malnet(args, batch)
        packed_train_data.append((batch.x, batch.y, batch.in_degree, batch.out_degree, batch.edge_index, batch.sub_split_seq_lens))
        cnt += 1
        # if cnt > 3:
        #     break
        
    cnt = 0
    for batch in val_loader:
    #     data_i = get_batch_from_loader_malnet(args, batch)
        packed_val_data.append((batch.x, batch.y, batch.in_degree, batch.out_degree, batch.edge_index, batch.sub_split_seq_lens))
        cnt += 1
        # if cnt > 3:
        #     break

    cnt = 0
    for batch in test_loader:
        # data_i = get_batch_from_loader_malnet(args, batch)
        packed_test_data.append((batch.x, batch.y, batch.in_degree, batch.out_degree, batch.edge_index, batch.sub_split_seq_lens))
        cnt += 1
        # if cnt > 3:
        #     break
        
    # print(packed_test_data[0][0][0, :10, 0], packed_test_data[1][0][0, :10, 0])
    
    print(f"Pack data time: {time.time() - t0:.1f}s")

    
    #### Initialize distributed ####
    processes = []
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        n = len(devices)
    else:
        n = torch.cuda.device_count()
        devices = [f'{i}' for i in range(n)]

    mp.set_start_method('spawn', force=True)
    start_id = args.node_rank * n
    for i in range(start_id, min(start_id + n, args.world_size)):
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[i % len(devices)]
        p = mp.Process(target=main, args=(i, args, packed_train_data, packed_val_data, packed_test_data, dm.dataset["num_class"]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    