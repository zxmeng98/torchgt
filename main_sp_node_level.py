import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from utils.lr import PolynomialDecayLR
import argparse
import scipy.sparse as sp
import os
import time
import random
import pandas as pd
import torch.distributed as dist
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
)
from gt_sp.reducer import sync_params_and_buffers, Reducer
from gt_sp.evaluate import sparse_eval_gpu
from gt_sp.utils import random_split_idx, get_batch_reorder_blockize, check_conditions
from utils.parser_node_level import parser_add_main_args
from collections import deque
import dgl


def main():
    parser = argparse.ArgumentParser(description='TorchGT node-level training arguments.')
    parser_add_main_args(parser)
    args = parser.parse_args()
   
    # Initialize distributed 
    initialize_distributed(args)
    device = f'cuda:{torch.cuda.current_device()}' 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)
    
    # Dataset 
    feature = torch.load(args.dataset_dir + args.dataset + '/x.pt') # [N, x_dim]
    y = torch.load(args.dataset_dir + args.dataset + '/y.pt') # [N]
    edge_index = torch.load(args.dataset_dir + args.dataset + '/edge_index.pt') # [2, num_edges]
    N = feature.shape[0]

    if args.dataset == 'pokec':
        y = torch.clamp(y, min=0) 
    split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)

    if args.rank == 0:
        print(args)
        print('Dataset load successfully')
        print(f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}") 
        print(f"Training iters: {split_idx['train'].size(0) // args.seq_len + 1}, Val iters: {split_idx['valid'].size(0) // args.seq_len + 1}, Test iters: {split_idx['test'].size(0) // args.seq_len + 1}")
    
    # Broadcast train indexes to all ranks 
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()

    train_idx = split_idx['train']
    if args.rank == 0:
        flatten_train_idx = train_idx.to('cuda')
    else:
        total_numel = train_idx.numel()
        flatten_train_idx = torch.empty(total_numel,
                                device=device,
                                dtype=torch.int64)
    # Broadcast
    dist.broadcast(flatten_train_idx, src_rank, group=group)

    # Initialize global token indices
    seq_len_per_rank = get_sequence_length_per_rank()
    sub_real_seq_len = seq_len_per_rank + args.num_global_node
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))

    # Last batch fix sequence length
    if flatten_train_idx.shape[0] % args.seq_len != 0:
        last_batch_node_num = flatten_train_idx.shape[0] % args.seq_len
        if last_batch_node_num % seq_parallel_world_size != 0:
            div = last_batch_node_num // seq_parallel_world_size
            last_batch_node_num = div * seq_parallel_world_size + (seq_parallel_world_size - 1)

        x_dummy_list = [t for t in torch.tensor_split(
            torch.zeros(last_batch_node_num, ), seq_parallel_world_size, dim=0)]
        sub_split_seq_lens = [t.shape[0] for t in x_dummy_list] # e.g., [14, 14, 14, 13]
        sub_real_seq_len = max(sub_split_seq_lens) + args.num_global_node
        global_token_indices_last_batch = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
    else:
        sub_split_seq_lens = None
        global_token_indices_last_batch = None
    set_global_token_indices(global_token_indices)
    set_last_batch_global_token_indices(global_token_indices_last_batch)

    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item()+1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node
        ).to(device)
    elif args.model == "gt":
        model = GT(
           n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item()+1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node
        ).to(device)
        
    if args.rank == 0:
        print('Model params:', sum(p.numel() for p in model.parameters()))

    # Sync params and buffers. Ensures all rank models start off at the same value
    sync_params_and_buffers(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_updates,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)
    
    val_acc_list, test_acc_list, epoch_t_list = [], [], []
    best_model, best_val, best_test = None, float('-inf'), float('-inf')

    num_batch = flatten_train_idx.size(0) // args.seq_len + 1

    compare_ldr = deque([0, 0, 0, 0, 0]) 
    beta_coeffi_list = [0, 1, 1.5, 5, 7, 10, '1']
    beta_max, beta_idx  = 1, 1

    for epoch in range(1, args.epochs + 1):
        model.to(device)
        model.train()
        
        loss_list, iter_t_list = [], []
        
        if args.attn_type == "hybrid":
            percent_list  = [(i + 1) / args.switch_freq for i in range(args.switch_freq)]
            switch_points = [int(num_batch * percentage) for percentage in percent_list]
        iter = 1
        
        for i in range(num_batch):
            idx_i = flatten_train_idx[i*args.seq_len: (i+1)*args.seq_len]
            packed_data = get_batch_reorder_blockize(args, feature, y, idx_i.to("cpu"), sub_split_seq_lens, device, edge_index, N, k=8, block_size=16, beta_coeffi=beta_coeffi_list[beta_idx])

            x_i, y_i, edge_index_i, attn_bias = packed_data
            if attn_bias is not None:
                x_i, y_i, edge_index_i, attn_bias = x_i.to(device), y_i.to(device), edge_index_i.to(device), attn_bias.to(device)
            else:
                x_i, y_i, edge_index_i = x_i.to(device), y_i.to(device), edge_index_i.to(device)
        
            if args.attn_type == "sparse":
                attn_type = "sparse"
            elif args.attn_type == "full":
                attn_type = "full"
            elif args.attn_type == "flash":
                attn_type = "flash"
            
            # if args.attn_type == "hybrid":
                # if args.rank == 0: 
                #     con_result = check_conditions(edge_index, idx_i.shape[0])

                # if con_result:
                #     attn_type = "sparse"
                # else:
                #     attn_type = "full"       
            t1 = time.time()
                
            out_i = model(x_i, attn_bias, edge_index_i, attn_type=attn_type)    
            loss = F.nll_loss(out_i, y_i.long())
            optimizer.zero_grad(set_to_none=True) 
            loss.backward()
            
            # Sync all-reduce gradient 
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad.div_(get_sequence_parallel_world_size())
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())

            optimizer.step()  
            torch.cuda.synchronize()   
            t2 = time.time() 
             
            iter_t_list.append(t2 - t1) 
     
        loss_list.append(loss.item()) 
        lr_scheduler.step()
        
        if epoch > 4 and args.rank == 0:  
            epoch_t_list.append(np.sum(iter_t_list))
            print("------------------------------------------------------------------------------------")
            print("Epoch: {:03d}, Loss: {:.4f}, Epoch Time: {:.3f}s".format(epoch, np.mean(loss_list), np.mean(epoch_t_list)))
            print("------------------------------------------------------------------------------------")

        if args.rank == 0 and epoch % 5 == 0:   
            t4 = time.time()
            train_acc = sparse_eval_gpu(args, model, feature, y, split_idx['train'], attn_bias, edge_index, device) 
            val_acc = sparse_eval_gpu(args, model, feature, y, split_idx['valid'], attn_bias, edge_index, device)
            test_acc = sparse_eval_gpu(args, model, feature, y, split_idx['test'], attn_bias, edge_index, device)
            t5 = time.time()
            print("------------------------------------------------------------------------------------")
            print(f'Eval time {t5-t4}s')
            print("Epoch: {:03d}, Loss: {:4f}, Train acc: {:.2%}, Val acc: {:.2%}, Test acc: {:.2%}, Epoch Time: {:.3f}s".format(
                epoch, np.mean(loss_list), train_acc, val_acc, test_acc, np.mean(epoch_t_list)))
            print("------------------------------------------------------------------------------------")
            
            if val_acc > best_val:
                best_val = val_acc
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}.pkl')
            
            if test_acc > best_test:
                best_test = test_acc
            
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

        # Adaptive beta
        if args.rank == 0:
            if epoch == 1:
                f_loss = loss.item() 
            else:
                f_loss_old = f_loss
                f_loss = 0.9 * f_loss + 0.1 * loss.item()
                if epoch >= 5:
                    v_loss = abs(f_loss - f_loss_old) / np.sum(iter_t_list)
                    compare_ldr.popleft()
                    compare_ldr.append(v_loss)
                    if epoch >= 9:
                        increase_beta, reduce_beta = True, True
                        for k in range(1, len(compare_ldr)):
                            if compare_ldr[k] > compare_ldr[k-1]:
                                reduce_beta = False
                                break
                        for k in range(1, len(compare_ldr)):
                            if compare_ldr[k] < compare_ldr[k-1]:
                                increase_beta = False
                                break
                        if increase_beta:
                            if beta_idx < len(beta_coeffi_list)-1:
                                beta_idx = beta_idx + 1
                        if reduce_beta:
                            if beta_idx > 0:
                                beta_idx = beta_idx - 1

        # Notify other ranks on the beta change           
        if args.rank == 0:
            beta_idx_broad = torch.LongTensor([beta_idx]).to(device)
        else:
            beta_idx_broad = torch.empty(1, dtype=torch.int64, device=device)

        dist.barrier()
        dist.broadcast(beta_idx_broad, src_rank, group=group)
        beta_idx = int(beta_idx_broad.item())

    if args.rank == 0:
        print("Best validation accuracy: {:.2%}, test accuracy: {:.2%}".format(best_val, best_test))

        if not os.path.exists(f'./exps/{args.dataset}'): 
            os.makedirs(f'./exps/{args.dataset}')
            
        if args.attn_type != "hybrid":
            if args.reorder:
                np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_reorder_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_test-fp16', np.array(test_acc_list))
                # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
                np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_reorder_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_loss-fp16', np.array(loss_list))
            else:
                np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_test', np.array(test_acc_list))
                # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
                np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_loss', np.array(loss_list))
        else:
            np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_{args.switch_freq}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_test', np.array(test_acc_list))
            # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
            np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_{args.switch_freq}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_loss', np.array(loss_list))


if __name__ == "__main__":
    main()
