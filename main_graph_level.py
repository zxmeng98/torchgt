import torch
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
from gt_sp.early_stop import EarlyStopping, Stop_args
from data.dataset import GraphormerDataset
from models.graphormer_sparse_graph_level import Graphormer
from models.gt_graph_level import GT
from utils.parser import parser_add_main_args
from gt_sp.evaluate import calc_acc
from torch.utils.data import DataLoader, Subset


def train(args, model, device, loader, optimizer, criterion, epoch, lr_scheduler):
    model.train()

    loss_list = []
    percent_list  = [(i + 1) / args.switch_freq for i in range(args.switch_freq)]
    switch_points = [int(len(loader) * percentage) for percentage in percent_list]
    iter = 1
    for batch in loader:
        if args.attn_type == "hybrid":
            if iter in switch_points:
                attn_type = "full"  
            else:
                attn_type = "sparse"
        elif args.attn_type == "sparse":
            attn_type = "sparse"
        elif args.attn_type == "full":
            attn_type = "full"

        batch = batch.to(device)
        pred = model(batch, attn_type=attn_type)

        if args.dataset == "ZINC":
            pred = pred.view(-1)
            y_true = batch.y.view(-1)
        elif args.dataset in ["ogbg-molhiv","ogbg-molpcba"]:
            pred = pred.view(-1)
            y_true = batch.y.view(-1).float()
            mask = ~torch.isnan(y_true)
            pred = pred[mask]
            y_true = y_true[mask]
        elif args.dataset in ["MalNetTiny","MalNet","CIFAR10"]:
            y_true = batch.y.view(-1)
            
        loss = criterion(pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
           
        loss_list.append(loss.item())
        lr_scheduler.step()

        iter += 1
        # print(f"batch loss: {loss.item()}")
    
    # print(f'Epoch: {epoch} train loss: {np.mean(loss_list)}')
    return np.mean(loss_list)


@torch.no_grad()
def eval(args, model, device, loader, criterion, evaluator, metric, str):
    model.eval()

    y_pred_list = []
    y_true_list = []
    loss_list = []

    cnt = 0
    # for batch in tqdm(loader, desc="Iteration"):
    for batch in loader:
        batch = batch.to(device)

        if args.attn_type == "full":
            attn_type = "full"
        else:
            attn_type = "sparse"
        pred = model(batch, attn_type=attn_type)

        if args.dataset == "ZINC":
            pred = pred.view(-1)
            y_true = batch.y.view(-1)
        elif args.dataset in ["ogbg-molhiv","ogbg-molpcba"]:
            pred = pred
            y_true = batch.y
        elif args.dataset in ["MalNetTiny","MalNet","CIFAR10"]:
            y_true = batch.y.view(-1)
            
        # loss = criterion(pred, y_true)
        # loss_list.append(loss.item())

        if args.dataset in ["ZINC","ogbg-molhiv","ogbg-molpcba"]:
            y_pred_list.append(pred)
        elif args.dataset in ["MalNetTiny","MalNet","CIFAR10"]:
            y_pred_list.append(pred.argmax(1))
        y_true_list.append(y_true)
        cnt += 1

    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)

    if metric in ["mae", "rocauc", "ap"]:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        eval_metric = evaluator.eval(input_dict)[metric]    
    elif metric in ["accuracy"]:
        eval_metric = calc_acc(y_true, y_pred)

    print(f'{str} {metric}: {eval_metric}')
        
    return eval_metric



def main():
    parser = argparse.ArgumentParser(description='Implementation of graphormer')
    parser_add_main_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(args)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)
    
    # Data stuff.
    dm = GraphormerDataset(dataset_name=args.dataset, dataset_dir=args.dataset_dir,
                           num_workers=args.num_workers, batch_size=args.batch_size, seed=args.seed, multi_hop_max_dist=args.multi_hop_max_dist, spatial_pos_max=args.spatial_pos_max, myargs=args)
    shuffle_idx = torch.randperm(len(dm.dataset_train))
    rand_train = Subset(dm.dataset_train, shuffle_idx.tolist())
    dm.dataset_train = rand_train
    train_loader, val_loader, test_loader = dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()
    print('Dataset load and process successfully')
    
    
       # Model 
    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.input_dropout_rate,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            attention_dropout_rate=args.attention_dropout_rate,     
            output_dim=dm.dataset["num_class"],
        ).to(device)
    elif args.model == "gt":
        model = GT(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.input_dropout_rate,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            attention_dropout_rate=args.attention_dropout_rate,     
            output_dim=dm.dataset["num_class"],
        ).to(device)
        
    print('Model params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_updates,
            tot=args.tot_updates,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)
    criterion = dm.dataset["loss_fn"]
    
    val_acc_list, test_acc_list, epoch_t_list, loss_list = [], [], [], []
    best_model, best_val, best_test = None, float('-inf'), float('-inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train(args, model, device, train_loader, optimizer, criterion, epoch, lr_scheduler)
        t1 = time.time()
        
        epoch_t_list.append(t1 - t0)
        print(f"Epoch: {epoch}, train loss: {train_loss}, epoch time: {np.mean(epoch_t_list)}")

        valid_mae = eval(args, model, device, val_loader, criterion, dm.dataset["evaluator"], dm.dataset["metric"], "val")
        test_mae = eval(args, model, device, test_loader, criterion, dm.dataset["evaluator"], dm.dataset["metric"], "test")
        val_acc_list.append(valid_mae)
        test_acc_list.append(test_mae)
        loss_list.append(train_loss)
        
        # exit(0)

    if not os.path.exists(f'./exps/{args.dataset}'): 
        os.makedirs(f'./exps/{args.dataset}')

    if args.attn_type != "hybrid":
        np.save(f'./exps/{args.dataset}/{str(args.attn_type)}_bs{args.batch_size}_e{args.epochs}_test', np.array(test_acc_list))
        # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
        np.save(f'./exps/{args.dataset}/{str(args.attn_type)}_bs{args.batch_size}_e{args.epochs}_loss', np.array(loss_list))
    else:
        np.save(f'./exps/{args.dataset}/{str(args.attn_type)}_{args.switch_freq}_s{args.batch_size}_e{args.epochs}_test', np.array(test_acc_list))
        # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
        np.save(f'./exps/{args.dataset}/{str(args.attn_type)}_{args.switch_freq}_s{args.batch_size}_e{args.epochs}_loss', np.array(loss_list))
    


if __name__ == "__main__":
    main()
