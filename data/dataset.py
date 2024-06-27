# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data.collator import collator
from data.collator_malnet import collator_malnet
from data.wrapper import (
    MyGraphPropPredDataset,
    MyPygPCQM4MDataset,
    MyZINCDataset,
    MyCoraDataset,
    MyMalNetTiny,
    MyGNNBenchmarkDataset,
    MalNet,
)

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import ogb
import ogb.lsc
import ogb.graphproppred
from ogb.nodeproppred import Evaluator as NodePropPredEvaluator
from functools import partial
import torch.distributed as dist
import random
from tqdm import tqdm

dataset = None


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG full dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def get_dataset(dataset_name="abaaba", dataset_dir_raw="./dataset/"):
    dataset_dir = dataset_dir_raw + f"{dataset_name}/"
    global dataset
    if dataset is not None:
        return dataset

    # max_node is set to max(max(num_val_graph_nodes), max(num_test_graph_nodes))
    if dataset_name == "ogbg-molpcba":
        dataset = {
            "num_class": 128,
            "loss_fn": F.binary_cross_entropy_with_logits,
            "metric": "ap",
            "metric_mode": "max",
            "evaluator": ogb.graphproppred.Evaluator("ogbg-molpcba"),
            "dataset": MyGraphPropPredDataset("ogbg-molpcba", root=dataset_dir),
            "max_node": 128,
        }
    elif dataset_name == "ogbg-molhiv":
        dataset = {
            "num_class": 1,
            "loss_fn": F.binary_cross_entropy_with_logits,
            "metric": "rocauc",
            "metric_mode": "max",
            "evaluator": ogb.graphproppred.Evaluator("ogbg-molhiv"),
            "dataset": MyGraphPropPredDataset("ogbg-molhiv", root=dataset_dir),
            "max_node": 128,
        }
    elif dataset_name == "PCQM4M-LSC":
        dataset = {
            "num_class": 1,
            "loss_fn": F.l1_loss,
            "metric": "mae",
            "metric_mode": "min",
            "evaluator": ogb.lsc.PCQM4MEvaluator(),
            "dataset": MyPygPCQM4MDataset(root=dataset_dir),
            "max_node": 128,
        }
    elif dataset_name == "ZINC":
        dataset = {
            "num_class": 1,
            "loss_fn": F.l1_loss,
            "metric": "mae",
            "metric_mode": "min",
            "evaluator": ogb.lsc.PCQM4MEvaluator(),  # same objective function, so reuse it
            "train_dataset": MyZINCDataset(
                subset=True, root=dataset_dir, split="train" # subset=True, train: 10000, val/test: 1000
            ),
            "valid_dataset": MyZINCDataset(
                subset=True, root=dataset_dir, split="val"
            ),
            "test_dataset": MyZINCDataset(
                subset=True, root=dataset_dir, split="test"
            ),
            "max_node": 128,
        }
    elif dataset_name == "CORA":
        dataset = {
            "num_class": 7,
            "loss_fn": F.cross_entropy,
            "metric": "cross_entropy",
            "metric_mode": "min",
            "evaluator": NodePropPredEvaluator(name="ogbn-arxiv"),
            "dataset": MyCoraDataset(
                name="Cora", root=dataset_dir, split="public"
            ),
            "max_node": 2708,
        }
    elif dataset_name == "MalNetTiny":
        # TODO collator改为dummy bias的
        dataset = {
            "num_class": 5,
            "loss_fn": F.cross_entropy,
            "metric": "accuracy",
            "metric_mode": "max",
            "evaluator": None,
            "train_dataset": MyMalNetTiny(
                root=dataset_dir, split="train"
            ),
            "valid_dataset": MyMalNetTiny(
                root=dataset_dir, split="val"
            ),
            "test_dataset": MyMalNetTiny(
                root=dataset_dir, split="test"
            ),      
            "max_node": 10000,
        }
        # pre_transform_in_memory(dataset["train_dataset"], preprocess_item_malnet, show_progress=True)
        # print(dataset[0])
        # exit(0)

    elif dataset_name == "MalNet":
        dataset_dir = dataset_dir_raw
        malnet_dataset = MalNet(root=dataset_dir)
        split_dict = malnet_dataset.get_idx_split() # train: 691, val: 691, test: 5528
        train_idx, val_idx, test_idx = split_dict['train'], split_dict['valid'], split_dict['test']
        dataset = {
            "num_class": 5,
            "loss_fn": F.cross_entropy,
            "metric": "accuracy",
            "metric_mode": "max",
            "evaluator": None,
            "train_dataset": malnet_dataset[train_idx],
            "valid_dataset": malnet_dataset[val_idx],
            "test_dataset": malnet_dataset[test_idx],      
            "max_node": 1e10,
        }

    elif dataset_name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        dataset = {
            "num_class": 10,
            "loss_fn": F.cross_entropy,
            "metric": "accuracy",
            "metric_mode": "max", 
            "evaluator": None,
            "train_dataset": MyGNNBenchmarkDataset(
                root=dataset_dir, name=dataset_name, split="train"
            ),
            "valid_dataset": MyGNNBenchmarkDataset(
                root=dataset_dir, name=dataset_name, split="val"
            ),
            "test_dataset": MyGNNBenchmarkDataset(
                root=dataset_dir, name=dataset_name, split="test"
            ),      
            "max_node": 10000,
        }
        
    else:
        raise NotImplementedError
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f" > {dataset_name} loaded!")
        # print(dataset)
        # print(f" > dataset info ends")
    return dataset


class GraphormerDataset:
    def __init__(
        self,
        dataset_name: str = "ogbg-molpcba",
        dataset_dir: str = "./dataset",
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        spatial_pos_max: int = 1024,
        myargs = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name, dataset_dir)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ... # x 是一个占位符或默认值
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.myargs = myargs
        self.setup()

    def setup(self):
        if self.dataset_name in ["ZINC", "MalNetTiny", "CIFAR10", "MalNet"]:
            self.dataset_train = self.dataset["train_dataset"]
            self.dataset_val = self.dataset["valid_dataset"]
            self.dataset_test = self.dataset["test_dataset"]

        elif self.dataset_name == "CORA":
            train_mask = self.dataset["dataset"].data.train_mask
            val_mask = self.dataset["dataset"].data.val_mask
            test_mask = self.dataset["dataset"].data.test_mask
        else:
            split_idx = self.dataset["dataset"].get_idx_split()
            self.dataset_train = self.dataset["dataset"][split_idx["train"]]
            self.dataset_val = self.dataset["dataset"][split_idx["valid"]]
            self.dataset_test = self.dataset["dataset"][split_idx["test"]]

    def train_dataloader(self):
        
        if self.dataset_name == "CORA":

            loader = DataLoader(
                self.dataset["dataset"],
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=partial(
                    collator,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        elif self.dataset_name == "MalNet":
            loader = DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_malnet, # collator是在生成batch的时候会被调用: for batch in loader
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        else:
            loader = DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator, # collator是在生成batch的时候会被调用: for batch in loader
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("len(train_dataloader)", len(loader))
        return loader

    def val_dataloader(self):

        if self.dataset_name == "CORA":

            loader = DataLoader(
                self.dataset["dataset"],
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=partial(
                    collator,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        elif self.dataset_name == "MalNet":
            loader = DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_malnet,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        else:
            loader = DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=partial(
                    collator,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("len(val_dataloader)", len(loader))
        return loader

    def test_dataloader(self):

        if self.dataset_name == "CORA":

            loader = DataLoader(
                self.dataset["dataset"],
                batch_size=1,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        elif self.dataset_name == "MalNet":
            loader = DataLoader(
                self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_malnet,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        else:
            loader = DataLoader(
                self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=partial(
                    collator,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    spatial_pos_max=self.spatial_pos_max,
                    myargs = self.myargs,
                ),
            )
        if not dist.is_initialized() or dist.get_rank() == 0:    
            print("len(test_dataloader)", len(loader))
        return loader
