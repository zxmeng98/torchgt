# <p align="center">TorchGT</p>
TorchGT is an efficient, scalable, and accurate graph transformer training system. It intelligently realizes long sequence training with high efficiency in an algorithm and system co-design way. To learn more about how TorchGT works, please refer our paper.

# Artifact Evaluation

We provide code and document describing how to reproduce the key results presented in the SC'24 paper.

## Environment Preparation

For convenient artifact evaluation, we rent **a 4-GPU cloud server (4 x RTX 3090 24GB) with PCIe 4.0x16 lanes, AMD EPYC 7302P 16-Core CPU** for reviewers to reproduce the experiments. And we have already set up the environment and codebase on the server. 
The connection to the server is:
```bash
Host SC-AE
    HostName 185.158.179.210   
    User root
    Port 40061
```
After connection:
```bash
cd torchgt
conda activate gt
```

If you want to reproduce the results on your own machines, we provide two options to prepare the environment:

(Option 1) We provide a Docker image to run the experiments in a container. One can pull the image and run in a container.

```bash
docker pull zxmeng98/torchgt
docker run --gpus all -it zxmeng98/torchgt
```

(Option 2) Build the environment yourself. We suggest using a conda environment to install the dependencies.

```bash
conda create --name torchgt python=3.10
conda activate torchgt
cd torchgt
pip install -r requirements.txt
```
### Note

All experiment results are highly dependent on the GPUs used and network settings. 

## Table VI: Training Efficiency

To reproduce the training throughput and test accuracy of training $GPH_{Slim}$ on ogbn-arxiv with TorchGT in Table VI, one can use the following command line:

```bash
bash ./scripts/1_efficiency.sh
```
When training on ogbn-arxiv, we set the sequence length to 64K for $GPH_{Slim}$ and GT. This is the end-to-end evaluation results of training throughput and test accuracy of training graph transformer on ogbn-arxiv dataset. For faster job completion time, we use dummy input bias. 

The output of this script looks like this:

```bash
*****************************************
> initializing torch distributed ...
************ Finish sequence parallel group Initialization. ***********
Namespace(device=0, dataset_dir='./dataset/', dataset='ogbn-arxiv', model='graphormer', n_layers=4, num_heads=8, hidden_dim=64, ffn_dim=64, attn_bias_dim=1, dropout_rate=0.3, input_dropout_rate=0.1, attention_dropout_rate=0.5, num_global_node=1, attn_type='sparse', seq_len=64000, weight_decay=0.01, warmup_updates=10, tot_updates=70, epochs=2000, patience=50, peak_lr=0.0002, end_lr=1e-09, seed=42, perturb_feature=False, save_model=False, load_model=False, model_dir='./model_ckpt/', switch_freq=5, reorder=True, rank=0, local_rank=0, world_size=4, distributed_backend='nccl', distributed_timeout_minutes=10, sequence_parallel_size=4)
Dataset load successfully
Train nodes: 101605, Val nodes: 33869, Test nodes: 33869
Training iters: 2, Val iters: 1, Test iters: 1
Model params: 111913
[07:14:28] /opt/dgl/src/graph/transform/metis_partition_hetero.cc:89: Partition a graph with 64001 nodes and 526450 edges into 8 parts and get 92140 edge cuts
[07:14:33] /opt/dgl/src/graph/transform/metis_partition_hetero.cc:89: Partition a graph with 37606 nodes and 228357 edges into 8 parts and get 43888 edge cuts
------------------------------------------------------------------------------------
Epoch: 005, Loss: 3.5716, Epoch Time: 0.083s
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Eval time 0.36298489570617676s
...
...
Epoch: 1996, Loss: 1.0244, Epoch Time: 0.109s
Epoch: 1997, Loss: 1.0381, Epoch Time: 0.109s
Epoch: 1998, Loss: 1.0226, Epoch Time: 0.109s
Epoch: 1999, Loss: 1.0281, Epoch Time: 0.109s
Epoch: 2000, Loss: 1.0302, Epoch Time: 0.109s
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Eval time 0.4491288661956787s
Epoch: 2000, Loss: 1.030161, Train acc: 56.32%, Val acc: 54.36%, Test acc: 54.06%, Epoch Time: 0.109s
------------------------------------------------------------------------------------
Best validation accuracy: 54.52%, test accuracy: 54.35%
```


- From the output we can see the efficiency improvement (around 3.9x) and model quality maintenance by using TorchGT. This is mainly because TorchGT significantly reduces the computation complexity of the attention module.


## Figure 8(a): Maximum sequence length 

To explore the supported maximum sequence length w.r.t. 4 GPUs, one can use the following command line:
```bash
bash ./scripts/2_scale_a.sh
```

The output header of this script looks like below. On the cloud server, it needs about **8 minutes in the beginning to process the data.** 
```bash
*****************************************
> initializing torch distributed ...
************ Finish sequence parallel group Initialization. ***********
Namespace(device=0, dataset_dir='./dataset/', dataset='ogbn-products', model='graphormer', n_layers=4, num_heads=8, hidden_dim=64, ffn_dim=64, attn_bias_dim=1, dropout_rate=0.3, input_dropout_rate=0.1, attention_dropout_rate=0.5, num_global_node=1, attn_type='sparse', seq_len=900000, weight_decay=0.01, warmup_updates=10, tot_updates=70, epochs=500, patience=50, peak_lr=0.0002, end_lr=1e-09, seed=42, perturb_feature=False, save_model=False, load_model=False, model_dir='./model_ckpt/', switch_freq=5, reorder=True, rank=0, local_rank=0, world_size=4, distributed_backend='nccl', distributed_timeout_minutes=10, sequence_parallel_size=4)
Dataset load successfully
Train nodes: 1469417, Val nodes: 489806, Test nodes: 489806
Training iters: 2, Val iters: 1, Test iters: 1
Model params: 110576
[07:52:11] /opt/dgl/src/graph/transform/metis_partition_hetero.cc:89: Partition a graph with 850001 nodes and 17438212 edges into 8 parts and get 1287531 edge cuts
[07:56:04] /opt/dgl/src/graph/transform/metis_partition_hetero.cc:89: Partition a graph with 619418 nodes and 9802209 edges into 8 parts and get 845872 edge cuts
Epoch: 005, Loss: 3.8062, Epoch Time: 1.645s
Eval time 25.416417598724365s
Epoch: 005, Loss: 3.806199, Train acc: 11.88%, Val acc: 12.03%, Test acc: 11.94%, Epoch Time: 1.645s
Epoch: 006, Loss: 3.7869, Epoch Time: 1.644s
Epoch: 007, Loss: 3.7633, Epoch Time: 1.643s
Epoch: 008, Loss: 3.7344, Epoch Time: 1.642s
Epoch: 009, Loss: 3.7002, Epoch Time: 1.641s
Epoch: 010, Loss: 3.6609, Epoch Time: 1.641s
Eval time 25.33997654914856s
...
```

And the GPU statistics look like:
```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.40.07              Driver Version: 550.40.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        On  |   00000000:01:00.0 Off |                  N/A |
| 66%   69C    P2            236W /  300W |   20760MiB /  24576MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        On  |   00000000:81:00.0 Off |                  N/A |
| 62%   65C    P2            222W /  300W |   20680MiB /  24576MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 3090        On  |   00000000:82:00.0 Off |                  N/A |
| 67%   69C    P2            229W /  300W |   20680MiB /  24576MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 3090        On  |   00000000:C1:00.0 Off |                  N/A |
| 58%   63C    P2            233W /  300W |   20680MiB /  24576MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```


- We can successfully run the experiment with sequence length nearly 900K on 4 GPUs. And by oberserving the GPU memory statistics, we can see the the maximum sequence length of TorchGT can reach up to 900K on 4 GPUs. By changing the input argument to ```--nproc_per_node=2```, we can see Torch supports up to 600K sequence length on 2 GPUs. It can also enable the sequence length of 400K with only 1 GPU, substantially larger than that of GP-RAW. The sequence length of TorchGT almost scales linearly w.r.t. the number of GPUs.

## Figure 11(a): Attention Module Computation Time
To see the impact of elastic computation reformation module, we record the attention computation time corresponding to different sequence lengths. For attention module microbenchmarks, we choose attention computation on $S$=64K for illustration. One can use the following command to reproduce the experiments:

```bash
python attn_module.py --s 64000 --method torchgt
python attn_module.py --s 64000 --method sparse
python attn_module.py --s 64000 --method flash
```

The output of the scripts looks like this:

```bash
Start loading dataset
Attn Computation: 10.9179 ms with 160.06982 TFLOPS
Allocated memory: 99.6 MB
Reserved memory: 662.0 MB
```
After running the command, a trace.jason file using the Pytorch profiler tool will also be generated. Using this file, we can observe a more fine-grained time breakdown in attention computation.

One can also modify the input argument ```--s``` and ```--hn``` to get full results of Figuer 11. Please note that the printed attention computation time is not accurate, since it includes memory copy time which accounts for a large margin. Therefore, it is suggested to use the profiler tool to analyze the attention time more precisely.

- As the sequence length increases, the computation time of FlashAttention should grow quadratically. The sparse attention shares a similar computation speed as
FlashAttention when the sequence length is small. In contrast, TorchGT should improve the computation efficiency by a large margin compared to FlashAttention and sparse attention.



