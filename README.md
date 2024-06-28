# TorchGT
TorchGT is an efficient, scalable, and accurate graph transformer training system. To learn more about how TorchGT works, please refer our paper.


# Artifact Evaluation

## Installation

(Option1) For convenient artifact evaluation, we will rent a 4-GPU cloud server (4x RTX 3090) for reviewers to reproduce all the experiments. And we provide a Docker image to run the experiments in a container:

(comming soon...)

(Option2) Build the environment yourself: We suggest using a conda environment to install the dependencies.

```bash
conda create --name torchgt python=3.10
conda activate torchgt
cd torchgt
pip install -r requirements.txt
```


## Run the code  
Environment: gt

### Graph-level
```bash
torchrun --nproc_per_node=4 main_sp_graph_level.py --dataset MalNet
```
 
### Node-level
Use preprocess_data.py for data downloading and preprocessing

```bash
torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level.py --dataset ogbn-products
```

More coming...



