# ================================== Graph-level ==================================
# Graphormer
python main_sp_sparse_malnet.py --dataset MalNet --dataset_dir="/home/mzhang/work/sj/" --batch_size 16  --attn_type sparse --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 200 --num_workers 0 --reorder

python main_graph_level.py --dataset MalNet --dataset_dir="/home/mzhang/work/sj/" --batch_size 8 --model gt --device 0 --reorder False

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=3 main_sp_sparse_graph_level.py --dataset MalNet --batch_size 16 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 4 --num_workers 8 --dataset_dir="/home/mzhang/work/sj/" --reorder

# GT
python main_sp_sparse_malnet.py --dataset MalNet --dataset_dir="/home/mzhang/work/sj/" --batch_size 8 --model gt --attn_type sparse --n_layers 4 --hidden_dim 128 --num_heads 8 --epochs 500 --reorder --num_workers 2

torchrun --nproc_per_node=4 main_sp_sparse_graph_level.py --dataset MalNet --batch_size 8  --n_layers 4 --hidden_dim 128 --num_heads 8 --num_workers 8 --epochs 200 --dataset_dir="/home/mzhang/work/sj/" --model gt --reorder


# ================================== Node-level ==================================
# Graphormer
python main_node_level.py --dataset ogbn-products  --seq_len 000 --epochs 100 --device 0 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --attn_type full

python main_node_level.py --dataset ogbn-products  --seq_len 400000 --epochs 100 --device 0 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --attn_type sparse --reorder

python main_node_level.py --dataset ogbn-products  --seq_len 60000 --epochs 100 --device 0 --n_layers 12 --hidden_dim 768 --ffn_dim 768 --num_heads 32 --attn_type flash

torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level_sparse_dummybias.py --dataset ogbn-products --seq_len 32000 --n_layers 12 --hidden_dim 768 --ffn_dim 768 --num_heads 32 --epochs 500 --attn_type sparse --reorder

torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level_sparse_dummybias.py --dataset ogbn-products --seq_len 256000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 500 --attn_type sparse --reorder

torchrun --nproc_per_node=3 --master_port 1234 main_sp_node_level.py --dataset ogbn-arxiv --seq_len 64000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 2000 --attn_type sparse --reorder 

torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level_sparse_dummybias.py --dataset amazon --seq_len 256000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --attn_type sparse --reorder 

python main_sp_papers100m.py --dataset ogbn-papers100M --seq_len 256000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 200 --attn_type flash 2>&1|tee 3090log/gp-sp4-flash-papers.log 

python main_sp_papers100m.py --dataset ogbn-papers100M --seq_len 256000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 200 --attn_type sparse --reorder

# GT
torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level_sparse_dummybias.py --dataset ogbn-products --seq_len 256000 --n_layers 4 --hidden_dim 128 --num_heads 8 --epochs 1000 --attn_type sparse --model gt

torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level_sparse_dummybias.py --dataset ogbn-arxiv --seq_len 64000 --n_layers 4 --hidden_dim 128 --num_heads 8 --epochs 2000 --model gt --attn_type sparse --reorder 2>&1|tee 3090log/test.log 

torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level_sparse_dummybias.py --dataset ogbn-products --seq_len 256000 --n_layers 4 --hidden_dim 128 --num_heads 8 --epochs 1000 --attn_type sparse --model gt 2>&1|tee 3090log/test.log 

torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level_sparse_dummybias.py --dataset ogbn-products --seq_len 256000 --n_layers 4 --hidden_dim 128 --num_heads 8 --epochs 1500 --attn_type flash --model gt

torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level_sparse_dummybias.py --dataset amazon --seq_len 256000 --n_layers 4 --hidden_dim 128 --num_heads 8 --epochs 1000 --attn_type sparse --model gt

python main_sp_papers100m.py --dataset ogbn-papers100M --seq_len 256000 --n_layers 4 --hidden_dim 128 --num_heads 8 --epochs 200 --attn_type flash --model gt

pkill -9 python3 && pkill -9 python


# multi-node
torchrun --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr "192.168.1.40" \
    --master_port 2923 \
    main_sp_node_level_sparse_dummybias.py --dataset ogbn-products --seq_len 1300000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 500 --attn_type flash

torchrun --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr "192.168.1.40" \
    --master_port 2923 \
    main_sp_node_level_sparse_dummybias.py --dataset ogbn-products --seq_len 1300000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 500 --attn_type flash
