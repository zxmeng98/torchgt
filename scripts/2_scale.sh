# TorchGT on 4 GPUs
torchrun --nproc_per_node=4 main_sp_sparse_node_level.py --dataset ogbn-products --seq_len 850000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 500 --attn_type sparse --reorder

# TorchGT on 2 GPUs
# torchrun --nproc_per_node=2 main_sp_sparse_node_level.py --dataset ogbn-products --seq_len 600000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 500 --attn_type sparse --reorder 

# TorchGT on 1 GPU
# torchrun --nproc_per_node=1 main_sp_sparse_node_level.py --dataset ogbn-products --seq_len 400000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 500 --attn_type sparse --reorder