torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level.py --dataset ogbn-products --seq_len 256000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 500 --attn_type sparse --reorder

torchrun --nproc_per_node=4 --master_port 1234 main_sp_node_level.py --dataset ogbn-arxiv --seq_len 64000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 2000 --attn_type sparse --reorder 





