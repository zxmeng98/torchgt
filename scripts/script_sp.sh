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