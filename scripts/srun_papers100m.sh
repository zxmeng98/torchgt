#!/usr/bin/env sh
export HOME=/mnt/petrelfs/zhangmeng/
echo $HOME


mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

# srun --partition=llm_s --gres=gpu:4 -n1 --ntasks-per-node=1 --job-name=paper100m --kill-on-bad-exit=1 \
srun python main_sp_papers100m.py --dataset ogbn-papers100M --seq_len 256000 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 50 --attn_type flash --sequence-parallel-size 8 \
2>&1|tee a800log/gp-sp8-flash-papers100m.log

