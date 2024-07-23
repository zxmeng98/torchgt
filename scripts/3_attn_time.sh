# S = 64K
python attn_module.py --s 64000 --method torchgt # Cluster-sparse Attn
python attn_module.py --s 64000 --method sparse # Sparse Attn
python attn_module.py --s 64000 --method flash # Flash Attn

# S = 128K
python attn_module.py --s 128000 --method torchgt # Cluster-sparse Attn
python attn_module.py --s 128000 --method sparse # Sparse Attn
python attn_module.py --s 128000 --method flash # Flash Attn

# S = 512K
python attn_module.py --s 512000 --method torchgt # Cluster-sparse Attn
python attn_module.py --s 512000 --method sparse # Sparse Attn
python attn_module.py --s 512000 --method flash # Flash Attn