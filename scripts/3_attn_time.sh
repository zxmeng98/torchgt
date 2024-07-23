python attn_module.py --s 64000 --method torchgt # Cluster-sparse Attn
python attn_module.py --s 64000 --method sparse # Sparse Attn
python attn_module.py --s 64000 --method flash # Flash Attn

python attn_module.py --s 128000 --method torchgt # Cluster-sparse Attn
python attn_module.py --s 128000 --method sparse # Sparse Attn
python attn_module.py --s 128000 --method flash # Flash Attn

python attn_module.py --s 512000 --method torchgt # Cluster-sparse Attn
python attn_module.py --s 512000 --method sparse # Sparse Attn
python attn_module.py --s 512000 --method flash # Flash Attn