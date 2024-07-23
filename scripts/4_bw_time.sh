# S = 64K
python attn_module.py --s 64000 --method sparse # Topology-pattern BW. Time
python attn_module.py --s 64000 --method torchgt # Dense BW. Time

# S = 128K
python attn_module.py --s 128000 --method sparse # Topology-pattern BW. Time,
python attn_module.py --s 128000 --method torchgt # Dense BW. Time

# S = 256K
python attn_module.py --s 256000 --method sparse # Topology-pattern BW. Time
python attn_module.py --s 256000 --method torchgt # Dense BW. Time

# S = 512K
python attn_module.py --s 512000 --method sparse # Topology-pattern BW. Time
python attn_module.py --s 512000 --method torchgt # Dense BW. Time