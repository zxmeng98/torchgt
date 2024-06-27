for SEQ in 512 1024 2048 4096 8192 16384 
do 
    echo -e "\033[1mclean python processes\033[0m"
    sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
    echo -e "\033[1mseq length ${SEQ}\033[0m"
    torchrun --nproc_per_node=4 main_sp.py --dataset aminer --seq_len $SEQ --peak_lr 0.001 --end_lr 0.0001 --epochs 500
done

torchrun --nproc_per_node=4 main_sp.py --dataset aminer --seq_len 512 --peak_lr 0.001 --end_lr 0.0001 --epochs 500