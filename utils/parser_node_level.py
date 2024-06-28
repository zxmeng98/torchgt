import argparse


def parser_add_main_args(parser):
   
    # main args
    parser.add_argument('--device', type=int, default=0, help='device id')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--dataset', type=str, default='pubmed')

    # model args
    parser.add_argument('--model', type=str, default="graphormer")
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--attn_bias_dim', type=int, default=1) # must match M power adj in preprocess_data
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--input_dropout_rate', type=float, default=0.1)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_global_node', type=int, default=1)
    parser.add_argument('--attn_type', type=str, default="sparse", help='whether to use sparse attention')

    # training args
    parser.add_argument('--seq_len', type=int, default=256000, help='total sequence length here')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_updates', type=int, default=10,
                        help='warmup steps for optimizer learning rate scheduling')
    parser.add_argument('--tot_updates',  type=int, default=70,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--epochs', type=int, default=100) # larger seq len more training epochs
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    parser.add_argument('--peak_lr', type=float, default=2e-4)  
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perturb_feature', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False, help='whether to save model')
    parser.add_argument('--load_model', action='store_true', default=False, help='whether to load saved model')
    parser.add_argument('--model_dir', type=str, default='./model_ckpt/')
    parser.add_argument('--switch_freq', type=int, default=5)
    parser.add_argument('--reorder', action='store_true', default=False,
                        help='TorchGT mode')
    
    # distributed args
    parser.add_argument('--rank', type=int, default=None,
                       help='rank passed from distributed launcher.')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    parser.add_argument('--world-size', type=int, default=None,
                       help='world size of sequence parallel group.')
    parser.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'ccl'],
                       help='Which backend to use for distributed training.')
    parser.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    parser.add_argument('--sequence-parallel-size', type=int, default=4,
                       help='Enable DeepSpeed\'s sequence parallel.')