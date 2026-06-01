import argparse
import importlib
from utils import *
import traceback
import os
import torch

MODEL_DIR = None
DATA_DIR = 'data/'
PROJECT = 'cdi' 


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=10)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-lrw', type=float, default=0.1)
    parser.add_argument('-lrb', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=32)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the available training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos')  # using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos')  # using average data embedding and cosine classifier

    # for Contrastive Learning parameters
    parser.add_argument('-moco_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('-moco_k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('-moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('-moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('-mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument("-num_crops", type=int, default=[2, 4], nargs="+",
                        help="amount of crops")
    parser.add_argument("-size_crops", type=int, default=[224, 96], nargs="+",
                        help="resolution of inputs")
    parser.add_argument("-min_scale_crops", type=float, default=[0.14, 0.05], nargs="+",
                        help="min area of crops")
    parser.add_argument("-max_scale_crops", type=float, default=[1, 0.14], nargs="+",
                        help="max area of crops")
    parser.add_argument('-constrained_cropping', action='store_true',
                        help='condition small crops on key crop')
    parser.add_argument('-auto_augment', type=int, default=[], nargs='+',
                        help='Apply auto-augment 50% of times to the selected crops')
    parser.add_argument('-fantasy', type=str, default='rotation', help='fantasy method to generate virtual classes')
    parser.add_argument('-alpha', type=float, default=0.5, help='coefficient of the global contrastive loss')
    parser.add_argument('-beta', type=float, default=0.5, help='coefficient of the local contrastive loss')

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-incft', action='store_true', help='incremental finetuning')

    # Parameters for Causal Module (CDI)
    parser.add_argument('-causal_weight', type=float, default=0.1,
                        help='Weight coefficient to control causal feature diversity loss')
    parser.add_argument('-recon_weight', type=float, default=0.1,
                        help='Weight coefficient to control feature reconstruction loss')

    # Parameters for Counterfactual Intervention
    parser.add_argument('-counterfactual_weight', type=float, default=0.1,
                        help='Weight of the counterfactual intervention loss')
    parser.add_argument('-counterfactual_alpha', type=float, default=0.8,
                        help='Counterfactual feature mixing ratio/strength')
    parser.add_argument('-use_counterfactual', action='store_true',
                        help='Whether to enable counterfactual intervention training')
    parser.add_argument('-distill_temp', type=float, default=0.5,
                        help='Temperature coefficient for contrastive distillation')
    parser.add_argument('-counterfactual_mode', type=str, default='domain_shift',
                        choices=['adaptive_noise', 'masking', 'style_mixing',
                                 'feature_corruption', 'domain_shift', 'helper'],
                        help='Mode for generating counterfactual samples')
    return parser


if __name__ == '__main__':
    # Set thread parameters at the very beginning of the program
    os.environ["OMP_NUM_THREADS"] = "4"       # Control OpenMP parallelism
    os.environ["MKL_NUM_THREADS"] = "4"       # Control MKL mathematical operations
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Control OpenBLAS operations
    torch.set_num_threads(4)                  # Set PyTorch main thread count

    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    # Need to match the number of physical CPU cores
    args.num_workers = 4  

    # Dynamically import the trainer based on the project name
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
