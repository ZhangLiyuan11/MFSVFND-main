
import argparse
import os
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch


from run import Run

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='fakesv', help='fakett/fakesv')
parser.add_argument('--epoches', type=int, default=30)
parser.add_argument('--batch_size', type = int, default=16)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epoch_stop', type=int, default=5) 
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--lambd', type=float, default=0.2)
parser.add_argument('--dropout', type=float, default=0.1) 
parser.add_argument('--weight_decay', type=float, default=5e-5)

parser.add_argument('--path_param', default= './check_points/')
parser.add_argument('--path_tensorboard', default= './tb/')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print (args)

config = {
        'dataset':args.dataset,
        'epoches': args.epoches,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epoch_stop': args.epoch_stop,
        'seed': args.seed,
        'device': args.gpu,
        'lr': args.lr,
        'lambd': args.lambd,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'path_param': args.path_param,
        'path_tensorboard': args.path_tensorboard,
        }


if __name__ == '__main__':
    Run(config = config
        ).main()
