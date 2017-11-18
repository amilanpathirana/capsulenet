import argparse
import torch


parser = argparse.ArgumentParser(description='Example of CapsNet')

parser.add_argument('--epochs',       type=int,   default=5)
parser.add_argument('--data_path',    type=str,   default='./data')
parser.add_argument('--batch_size',   type=int,   default=128)
parser.add_argument('--use_gpu',      type=bool,  default=torch.cuda.is_available())

parser.add_argument('--lr',           type=float, default=0.01,
                    help='ADAM learning rate (0.01)')
parser.add_argument('--log_interval', type=int,   default=10,
                    help='number of batches between logging')
parser.add_argument('--visdom',       type=bool,  default=False,
                    help='Whether or not to use visdom for plotting progrss')
parser.add_argument('--dataset',   type=str,      default='MNIST',
                    help='The dataset to train on, currently supported: MNIST, Fashion MNIST')


args = parser.parse_args()
