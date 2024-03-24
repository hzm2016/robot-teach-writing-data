#!/usr/bin/python3

import argparse
import os
import torch
from models import CycleGAN


def main(args):

    key_pairs = {
        'G_A2B': '/home/cunjun/Robot-Teaching-Assiantant/gan/std_cyc_gan/G_A2B_199.pth',
        'G_B2A': '/home/cunjun/Robot-Teaching-Assiantant/gan/std_cyc_gan/G_B2A_199.pth'
    }

    model = eval(args.model)(args, train=False)
    model.load_networks(key_pairs)
    model.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CycleGAN', help='type of models')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='data/lantingkai/',
                        help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=1,
                        help='number of channels of input font_data')
    parser.add_argument('--output_nc', type=int, default=1,
                        help='number of channels of output font_data')
    parser.add_argument('--size', type=int, default=128,
                        help='size of the font_data (squared assumed)')
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--sequential', action='store_true',
                        help='if the dataset have a sequence')
    parser.add_argument('--ske', action='store_true',
                        help='if the dataset have a skeleton input')
    parser.add_argument('--output_dir', type=str,
                        default='./output', help='place to output result')
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists(args.output_dir + '/A'):
        os.makedirs(args.output_dir + '/A')
    if not os.path.exists(args.output_dir + '/B'):
        os.makedirs(args.output_dir + '/B')

    main(args)

