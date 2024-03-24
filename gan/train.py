#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import torch
from models import CycleGAN, StyleCycleGAN

def main(args):

    model = eval(args.model)(args)
    model.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--model', type=str, default='CycleGAN', help='type of models')
    parser.add_argument('--n_epochs', type=int, default=150,
                        help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=12,
                        help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./gan/data/seq',
                        help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=149,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=128,
                        help='size of the font_data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1,
                        help='number of channels of input font_data')
    parser.add_argument('--output_nc', type=int, default=1,
                        help='number of channels of output font_data')
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=1,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--frequency', type=int, default=10,
                        help='frequency of saving trained model')
    parser.add_argument('--ske', action='store_true',
                        help='if the dataset have a skeleton input')
    parser.add_argument('--output_dir', type=str,
                        default='./output', help='place to output result')
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
