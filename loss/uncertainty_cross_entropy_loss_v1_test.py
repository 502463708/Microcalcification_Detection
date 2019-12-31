import argparse
import os
import time
import torch

from loss.uncertainty_cross_entropy_loss_v1 import UncertaintyCrossEntropyLossV1

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_test',
                        type=int,
                        default=100,
                        help='number of test')

    parser.add_argument('--batch_size',
                        type=int,
                        default=3,
                        help='number of patches in each batch')

    parser.add_argument('--num_classes',
                        type=int,
                        default=1,
                        help='1 for grayscale, 3 for RGB images')

    args = parser.parse_args()

    return args


def TestUncertaintyTTestLossV1(args):
    for i in range(args.num_test):
        loss_func = UncertaintyCrossEntropyLossV1()

        start_time = time.time()
        residues = torch.rand(args.batch_size, args.num_channels, args.height, args.width).cuda()

        pixel_level_labels = torch.rand(args.batch_size, args.height, args.width).cuda()
        pixel_level_labels[pixel_level_labels <= 0.5] = 0
        pixel_level_labels[pixel_level_labels > 0.5] = 1
        pixel_level_labels = pixel_level_labels.long()

        uncertainty_maps = torch.rand(args.batch_size, args.height, args.width).cuda()

        loss = loss_func(residues, pixel_level_labels, uncertainty_maps)
        print('time:', time.time() - start_time, 'loss: ', loss.item())
        print('\n')

    return


if __name__ == '__main__':
    args = ParseArguments()
    TestUncertaintyTTestLossV1(args)
