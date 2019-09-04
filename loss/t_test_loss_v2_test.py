import os
import time
import torch

from loss.t_test_loss_v2 import TTestLossV2

os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def test_t_test_loss_v2(num_test, batch_size, num_channels, height, width, beta, lambda_p, lambda_n):
    for i in range(num_test):
        loss_func = TTestLossV2(beta, lambda_p, lambda_n)

        start_time = time.time()
        residues = torch.rand(batch_size, num_channels, height, width).cuda()

        image_level_labels = torch.rand(batch_size, 1).cuda()
        image_level_labels[image_level_labels <= 0.5] = 0
        image_level_labels[image_level_labels > 0.5] = 1
        image_level_labels = image_level_labels

        pixel_level_labels = torch.rand(batch_size, height, width).cuda()
        pixel_level_labels[pixel_level_labels <= 0.5] = 0
        pixel_level_labels[pixel_level_labels > 0.5] = 1

        loss = loss_func(residues, image_level_labels, pixel_level_labels)
        print('time:', time.time() - start_time, 'loss: ', loss.item())
        print('\n')

    return


if __name__ == '__main__':
    num_test = 100
    batch_size = 3
    num_channels = 1
    height = 112
    width = 112

    beta = 0.8
    lambda_p = 1
    lambda_n = 0.1

    test_t_test_loss_v2(num_test, batch_size, num_channels, height, width, beta, lambda_p, lambda_n)
