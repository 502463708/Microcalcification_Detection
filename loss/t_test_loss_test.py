import os
import time
import torch

from loss.t_test_loss import TTestLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def test_t_test_loss(num_test, batch_size, num_channels, height, width):
	for i in range(num_test):
		start_time = time.time()
		residues = torch.rand(batch_size, num_channels, height, width).cuda()
		image_level_labels = torch.rand(batch_size, 1).cuda()
		image_level_labels[image_level_labels <= 0.5] = 0
		image_level_labels[image_level_labels > 0.5] = 1
		image_level_labels = image_level_labels.byte()

		loss = TTestLoss()(residues, image_level_labels)
		print('time:', time.time() - start_time, 'loss: ', loss.item())

	return


if __name__ == '__main__':
	num_test = 100
	batch_size = 48
	num_channels = 1
	height = 112
	width = 112

	test_t_test_loss(num_test, batch_size, num_channels, height, width)
