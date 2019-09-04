import os
import torch
from torch.autograd import Variable
from net.vnet2d_v2 import VNet2d

kMega = 1e6


def TestVdnet2dOutputChannels(batch_size, in_channels=1, out_channels=3, dim_x=512, dim_y=512):
	assert in_channels > 0
	assert out_channels > 0

	model = VNet2d(num_in_channels=in_channels, num_out_channels=out_channels)
	model = torch.nn.DataParallel(model).cuda()
	model = model.cuda()
	print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / kMega))

	in_images = torch.zeros([batch_size, in_channels, dim_y, dim_x])
	in_images = in_images.cuda()
	in_images_v = Variable(in_images, requires_grad=False)

	reconstructions, residues = model(in_images_v)
	assert reconstructions.size()[0] == batch_size
	assert reconstructions.size()[1] == out_channels

	print("input shape = ", in_images.shape)
	print("reconstructions shape = ", reconstructions.shape)
	print("residue shape = ", residues.shape)
	print("min value of reconstructions = ", reconstructions.min())
	print("max value of reconstructions = ", reconstructions.max())

	# debug only
	# outputs = outputs.squeeze().cpu().detach().numpy()
	# print(outputs)


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	batch_size = 12
	in_channels = 1
	out_channels = 1
	dim_x = 112
	dim_y = 112

	while True:
		TestVdnet2dOutputChannels(batch_size,
								  in_channels,
								  out_channels,
								  dim_x,
								  dim_y)

# TestVdnet2dOutputChannels(10, 1, 512, 512)
