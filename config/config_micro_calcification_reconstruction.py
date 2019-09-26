"""
This file contains all of the common configuration items involved in
training, validation, and test stages of calcification reconstruction.
"""

from easydict import EasyDict as edict


__C = edict()
cfg = __C

# general parameters
__C.general = {}
__C.general.data_root_dir = '/data/lars/data/Inbreast-dataset-cropped-pathches/'
__C.general.saving_dir = '/data/lars/models/20190925_uCs_reconstruction_ttestlossv3_default_dilation_radius_7/'
__C.general.cuda_device_idx = '6, 7'  # specify the index of the gpu devices to be occupied

# dataset parameters
__C.dataset = {}
__C.dataset.image_channels = 1  # this is a single-channel image
__C.dataset.cropping_size = [112, 112]  # [H, W] (pixel)
__C.dataset.enable_random_sampling = True  # True: randomly sample only during training
__C.dataset.pos_to_neg_ratio = 1  # hyper-parameter of randomly sampling
__C.dataset.dilation_radius = 7  # pixel-level label to be dilated, 0 -> will not be dilated

# data augmentation parameters
__C.dataset.augmentation = {}
__C.dataset.augmentation.enable_data_augmentation = True  # whether implement augmentation during training
__C.dataset.augmentation.enable_vertical_flip = True
__C.dataset.augmentation.enable_horizontal_flip = True

# loss
__C.loss = {}
__C.loss.name = 'TTestLossV3'  # only 'TTestLoss', 'TTestLossV2', 'TTestLossV3' is supported now
__C.loss.beta = 0.8
__C.loss.lambda_p = 1
__C.loss.lambda_n = 0.1

# net
__C.net = {}
__C.net.name = 'vnet2d_v2'  # name of the .py file implementing network architecture
__C.net.in_channels = 1
__C.net.out_channels = 1

# training parameters
__C.train = {}
__C.train.num_epochs = 501  # number of training epoch
__C.train.save_epochs = 10  # save ckpt every x epochs
__C.train.batch_size = 480
__C.train.num_threads = 8

# learning rate scheduler
__C.lr_scheduler = {}
__C.lr_scheduler.lr = 1e-3  # the initial learning rate
__C.lr_scheduler.step_size = 50  # decay learning rate every x epochs
__C.lr_scheduler.gamma = 0.95  # the learning rate decay

# the metrics related parameters
__C.metrics = {}
__C.metrics.prob_threshold = 0.2  # residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1
__C.metrics.area_threshold = 3.14 * 14 * 14 / 3  # connected components whose area < area_threshold will be discarded
__C.metrics.distance_threshold = 14  # candidates (distance between calcification < distance_threshold) is recalled

# the visdom related parameters
__C.visdom = {}
__C.visdom.port = 9999  # port of visdom
__C.visdom.update_batches = 5  # update images displayed in visdom every x batch


'''
One NVDIA TITAN XP with 12 Mb memory can bear a training load with batch_size=260 
'''