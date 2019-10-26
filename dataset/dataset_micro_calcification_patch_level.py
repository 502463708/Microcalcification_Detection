import cv2
import numpy as np
import os
import random
import torch

from common.utils import dilate_image_level_label
from skimage import measure
from torch.utils.data import Dataset


class MicroCalcificationDataset(Dataset):
    def __init__(self, data_root_dir, mode, enable_random_sampling, pos_to_neg_ratio, image_channels, cropping_size,
                 dilation_radius, calculate_micro_calcification_number, enable_data_augmentation,
                 enable_vertical_flip=False, enable_horizontal_flip=False):

        super(MicroCalcificationDataset, self).__init__()

        # the data root path must exist
        assert os.path.isdir(data_root_dir)

        # the mode must be one of 'training', 'validation' or 'test'
        assert mode in ['training', 'validation', 'test']
        self.mode = mode

        # the image directory of positive and negative patches respectively
        self.positive_patch_image_dir = os.path.join(data_root_dir, 'positive_patches', self.mode, 'images')
        self.negative_patch_image_dir = os.path.join(data_root_dir, 'negative_patches', self.mode, 'images')

        # the image filename list of positive and negative patches respectively
        self.positive_patch_filename_list = self.generate_and_check_filename_list(self.positive_patch_image_dir)
        self.negative_patch_filename_list = self.generate_and_check_filename_list(self.negative_patch_image_dir)

        # the mixed image filename list including the positive and negative patches
        self.mixed_patch_filename_list = self.positive_patch_filename_list + self.negative_patch_filename_list
        self.mixed_patch_filename_list.sort()

        # enable_random_sampling must be a bool variable
        assert isinstance(enable_random_sampling, bool)
        self.enable_random_sampling = enable_random_sampling

        # pos_to_neg_ratio must be a positive number
        assert pos_to_neg_ratio > 0
        self.pos_to_neg_ratio = pos_to_neg_ratio

        # image_channels must be a positive number
        assert image_channels > 0
        self.image_channels = image_channels

        # cropping_size contains length of height and width
        assert len(cropping_size) == 2
        self.cropping_size = cropping_size

        # dilation_radius must be a non-negative integer
        assert dilation_radius >= 0
        assert dilation_radius == int(dilation_radius)
        self.dilation_radius = dilation_radius

        # calculate_micro_calcification_number must be a bool variable
        assert isinstance(calculate_micro_calcification_number, bool)
        self.calculate_micro_calcification_number = calculate_micro_calcification_number

        # enable_data_augmentation is a bool variable
        assert isinstance(enable_data_augmentation, bool)
        self.enable_data_augmentation = enable_data_augmentation

        # enable_vertical_flip is a bool variable
        assert isinstance(enable_vertical_flip, bool)
        self.enable_vertical_flip = enable_vertical_flip

        # enable_horizontal_flip is a bool variable
        assert isinstance(enable_horizontal_flip, bool)
        self.enable_horizontal_flip = enable_horizontal_flip

    def __getitem__(self, index):
        """
        :param index
        :return: Tensor
        """
        if self.enable_random_sampling:
            # calculate probability threshold for the random number according the specified pos_to_neg_ratio
            prob_threshold = self.pos_to_neg_ratio / (self.pos_to_neg_ratio + 1)

            # sample positive or negative patch randomly
            if random.random() >= prob_threshold:
                filename = random.sample(self.positive_patch_filename_list, 1)[0]
                image_path = os.path.join(self.positive_patch_image_dir, filename)
                image_level_label = [1]
            else:
                filename = random.sample(self.negative_patch_filename_list, 1)[0]
                image_path = os.path.join(self.negative_patch_image_dir, filename)
                image_level_label = [0]
        else:
            # sample an image file from mixed_patch_filename_list
            filename = self.mixed_patch_filename_list[index]

            # firstly assume this is a positive patch
            image_path = os.path.join(self.positive_patch_image_dir, filename)
            image_level_label = [1]

            # turn it to a negative one, if positive patch directory doesn't contain this image
            if not os.path.exists(image_path):
                image_path = os.path.join(self.negative_patch_image_dir, filename)
                image_level_label = [0]

        # get the corresponding pixel-level label path
        pixel_level_label_path = image_path.replace('images', 'labels')

        # check the existence of the sampled patch
        assert os.path.exists(image_path)
        assert os.path.exists(pixel_level_label_path)

        # load image
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_np = image_np.astype(np.float)

        # normalize the intensity range of image: [0, 255] -> [0, 1]
        image_np /= 255.0

        # load pixel-level label
        pixel_level_label_np = cv2.imread(pixel_level_label_path, cv2.IMREAD_GRAYSCALE)
        pixel_level_label_np = pixel_level_label_np.astype(np.float)
        if pixel_level_label_np.max() == 255:
            pixel_level_label_np /= 255.0

        # check the consistency of size between image and its pixel-level label
        assert image_np.shape == pixel_level_label_np.shape

        # implement data augmentation only when the variable enable_data_augmentation is set True
        if self.enable_data_augmentation:
            if self.enable_vertical_flip and random.random() >= 0.5:
                image_np = np.flipud(image_np)
                pixel_level_label_np = np.flipud(pixel_level_label_np)
            if self.enable_horizontal_flip and random.random() >= 0.5:
                image_np = np.fliplr(image_np)
                pixel_level_label_np = np.fliplr(pixel_level_label_np)

        # guarantee image_np and pixel_level_label_np keep contiguous after data augmentation
        image_np = np.ascontiguousarray(image_np)
        pixel_level_label_np = np.ascontiguousarray(pixel_level_label_np)

        # dilate pixel-level label only when the variable dilation_radius is a positive integer
        if self.dilation_radius > 0:
            pixel_level_label_dilated_np = dilate_image_level_label(pixel_level_label_np, self.dilation_radius)
        else:
            pixel_level_label_dilated_np = pixel_level_label_np

        # calculate the number of the annotated micro calcifications
        if self.calculate_micro_calcification_number:
            # when it is a positive patch
            if image_level_label[0] == 1:
                # generate the connected component matrix
                connected_components = measure.label(pixel_level_label_np, connectivity=2)
                micro_calcification_number = [connected_components.max()]
            # when it is a negative patch
            else:
                micro_calcification_number = [0]
        else:
            micro_calcification_number = [-1]

        # convert ndarray to tensor
        #
        # image tensor
        image_tensor = torch.FloatTensor(image_np).unsqueeze(dim=0)  # shape: [C, H, W]
        #
        # pixel-level label tensor
        pixel_level_label_tensor = torch.LongTensor(pixel_level_label_np)  # shape: [H, W]
        #
        # dilated pixel-level label tensor
        pixel_level_label_dilated_tensor = torch.LongTensor(pixel_level_label_dilated_np)  # shape: [H, W]
        #
        # image-level label tensor
        image_level_label_tensor = torch.LongTensor(image_level_label)  # shape: [1]
        #
        # micro calcification number label tensor
        micro_calcification_number_label_tensor = torch.LongTensor(micro_calcification_number)  # shape: [1]

        assert len(image_tensor.shape) == 3
        assert len(pixel_level_label_tensor.shape) == 2
        assert pixel_level_label_tensor.shape == pixel_level_label_dilated_tensor.shape
        assert len(image_level_label_tensor.shape) == 1
        assert len(micro_calcification_number_label_tensor.shape) == 1
        assert image_tensor.shape[0] == self.image_channels
        assert image_tensor.shape[1] == pixel_level_label_tensor.shape[0] == self.cropping_size[0]
        assert image_tensor.shape[2] == pixel_level_label_tensor.shape[1] == self.cropping_size[1]

        return image_tensor, pixel_level_label_tensor, pixel_level_label_dilated_tensor, image_level_label_tensor, \
               micro_calcification_number_label_tensor, filename

    def __len__(self):

        return len(self.mixed_patch_filename_list)

    def generate_and_check_filename_list(self, path):
        # check the correctness of the given path
        assert os.path.isdir(path)

        print('-------------------------------------------------------------------------------------------------------')
        print('Starting checking the files in {0}...'.format(path))

        filename_list = os.listdir(path)
        for filename in filename_list:
            # each file's extension must be 'png'
            assert filename.split('.')[-1] == 'png'

        print('Checking passed: all of the involved {} files are legal with extension.'.format(len(filename_list)))
        print('-------------------------------------------------------------------------------------------------------')

        return filename_list
