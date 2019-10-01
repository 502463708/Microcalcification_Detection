import cv2
import numpy as np
import os
import random
import shutil

from skimage import measure


class RadiographLevelDatasetSelect:
    def __init__(self, data_root_dir, connected_component_threshold, output_dir, enable_random=False,
                 training_ratio=3, validation_ratio=1, test_ratio=1):
        assert os.path.isdir(data_root_dir), '{} does not exist'.format(data_root_dir)
        assert connected_component_threshold > 0, 'connected_component_threshold shall be a positive number'
        assert isinstance(enable_random, bool), 'enable_random shall be a bool variable'
        assert isinstance(training_ratio, int) and training_ratio > 0, 'training_ratio shall be a positive integer'
        assert isinstance(validation_ratio,
                          int) and validation_ratio > 0, 'validation_ratio shall be a positive integer'
        assert isinstance(test_ratio, int) and test_ratio > 0, 'test_ratio shall be a positive integer'

        self.data_root_dir = data_root_dir
        self.connected_component_threshold = connected_component_threshold
        self.output_dir = output_dir
        self.enable_random = enable_random
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        self.selected_positive_patch_path_list = list()
        self.selected_negative_patch_path_list = list()
        self.selected_positive_patch_num = 0

        self.training_positive_patch_path_list = list()
        self.validation_positive_patch_path_list = list()
        self.test_positive_patch_path_list = list()
        self.training_negative_patch_path_list = list()
        self.validation_negative_patch_path_list = list()
        self.test_negative_patch_path_list = list()

        # create output tree structure dirs
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        #
        patch_class_list = ['positive_patches', 'negative_patches']
        data_class_list = ['training', 'validation', 'test']
        image_class_list = ['images', 'labels']
        #
        os.mkdir(output_dir)
        for patch_class in patch_class_list:
            curr_path = os.path.join(output_dir, patch_class)
            os.mkdir(curr_path)
            for data_class in data_class_list:
                curr_path = os.path.join(output_dir, patch_class, data_class)
                os.mkdir(curr_path)
                for image_class in image_class_list:
                    curr_path = os.path.join(output_dir, patch_class, data_class, image_class)
                    os.mkdir(curr_path)

        return

    def process_positive_patch_dir(self, positive_patch_dir):

        return

    def process_negative_patch_dir(self, negative_patch_dir):

        return

    def select_patch_path_list(self):

        return

    def copy_images_and_labels(self, image_path_list, dst_dir):

        return

    def run(self):

        return
