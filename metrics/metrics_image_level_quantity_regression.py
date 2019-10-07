"""
This file implements a class which can evaluate the accuracy
"""
import numpy as np
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from common.utils import post_process_residue
from skimage import measure


class MetricsImageLEvelQuantityRegression(object):
    def __int__(self, dilated_raduis):
        assert dilated_raduis >= 0
        self.dilated_raduis = dilated_raduis

        return

    def metric_batch_level(self, preds, labels):
        """
        evaluate at batch-level
        :param preds: the number of predict calcification
        :param labels: pixel-level label without dilated
        :return: the accuracy of image level quantity regression
        :return: the number of recalled calcification and FP
        """

        assert len(preds.shape) == 2  # shape: B, Num
        assert len(labels.shape) == 3  # shape: B, H, W

        # transfer the tensor into cpu device
        if torch.is_tensor(preds):
            if preds.device.type != 'cpu':
                preds = preds.cpu().detach()
            # transform the tensor into ndarray format
            preds = preds.numpy()

        # transfer the tensor into cpu device
        if torch.is_tensor(labels):
            if labels.device.type != 'cpu':
                labels = labels.cpu()
            # transform the tensor into ndarray format
            labels = labels.numpy()

        post_process_preds_list = list()

        for patch_idx in range(preds.shape[0]):
            pred = preds[patch_idx, :, :]
            label = labels[patch_idx, :, :]

            post_process_pred = self.metric_patch_level(pred, label)

            post_process_preds_list.append(post_process_pred)

        post_process_preds_np = np.array(post_process_preds_list)  # shape: B,Num

        assert post_process_preds_np.shape == preds.shape

        return post_process_preds_np

    def metric_patch_level(self, pred, label):
        assert len(pred.shape) == 2

        # post-process residue
        post_process_pred = post_process_residue(pred, self.prob_threshold, self.area_threshold)

        # extract connected components
        post_process_pred_connected_components = measure.label(post_process_pred, connectivity=2)
        label_connected_components = measure.label(label, connectivity=2)

        # analyze properties of each connected component
        post_process_pred_props = measure.regionprops(post_process_pred_connected_components)
        label_props = measure.regionprops(label_connected_components)

        # detected
        pred_num = len(post_process_pred_props)
        label_num = len(label_props)

        # transform into 112*112 images
        image = np.zeros((112, 112), dtype=np.uint8)
        image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype("arial.ttf", 60)

        draw.text((40, 20), str(pred_num), (255), font=font)

        pred_img = np.array(image)

        # again for label
        label_image = np.zeros((112, 112), dtype=np.uint8)
        label_image = Image.fromarray(label_image)

        draw = ImageDraw.Draw(label_image)
        font = ImageFont.truetype("arial.ttf", 60)

        draw.text((40, 20), str(label_num), (255), font=font)
        label_img = np.array(label_image)

        return pred_num, label_num, pred_img, label_img
