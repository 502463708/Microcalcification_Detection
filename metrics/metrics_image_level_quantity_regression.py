"""
This file implements a class which can evaluate the accuracy
"""
import numpy as np
import torch

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
        pred_list = list()
        pred_num = len(post_process_pred_props)
        if pred_num > 0:
            for idx in range(pred_num):
                pred_list.append(np.array(post_process_pred_props[idx].centroid))

        # annotated
        label_list = list()
        label_num = len(label_props)
        if label_num > 0:
            for idx in range(label_num):
                label_list.append(np.array(label_props[idx].centroid))

        calcification_num = label_num
        recall_num = 0
        FP_num = 0

        # for the negative patch case
        if label_num == 0:
            FP_num = pred_num

        # for the positive patch case with failing to detect anything
        elif pred_num == 0:
            recall_num = 0

        # for the positive patch case with something being detected
        else:
            # calculate recall
            for label_idx in range(label_num):
                for pred_idx in range(pred_num):
                    if np.linalg.norm(label_list[label_idx] - pred_list[pred_idx]) <= self.distance_threshold:
                        recall_num += 1
                        break

            # calculate FP
            for pred_idx in range(pred_num):
                for label_idx in range(label_num):
                    if np.linalg.norm(label_list[label_idx] - pred_list[pred_idx]) <= self.distance_threshold:
                        break
                    if label_idx == label_num - 1:
                        FP_num += 1

        return post_process_pred, calcification_num, recall_num, FP_num
