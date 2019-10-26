"""
This file implements a class which can evaluate the accuracy
"""
import cv2
import numpy as np
import torch


class MetricsImageLEvelQuantityRegression(object):

    def __init__(self, image_size):

        """

        :param image_size: used for generating image-level mask

        """

        assert isinstance(image_size, list)

        assert len(image_size) == 2

        self.image_size = image_size

        # record metric on validation set for determining the best model to be saved

        self.determine_saving_metric_on_validation_list = list()

        return

    def metric_batch_level(self, preds, labels):
        """
        evaluate at batch-level
        :param preds: the number of predict calcification
        :param labels: pixel-level label without dilated
        :return: the accuracy of image level quantity regression
        :return: the number of recalled calcification and FP
        """

        assert len(preds.shape) == 2  # shape: B*1
        assert len(labels.shape) == 2  # shape: B*1

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

        visual_preds_list = list()
        visual_labels_list = list()

        for patch_idx in range(preds.shape[0]):
            pred = preds[patch_idx, 0]
            label = labels[patch_idx, 0]
            pred_img, label_img = self.metric_patch_level(pred, label)

            visual_preds_list.append(pred_img)
            visual_labels_list.append(label_img)

        visual_preds_np = np.array(visual_preds_list)  # shape : B,112,112
        visual_labels_np = np.array(visual_labels_list)
        distance_batch_level = np.abs(np.subtract(preds, labels))
        assert preds.shape == labels.shape
        correct_pred = np.sum(np.round(preds) == labels)

        return visual_preds_np, visual_labels_np, distance_batch_level, correct_pred

    def metric_patch_level(self, pred, label):
        # pred and label is a number
        assert len(pred.shape) == 1
        assert len(label.shape) == 1

        # transform into 112*112 images
        pred_image = np.zeros((112, 112), dtype=np.uint8)
        cv2.putText(pred_image, str(pred), (20, 65), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 1)
        # image , input text ,left down coord, font ,font size ,color , thickness

        # again for label
        label_image = np.zeros((112, 112), dtype=np.uint8)
        cv2.putText(label_image, str(pred), (20, 65), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 1)
        # image , input text ,left down coord, font ,font size ,color , thickness

        return pred_image, label_image
