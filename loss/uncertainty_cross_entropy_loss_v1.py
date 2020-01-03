"""
This file implements a modified version of Cross-Entropy loss.
The uncertainty maps are equipped with the ability of transforming
negative samples -> positive samples.
"""
import torch
import torch.nn as nn


class UncertaintyCrossEntropyLossV1(nn.Module):
    def __init__(self, uncertainty_threshold):
        super(UncertaintyCrossEntropyLossV1, self).__init__()
        assert 0 <= uncertainty_threshold

        self.uncertainty_threshold = uncertainty_threshold
        self.loss_func = nn.CrossEntropyLoss()

        return

    def forward(self, preds, labels, uncertainty_maps, logger=None):
        assert torch.is_tensor(preds)
        assert torch.is_tensor(labels)
        assert 2 <= len(preds.shape) <= 5
        assert len(labels.shape) == len(preds.shape) - 1
        assert len(uncertainty_maps.shape) == 5

        num_dimensions = len(preds.shape) - 2  # 0D, 1D, 2D, or 3D
        num_channels = preds.shape[1]

        # labels must be a tensor on gpu devices
        if labels.device.type != 'cuda':
            labels = labels.cuda()

        # reshape preds into vectors
        if num_dimensions == 1:
            preds = preds.permute(0, 2, 1).contiguous()
        elif num_dimensions == 2:
            preds = preds.permute(0, 2, 3, 1).contiguous()
        elif num_dimensions == 3:
            preds = preds.permute(0, 2, 3, 4, 1).contiguous()

        preds = preds.view(-1, num_channels)

        # reshape pixel_level_labels into vectors
        labels = labels.view(-1)

        # calculate image-level uncertainty based on pixel-level uncertainty
        image_level_uncertainty = uncertainty_maps > 0.05
        image_level_uncertainty = image_level_uncertainty.sum(1).sum(1)
        # ---write here--- #
        # ---write here--- #
        # ---write here--- #
        # ---write here--- #
        # ---write here--- #

        # modify the labels based on uncertainty
        # ---write here--- #
        # ---write here--- #
        # ---write here--- #
        # ---write here--- #
        # ---write here--- #

        assert preds.shape[0] == labels.shape[0]

        loss = self.loss_func(preds, labels)

        return loss

    def get_name(self):
        return 'UncertaintyCrossEntropyLossV1'
