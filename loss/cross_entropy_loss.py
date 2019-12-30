"""
This file implements
"""
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

        return

    def forward(self, preds, labels, logger=None):
        assert torch.is_tensor(preds)
        assert torch.is_tensor(labels)
        assert 2 <= len(preds.shape) <= 5
        assert len(labels.shape) == len(preds.shape) - 1

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

        assert preds.shape[0] == labels.shape[0]

        loss = self.loss_func(preds, labels)

        return loss

    def get_name(self):
        return 'CrossEntropyLoss'
