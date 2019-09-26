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

    def forward(self, preds, pixel_level_labels, logger=None):
        assert torch.is_tensor(preds)
        assert torch.is_tensor(pixel_level_labels)
        assert len(preds.shape) == 4  # shape: B, C, H, W
        assert len(pixel_level_labels.shape) == 3  # shape: B, H, W

        # pixel_level_labels must be a tensor on gpu devices
        if pixel_level_labels.device.type != 'cuda':
            pixel_level_labels = pixel_level_labels.cuda()

        # reshape preds into vectors
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, 2)

        # reshape pixel_level_labels into vectors
        pixel_level_labels = pixel_level_labels.view(-1)

        assert preds.shape[0] == pixel_level_labels.shape[0]

        loss = self.loss_func(preds, pixel_level_labels)

        return loss

    def get_name(self):
        return 'CrossEntropyLoss'
