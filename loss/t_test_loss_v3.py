"""
This file implements another version of t-test loss which is derived from the original version of t-test loss
implemented in t_test_loss.py
This version implements what the authors actually did
"""
import numpy as np
import torch
import torch.nn as nn


class TTestLossV3(nn.Module):
    def __init__(self, beta=0.8, lambda_p=1, lambda_n=0.1):
        super(TTestLossV3, self).__init__()

        self.beta = beta
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n

        return

    def forward(self, residues, pixel_level_labels, logger=None):
        assert len(residues.shape) == 4  # shape: B, C, H, W

        if len(pixel_level_labels.shape) == 3:
            pixel_level_labels = pixel_level_labels.unsqueeze(dim=1)
        assert residues.shape == pixel_level_labels.shape

        pixel_level_labels=pixel_level_labels.cuda()

        negative_mask = torch.sub(torch.ones_like(pixel_level_labels), pixel_level_labels).cuda()

        # masked piexls are labeled as 0
        negative_mask = negative_mask.int()

        # masked piexls are labeled as 1
        positive_mask = pixel_level_labels.int()
        shape = pixel_level_labels.shape

        negative_mask = negative_mask.reshape([shape[0], shape[1], 1, -1]).type(torch.ByteTensor)
        positive_mask = positive_mask.reshape([shape[0], shape[1], 1, -1]).type(torch.ByteTensor)
        residues=residues.reshape([shape[0],shape[1],1,-1])

        residues=residues.cuda()
        positive_mask=positive_mask.cuda()
        negative_mask=negative_mask.cuda()

        positive_mask_residues = torch.masked_select(residues, positive_mask)
        negative_mask_residues = torch.masked_select(residues, negative_mask)

        mean_positive_mask_residues = positive_mask_residues.mean()
        mean_negative_mask_residues = negative_mask_residues.mean()

        var_positive_mask_residues = positive_mask_residues.var()
        var_negative_mask_residues = negative_mask_residues.var()

        loss = torch.FloatTensor([0]).cuda()

        if positive_mask_residues.shape[0] > 0:
            loss += torch.max(self.beta - mean_positive_mask_residues, torch.FloatTensor([0]).cuda())

            if positive_mask_residues.shape[0] > 1:
                loss += self.lambda_n * var_positive_mask_residues

        if negative_mask_residues.shape[0] > 0:
            loss += mean_negative_mask_residues

            if negative_mask_residues.shape[0] > 1:
                loss += self.lambda_p * var_negative_mask_residues

        log_message = 'num_b: {}, num_n: {}, m_r_p: {:.4f}, m_r_n: {:.4f}, v_r_p: {:.4f}, v_r_n: {:.4f}, loss: {:.4f}'.format(
            positive_mask_residues.shape[0],
            negative_mask_residues.shape[0],
            mean_positive_mask_residues.item(),
            mean_negative_mask_residues.item(),
            var_positive_mask_residues.item(),
            var_negative_mask_residues.item(),
            loss.item())

        if logger is not None:
            logger.write(log_message)
        else:
            print(log_message)

        return loss


def get_name(self):
    return 'TTestLossV3'
