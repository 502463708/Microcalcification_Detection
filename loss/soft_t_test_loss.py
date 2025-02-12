"""
This file implements another version of t-test loss which is derived from the original version of implemented in
t_test_loss_v3.py. This version implements a novel method to pick up soft positive pixels
"""
import numpy as np
import torch
import torch.nn as nn


class SoftTTestLoss(nn.Module):
    def __init__(self, beta=0.8, lambda_p=1, lambda_n=0.1, sp_ratio=0.01):
        super(SoftTTestLoss, self).__init__()

        self.beta = beta
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n
        self.sp_ratio = sp_ratio

        return

    def forward(self, residues, pixel_level_labels, logger=None):
        assert torch.is_tensor(residues)
        assert torch.is_tensor(pixel_level_labels)
        assert len(residues.shape) == 4  # shape: B, C, H, W

        # add the missing dimension of channel in pixel_level_labels
        if len(pixel_level_labels.shape) == 3:
            pixel_level_labels = pixel_level_labels.unsqueeze(dim=1)
        assert residues.shape == pixel_level_labels.shape

        # pixel_level_labels must be a tensor on gpu devices
        if pixel_level_labels.device.type != 'cuda':
            pixel_level_labels = pixel_level_labels.cuda()

        # reshaped into vectors
        residues = residues.view(-1)
        pixel_level_labels = pixel_level_labels.view(-1)

        # generate soft positive pixel indexes
        residues_np = residues.cpu().detach().numpy()
        K = int(self.sp_ratio * len(residues_np))
        residue_threshold = residues_np[np.argpartition(residues_np, -K)[-K:]].min()
        soft_positive_pixel_idx = (residues >= residue_threshold).bool()

        # bool variable for the following torch.masked_select() operation
        true_positive_pixel_idx = pixel_level_labels.bool()
        positive_pixel_idx = true_positive_pixel_idx | soft_positive_pixel_idx
        negative_pixel_idx = ~positive_pixel_idx

        # split residues into positive and negative one
        positive_residue_pixels = torch.masked_select(residues, positive_pixel_idx)
        negative_residue_pixels = torch.masked_select(residues, negative_pixel_idx)

        loss = torch.FloatTensor([0]).cuda()

        if positive_residue_pixels.shape[0] > 0:
            mean_residue_pixels_positive = positive_residue_pixels.mean()
            loss += torch.max(self.beta - mean_residue_pixels_positive, torch.FloatTensor([0]).cuda())
            # calculate variance only when the number of the positive pixels > 1
            if positive_residue_pixels.shape[0] > 1:
                var_residue_pixels_positive = positive_residue_pixels.var()
                loss += self.lambda_n * var_residue_pixels_positive

        if negative_residue_pixels.shape[0] > 0:
            mean_residue_pixels_negative = negative_residue_pixels.mean()
            loss += mean_residue_pixels_negative
            # calculate variance only when the number of the negative pixels > 1
            if negative_residue_pixels.shape[0] > 1:
                var_residue_pixels_negative = negative_residue_pixels.var()
                loss += self.lambda_p * var_residue_pixels_negative

        log_message = 'num_p: {}, num_n: {}, m_r_p: {:.4f}, m_r_n: {:.4f}, v_r_p: {:.4f}, v_r_n: {:.4f}, loss: {:.4f}'.format(
            positive_residue_pixels.shape[0],
            negative_residue_pixels.shape[0],
            mean_residue_pixels_positive.item() if positive_residue_pixels.shape[0] > 0 else -1,
            mean_residue_pixels_negative.item() if negative_residue_pixels.shape[0] > 0 else -1,
            var_residue_pixels_positive.item() if positive_residue_pixels.shape[0] > 1 else -1,
            var_residue_pixels_negative.item() if negative_residue_pixels.shape[0] > 1 else -1,
            loss.item())

        if logger is not None:
            logger.write_and_print(log_message)
        else:
            print(log_message)

        return loss

    def get_name(self):
        return 'SoftTTestLoss'
