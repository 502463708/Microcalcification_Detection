import cv2
import numpy as np
import os
import torch

from skimage import measure


def dilate_image_level_label(image_level_label, dilation_radius):
    """
    This function implements the dilation for mask
    :param image_level_label:
    :param dilation_radius:
    :return:
    """
    assert dilation_radius > 0

    dilation_diameter = 2 * dilation_radius + 1
    kernel = np.zeros((dilation_diameter, dilation_diameter), np.uint8)

    for row_idx in range(dilation_diameter):
        for column_idx in range(dilation_diameter):
            if np.linalg.norm(np.array([row_idx, column_idx]) - np.array(
                    [dilation_radius, dilation_radius])) <= dilation_radius:
                kernel[row_idx, column_idx] = 1

    dilated_image_level_label = cv2.dilate(image_level_label, kernel, iterations=1)

    assert dilated_image_level_label.shape == image_level_label.shape

    return dilated_image_level_label


def post_process_residue(image_np, prob_threshold, area_threshold):
    """
    This function implements the post-process for the residue, including
    the following 2 steps:
        1. transform the residue into binary mask based on prob_threshold
        2. discard connected components whose area < area_threshold
    :param image_np: residue
    :param prob_threshold:
    :param area_threshold:
    :return:
    """
    assert len(image_np.shape) == 2

    image_np[image_np <= prob_threshold] = 0
    image_np[image_np > prob_threshold] = 1

    connected_components = measure.label(image_np, connectivity=2)

    props = measure.regionprops(connected_components)

    connected_component_num = len(props)

    if connected_component_num > 0:
        for connected_component_idx in range(connected_component_num):
            if props[connected_component_idx].area < area_threshold:
                connected_components[connected_components == connected_component_idx + 1] = 0

    post_processed_image_np = np.zeros_like(image_np)
    post_processed_image_np[connected_components != 0] = 1

    return post_processed_image_np


def get_ckpt_path(model_saving_dir, epoch_idx=-1):
    """
    Given a dir (where the model is saved) and an index (which ckpt is specified),
    This function returns the absolute ckpt path
    :param model_saving_dir:
    :param epoch_idx:
        default mode: epoch_idx = -1 -> return the best ckpt
        specified mode: epoch_idx >= 0 -> return the specified ckpt
    :return: absolute ckpt path
    """
    assert os.path.exists(model_saving_dir)
    assert epoch_idx >= -1

    ckpt_dir = os.path.join(model_saving_dir, 'ckpt')
    # specified mode: epoch_idx is specified -> load the specified ckpt
    if epoch_idx >= 0:
        ckpt_path = os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(epoch_idx))
    # default mode: epoch_idx is not specified -> load the best ckpt
    else:
        saved_ckpt_list = os.listdir(ckpt_dir)
        best_ckpt_filename = [best_ckpt_filename for best_ckpt_filename in saved_ckpt_list if
                              'net_best_on_validation_set' in best_ckpt_filename][0]
        ckpt_path = os.path.join(ckpt_dir, best_ckpt_filename)

    return ckpt_path


def save_best_ckpt(metrics, net, ckpt_dir, epoch_idx):
    """
    This function can discriminatively save this ckpt in case that it is the currently best ckpt on validation set
    :param metrics:
    :param net:
    :param ckpt_dir:
    :param epoch_idx:
    :return:
    """

    is_best_ckpt = metrics.determine_saving_metric_on_validation_list[-1] == max(
        metrics.determine_saving_metric_on_validation_list)
    if is_best_ckpt:
        # firstly remove the last saved best ckpt
        saved_ckpt_list = os.listdir(ckpt_dir)
        for saved_ckpt_filename in saved_ckpt_list:
            if 'net_best_on_validation_set' in saved_ckpt_filename:
                os.remove(os.path.join(ckpt_dir, saved_ckpt_filename))

        # save the current best ckpt
        torch.save(net.state_dict(),
                   os.path.join(ckpt_dir, 'net_best_on_validation_set_epoch_{}.pth'.format(epoch_idx)))

    return


def extract_classification_preds_channel(preds, channel_idx, use_softmax=True, keep_dim=True):
    """

    :param preds:
    :param channel_idx:
    :param use_softmax:
    :param keep_dim:
    :return:
    """
    assert torch.is_tensor(preds)
    assert len(preds.shape) == 4
    assert channel_idx >= 0

    if use_softmax:
        preds = torch.softmax(preds, dim=1)

    extracted_preds = preds[:, channel_idx, :, :]

    if keep_dim:
        extracted_preds = extracted_preds.unsqueeze(dim=1)
        assert len(preds.shape) == len(extracted_preds.shape)

    return extracted_preds
