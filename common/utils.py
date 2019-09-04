import cv2
import numpy as np

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
