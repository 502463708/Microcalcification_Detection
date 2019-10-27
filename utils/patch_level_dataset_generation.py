import cv2
import numpy as np
import os


def crop_patches_and_labels(image_path, label_path, patch_size, stride):
    assert os.path.exists(image_path)
    assert os.path.exists(label_path)

    image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    height, width = image_np.shape

    assert len(image_np.shape) == 2
    assert image_np.shape == label_np.shape

    # for saving patches and labels
    image_patch_list = list()
    label_patch_list = list()

    for row_stride_idx in range(height // stride):
        # start and end row index for cropping this patch
        start_row_idx = stride * row_stride_idx
        end_row_idx = start_row_idx + patch_size[0]
        if end_row_idx > height - 1:
            gap = end_row_idx - (height - 1)
            end_row_idx -= gap
            start_row_idx -= gap

        for column_stride_idx in range(width // stride):
            # start and end column index for cropping this patch
            start_column_idx = stride * column_stride_idx
            end_column_idx = start_column_idx + patch_size[1]
            if end_column_idx > width - 1:
                gap = end_column_idx - (width - 1)
                end_column_idx -= gap
                start_column_idx -= gap

            # debug only
            # print(
            #     'row idx range: {} - {} (height = {}), column idx range: {} - {} (width = {})'.format(start_row_idx,
            #                                                                                           end_row_idx,
            #                                                                                           height,
            #                                                                                           start_column_idx,
            #                                                                                           end_column_idx,
            #                                                                                           width))

            # crop this patch and its corresponding label
            image_patch = image_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx]
            label_patch = label_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx]

            # append into the lists
            image_patch_list.append(image_patch)
            label_patch_list.append(label_patch)

    assert len(image_patch_list) == len(label_patch_list)

    return image_patch_list, label_patch_list


def filter_and_save_patches_and_labels(save_dir, dataset_type, image_patch_list, label_patch_list, image_name,
                                       area_threshold=0.5, logger=None):
    pos_patch_count = 0
    neg_patch_count = 0
    other_lesion_patch_count = 0
    background_patch_count = 0

    for idx in range(len(image_patch_list)):
        image_patch = image_patch_list[idx]
        label_patch = label_patch_list[idx]

        assert image_patch.shape == label_patch.shape
        height, width = image_patch.shape

        this_patch_contain_other_lesion = False
        if np.where(label_patch == 165)[0].shape[0] > 0:
            this_patch_contain_other_lesion = True

        this_patch_contain_enough_background = False
        background_area_ratio = np.sum(label_patch == 85) / (height * width)
        if background_area_ratio >= area_threshold:
            this_patch_contain_enough_background = True

        this_is_a_positive_patch = False
        if label_patch.max() == 255:
            this_is_a_positive_patch = True

        if this_patch_contain_other_lesion:
            other_lesion_patch_count += 1
            continue

        if this_patch_contain_enough_background:
            background_patch_count += 1
            continue

        if this_is_a_positive_patch:
            pos_patch_count += 1

            dst_dir = os.path.join(save_dir, 'positive_patches', dataset_type)
            filename = 'positive' + '_' + image_name
            filename = filename.replace('.png', '_{}.png'.format(idx))

            absolute_image_dst_path = os.path.join(dst_dir, 'images', filename)
            absolute_label_dst_path = os.path.join(dst_dir, 'labels', filename)

            cv2.imwrite(absolute_image_dst_path, image_patch)
            cv2.imwrite(absolute_label_dst_path, label_patch)
        else:
            neg_patch_count += 1

            dst_dir = os.path.join(save_dir, 'negative_patches', dataset_type)
            filename = 'negative' + '_' + image_name
            filename = filename.replace('.png', '_{}.png'.format(idx))

            absolute_image_dst_path = os.path.join(dst_dir, 'images', filename)
            absolute_label_dst_path = os.path.join(dst_dir, 'labels', filename)

            cv2.imwrite(absolute_image_dst_path, image_patch)
            cv2.imwrite(absolute_label_dst_path, label_patch)

    if logger is None:
        print(
            'This image contains {} positive patches, {} negative patches, {} other_lesion_pathces, {} background_patches.'.format(
                pos_patch_count, neg_patch_count, other_lesion_patch_count, background_patch_count))
        print('Totally {} patches have been cropped.'.format(
            other_lesion_patch_count + background_patch_count + pos_patch_count + neg_patch_count))
        print('Totally {} patches have been discarded.'.format(other_lesion_patch_count + background_patch_count))
        print('Totally {} patches have been saved.'.format(pos_patch_count + neg_patch_count))
    else:
        logger.write_and_print(
            'This image contains {} positive patches, {} negative patches, {} other_lesion_pathces, {} background_patches.'.format(
                pos_patch_count, neg_patch_count, other_lesion_patch_count, background_patch_count))
        logger.write_and_print('Totally {} patches have been cropped.'.format(
            other_lesion_patch_count + background_patch_count + pos_patch_count + neg_patch_count))
        logger.write_and_print(
            'Totally {} patches have been discarded.'.format(other_lesion_patch_count + background_patch_count))
        logger.write_and_print('Totally {} patches have been saved.'.format(pos_patch_count + neg_patch_count))

    return pos_patch_count, neg_patch_count, other_lesion_patch_count, background_patch_count
