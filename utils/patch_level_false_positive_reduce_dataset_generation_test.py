import argparse
import copy
import cv2
import numpy as np
import os
import shutil
import sys
import torch
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.getcwd()))

from common.utils import get_ckpt_path
from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_radiograph_level import MicroCalcificationRadiographLevelDataset
from logger.logger import Logger
from net.vnet2d_v2 import VNet2d
from skimage import measure
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-radiograph-level-roi-extracted-data-split-dataset/',
                        help='The source data dir.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-false-positive-classification-datasets/dataset_debug',
                        help='The destination data dir.')
    parser.add_argument('--reconstruction_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='The reconstruction model saved dir.')
    parser.add_argument('--reconstruction_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--reconstruction_patch_size',
                        type=tuple,
                        default=(112, 112),
                        help='The patch size for reconstruction.')
    parser.add_argument('--patch_stride',
                        type=int,
                        default=56,
                        help='The patch moving stride from one patch to another.')
    parser.add_argument('--prob_threshold',
                        type=float,
                        default=0.1,
                        help='residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=3.14 * 7 * 7 * 0.2,
                        help='Connected components whose area < area_threshold will be discarded.')
    parser.add_argument('--patch_size',
                        type=tuple,
                        default=(56, 56),
                        help='The patch size for saving.')
    args = parser.parse_args()

    return args


def generate_radiograph_level_reconstructed_and_residue_result(images_tensor, reconstruction_net, pixel_level_label_np,
                                                               patch_size, stride):
    if images_tensor.device.type == 'cpu':
        images_tensor = images_tensor.cuda()

    assert stride > 0, 'The variable stride must be a positive integer.'

    height = images_tensor.shape[2]
    width = images_tensor.shape[3]

    reconstructed_radiograph = np.zeros((height, width))
    residue_radiograph = np.zeros((height, width))
    counting_mask = np.zeros((height, width))

    start_row_idx = -1
    end_row_idx = -1
    while end_row_idx < height:
        if start_row_idx == -1:
            start_row_idx = 0
            end_row_idx = start_row_idx + patch_size[0]
        else:
            start_row_idx += stride
            end_row_idx += stride
        if end_row_idx > height:
            gap_row = end_row_idx - height
            end_row_idx -= gap_row
            start_row_idx -= gap_row

        start_column_idx = -1
        end_column_idx = -1
        while end_column_idx < width:
            if start_column_idx == -1:
                start_column_idx = 0
                end_column_idx = start_column_idx + patch_size[1]
            else:
                start_column_idx += stride
                end_column_idx += stride
            if end_column_idx > width:
                gap_column = end_column_idx - width
                end_column_idx -= gap_column
                start_column_idx -= gap_column

            # debug only
            # print(
            #     'row idx range: {} - {} (height = {}), column idx range: {} - {} (width = {})'.format(start_row_idx,
            #                                                                                           end_row_idx,
            #                                                                                           height,
            #                                                                                           start_column_idx,
            #                                                                                           end_column_idx,
            #                                                                                           width))

            # generate the current patch label
            patch_label_np = pixel_level_label_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx]

            # this patch is totally occupied with outlier calcification -> skip
            if np.where(patch_label_np == 2)[0].shape[0] == patch_label_np.size:
                continue
            # this patch is totally occupied with background -> skip
            if np.where(patch_label_np == 3)[0].shape[0] == patch_label_np.size:
                continue
            # this patch is totally occupied with outlier calcification and background -> skip
            if np.where(patch_label_np == 2)[0].shape[0] + np.where(patch_label_np == 3)[0].shape[0] \
                    == patch_label_np.size:
                continue

            # crop this patch for model inference
            patch_image_tensor = images_tensor[:, :, start_row_idx:end_row_idx, start_column_idx:end_column_idx]

            # generate the current reconstructed patch and residue patch
            reconstructed_patch_tensor, residue_patch_tensor = reconstruction_net(patch_image_tensor)
            reconstructed_current_patch = reconstructed_patch_tensor.cpu().detach().numpy().squeeze()
            residue_current_patch = residue_patch_tensor.cpu().detach().numpy().squeeze()

            # get the existing reconstructed patch and residue patch
            reconstructed_existing_patch = reconstructed_radiograph[start_row_idx:end_row_idx,
                                           start_column_idx:end_column_idx]
            residue_existing_patch = residue_radiograph[start_row_idx:end_row_idx,
                                     start_column_idx:end_column_idx]

            # get the counting patch and weight matrix
            counting_mask_patch = counting_mask[start_row_idx:end_row_idx, start_column_idx:end_column_idx]
            current_weight_matrix = 1 / (counting_mask_patch + 1)
            existing_weight_matrix = 1 - current_weight_matrix

            # update the existing reconstructed radiopgraph and residue radiopgraph
            reconstructed_radiograph[start_row_idx:end_row_idx, start_column_idx:end_column_idx] = \
                reconstructed_existing_patch * existing_weight_matrix + \
                reconstructed_current_patch * current_weight_matrix
            residue_radiograph[start_row_idx:end_row_idx, start_column_idx:end_column_idx] = \
                residue_existing_patch * existing_weight_matrix + \
                residue_current_patch * current_weight_matrix

            # update the counting mask
            counting_mask_patch += 1
            counting_mask[start_row_idx:end_row_idx, start_column_idx:end_column_idx] = counting_mask_patch

    assert reconstructed_radiograph.shape == residue_radiograph.shape

    return reconstructed_radiograph, residue_radiograph


def post_process_residue_radiograph(raw_residue_radiograph_np, pixel_level_label_np, prob_threshold, area_threshold,
                                    remove_overlapped_connected_component=True):
    assert raw_residue_radiograph_np.shape == pixel_level_label_np.shape

    # created for labelling connected components
    residue_mask_radiograph_np = np.zeros_like(raw_residue_radiograph_np)

    # created for saving the post processed residue
    processed_residue_radiograph_np = copy.copy(raw_residue_radiograph_np)

    # only pixels with residue value >= prob_threshold can be remained
    processed_residue_radiograph_np[processed_residue_radiograph_np < prob_threshold] = 0

    # generate information for each connected component on processed_residue_radiograph_np
    residue_mask_radiograph_np[processed_residue_radiograph_np > 0] = 1
    connected_components = measure.label(residue_mask_radiograph_np)
    props = measure.regionprops(connected_components)

    # iterate each connected component
    connected_idx = 0
    for prop in props:
        connected_idx += 1
        indexes = connected_components == connected_idx
        # remove the detected results with area smaller than area_threshold
        if prop.area < area_threshold:
            processed_residue_radiograph_np[indexes] = 0
        if remove_overlapped_connected_component:
            # remove the detected results overlapped with background or other lesion
            if pixel_level_label_np[indexes].max() > 1:
                processed_residue_radiograph_np[indexes] = 0

    processed_residue_radiograph_np[pixel_level_label_np == 2] = 0
    processed_residue_radiograph_np[pixel_level_label_np == 3] = 0

    return processed_residue_radiograph_np


def generate_coordinate_list(residue_or_mask_np, mode='detected'):
    # mode must be either 'detected' or 'annotated'
    assert mode in ['detected', 'annotated']

    if mode == 'detected':
        # mode: detected -> iterate each connected component on processed_residue_radiograph_np
        mask_np = copy.copy(residue_or_mask_np)
        mask_np[residue_or_mask_np > 0] = 1
    else:
        # mode: annotated -> iterate each connected component on pixel_level_label_np
        mask_np = copy.copy(residue_or_mask_np)
        # remain micro calcifications and normal tissue label only
        mask_np[mask_np > 1] = 0

    # generate information of each connected component
    connected_components = measure.label(mask_np)
    props = measure.regionprops(connected_components)

    # created for saving the coordinates and the detected score for this connected component
    coordinate_list = list()

    if len(props) > 0:
        for prop in props:
            # record the centroid of this connected component
            coordinate_list.append(np.array(prop.centroid))

    return coordinate_list


def merge_coord_list(pred_coord_list, label_coord_list):
    if len(pred_coord_list) == 0 and len(label_coord_list) == 0:
        coord_list = list()
    elif len(pred_coord_list) == 0:
        coord_list = label_coord_list
    elif len(label_coord_list) == 0:
        coord_list = pred_coord_list
    else:
        coord_list = list()
        merged_coord_list = pred_coord_list + label_coord_list

        for coord_1 in merged_coord_list:
            near_num = 0
            for coord_2 in merged_coord_list:
                if np.linalg.norm(coord_1 - coord_2) < 2:
                    near_num += 1
            if near_num < 2:
                coord_list.append(coord_1)

    return coord_list


def save_images_and_labels(coord_list, image_np, pixel_level_label_np, filename, positive_dataset_type_dir,
                           negative_dataset_type_dir, patch_size):
    height, width = image_np.shape
    # negative_dataset_type_dir = positive_dataset_type_dir.replace('positive', 'negative')

    positive_patch_idx = 0
    negative_patch_idx = 0

    for coord in coord_list:
        # generate legal start and end idx for row and column
        centroid_row_idx = coord[0]
        centroid_column_idx = coord[1]
        #
        centroid_row_idx = np.clip(centroid_row_idx, patch_size[0] / 2, height - patch_size[0] / 2)
        centroid_column_idx = np.clip(centroid_column_idx, patch_size[1] / 2, width - patch_size[1] / 2)
        #
        start_row_idx = int(centroid_row_idx - patch_size[0] / 2)
        end_row_idx = int(centroid_row_idx + patch_size[0] / 2)
        start_column_idx = int(centroid_column_idx - patch_size[1] / 2)
        end_column_idx = int(centroid_column_idx + patch_size[1] / 2)

        # crop this patch from image and label
        image_patch_np = copy.copy(image_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx])
        pixel_level_label_patch_np = copy.copy(
            pixel_level_label_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx])
        image_level_label_patch_bool = True if (pixel_level_label_patch_np == 1).sum() > 0 else False

        # transformed into png format
        image_patch_np *= 255
        #
        pixel_level_label_patch_np[pixel_level_label_patch_np == 1] = 255
        pixel_level_label_patch_np[pixel_level_label_patch_np == 2] = 165
        pixel_level_label_patch_np[pixel_level_label_patch_np == 3] = 85
        #
        image_patch_np = image_patch_np.astype(np.uint8)
        pixel_level_label_patch_np = pixel_level_label_patch_np.astype(np.uint8)
        if image_level_label_patch_bool == False:
            negative_patch_idx += 1
            absolute_image_saving_path = os.path.join(negative_dataset_type_dir,
                                                      'negative_' + filename.split('.')[0] + '_{}.png'.format(
                                                          negative_patch_idx))
            absolute_label_saving_path = absolute_image_saving_path.replace('images', 'labels')
            cv2.imwrite(absolute_image_saving_path, image_patch_np)
            cv2.imwrite(absolute_label_saving_path, pixel_level_label_patch_np)

        elif image_level_label_patch_bool == True:
            positive_patch_idx += 1
            absolute_image_saving_path = os.path.join(positive_dataset_type_dir,
                                                      'positive_' + filename.split('.')[0] + '_{}.png'.format(
                                                          positive_patch_idx))
            absolute_label_saving_path = absolute_image_saving_path.replace('images', 'labels')
            cv2.imwrite(absolute_image_saving_path, image_patch_np)
            cv2.imwrite(absolute_label_saving_path, pixel_level_label_patch_np)

        # print(absolute_label_saving_path)

        # print('negative idx {} '.format(negative_patch_idx))
        # print('positive idx {}'.format(positive_patch_idx))
        # saving
        # cv2.imwrite(absolute_image_saving_path, image_patch_np)
        # cv2.imwrite(absolute_label_saving_path, pixel_level_label_patch_np)

    return


def TestMicroCalcificationRadiographLevelDetection(args):
    # start time of this dataset
    start_time_for_dataset = time()

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    for patch_type in ['positive_patches', 'negative_patches']:
        os.mkdir(os.path.join(args.dst_data_root_dir, patch_type))
        for dataset_type in ['training', 'validation', 'test']:
            os.mkdir(os.path.join(args.dst_data_root_dir, patch_type, dataset_type))
            for image_type in ['images', 'labels']:
                os.mkdir(os.path.join(args.dst_data_root_dir, patch_type, dataset_type, image_type))

    # initialize logger
    logger = Logger(args.dst_data_root_dir)
    logger.write_and_print('Dataset: {}'.format(args.src_data_root_dir))
    logger.write_and_print('Reconstruction model saving dir: {}'.format(args.reconstruction_model_saving_dir))
    logger.write_and_print('Reconstruction ckpt index: {}'.format(args.reconstruction_epoch_idx))

    # define the reconstruction network
    reconstruction_net = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)
    #
    # get the reconstruction absolute ckpt path
    reconstruction_ckpt_path = get_ckpt_path(args.reconstruction_model_saving_dir, args.reconstruction_epoch_idx)
    #
    # load ckpt and transfer net into gpu devices
    reconstruction_net = torch.nn.DataParallel(reconstruction_net).cuda()
    reconstruction_net.load_state_dict(torch.load(reconstruction_ckpt_path))
    reconstruction_net = reconstruction_net.eval()
    #
    logger.write_and_print('Load ckpt: {0}...'.format(reconstruction_ckpt_path))

    for dataset_type in ['training', 'validation', 'test']:
        positive_dataset_type_dir = os.path.join(args.dst_data_root_dir, 'positive_patches', dataset_type, 'images')
        negative_dataset_type_dir = os.path.join(args.dst_data_root_dir, 'negative_patches', dataset_type, 'images')

        # create dataset
        dataset = MicroCalcificationRadiographLevelDataset(data_root_dir=args.src_data_root_dir, mode=dataset_type)

        # create data loader
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_threads)

        for radiograph_idx, (images_tensor, pixel_level_labels_tensor, _, filenames) in enumerate(data_loader):
            filename = filenames[0]

            # logging
            logger.write_and_print('----------------------------------------------------------------------------------')
            logger.write_and_print('Start evaluating {} set radiograph {} out of {}: {}...'.format(dataset_type,
                                                                                                   radiograph_idx + 1,
                                                                                                   dataset.__len__(),
                                                                                                   filename))

            # start time of this radiograph
            start_time_for_radiograph = time()

            # transfer the tensor into gpu device
            images_tensor = images_tensor.cuda()

            # transfer the tensor into ndarray format
            image_np = images_tensor.cpu().numpy().squeeze()
            pixel_level_label_np = pixel_level_labels_tensor.cpu().numpy().squeeze()

            # generated raw radiograph-level residue
            _, raw_residue_radiograph_np = generate_radiograph_level_reconstructed_and_residue_result(images_tensor,
                                                                                                      reconstruction_net,
                                                                                                      pixel_level_label_np,
                                                                                                      args.reconstruction_patch_size,
                                                                                                      args.patch_stride)

            # post-process the raw radiograph-level residue
            processed_residue_radiograph_np = post_process_residue_radiograph(raw_residue_radiograph_np,
                                                                              pixel_level_label_np,
                                                                              args.prob_threshold,
                                                                              args.area_threshold)

            # generate coordinates list for the post-processed radiograph-level residue
            pred_coord_list = generate_coordinate_list(processed_residue_radiograph_np)

            # generate coordinates list for the mask
            label_coord_list = generate_coordinate_list(pixel_level_label_np, mode='annotated')

            # debug only
            logger.write_and_print('pred {}'.format(len(pred_coord_list)))
            logger.write_and_print('label {}'.format(len(label_coord_list)))

            # merge pred_coord_list and label_coord_list
            coord_list = merge_coord_list(pred_coord_list, label_coord_list)

            save_images_and_labels(coord_list, image_np, pixel_level_label_np, filename, positive_dataset_type_dir,
                                   negative_dataset_type_dir,
                                   args.patch_size)

            # logging
            # print logging information of this radiograph
            logger.write_and_print(
                'Finish evaluating radiograph: {}, consuming time: {:.4f}s'.format(radiograph_idx + 1,
                                                                                   time() - start_time_for_radiograph))
            logger.write_and_print(
                '--------------------------------------------------------------------------------------')
            logger.flush()

        # print logging information of this dataset
        logger.write_and_print(
            '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        logger.write_and_print(
            'Finished evaluating this dataset, consuming time: {:.4f}s'.format(time() - start_time_for_dataset))
        logger.write_and_print(
            '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationRadiographLevelDetection(args)
