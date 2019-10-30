import argparse
import cv2
import numpy as np
import os
import shutil
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn

from common.utils import get_ckpt_path
from config.config_micro_calcification_patch_level_classification import cfg as c_cfg
from config.config_micro_calcification_patch_level_reconstruction import cfg as r_cfg
from dataset.dataset_micro_calcification_radiograph_level import MicroCalcificationRadiographLevelDataset
from logger.logger import Logger
from metrics.metrics_radiograph_level_detection import MetricsRadiographLevelDetection
from net.resnet18 import ResNet18
from net.vnet2d_v2 import VNet2d
from skimage import measure
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-radiograph-level-roi-extracted-data-split-dataset/',
                        help='The source data dir.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset to be evaluated (training, validation, test).')
    parser.add_argument('--prediction_saving_dir',
                        type=str,
                        default='/data/lars/results/micro_calcification_radiograph_level_detection_results/',
                        help='The predicted results saving dir.')
    parser.add_argument('--reconstruction_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20190925_uCs_reconstruction_ttestlossv3_default_dilation_radius_7',
                        help='The reconstruction model saved dir.')
    parser.add_argument('--reconstruction_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--classification_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20190926_uCs_image_level_classification_CE_default/',
                        help='The classification model saved dir.')
    parser.add_argument('--classification_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--patch_size',
                        type=tuple,
                        default=(112, 112),
                        help='The height and width of patch.')
    parser.add_argument('--patch_stride',
                        type=int,
                        default=56,
                        help='The patch moving stride from one patch to another.')
    parser.add_argument('--prob_threshold',
                        type=float,
                        default=0.2,
                        help='residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=3.14 * 7 * 7 / 2,
                        help='Connected components whose area < area_threshold will be discarded.')
    parser.add_argument('--score_threshold_stride',
                        type=float,
                        default=0.1,
                        help='The score threshold stride for calculating recalls and FPs.')
    parser.add_argument('--distance_threshold',
                        type=int,
                        default=14,
                        help='Candidates whose distance between calcification < distance_threshold is a recalled one.')

    args = parser.parse_args()

    return args


def generate_radiograph_level_reconstructed_and_residue_result(images_tensor, reconstruction_net,
                                                               pixel_level_labels_tensor,
                                                               patch_size, stride):
    if images_tensor.device() == 'cpu':
        images_tensor = images_tensor.cuda()

    assert stride > 0, 'The variable stride must be a positive integer.'

    height = images_tensor.shape[2]
    width = images_tensor.shape[3]

    reconstructed_radiograph = np.zeros((height, width))
    residue_radiograph = np.zeros((height, width))
    counting_mask = np.zeros((height, width))

    mask_radiograph = pixel_level_labels_tensor.cpu().numpy().suqeeze()

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

            # generate the current patch label
            patch_label_np = mask_radiograph[start_row_idx:end_row_idx, start_column_idx:end_column_idx]

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

    reconstructed_radiograph[mask_radiograph == 2] = 0
    residue_radiograph[mask_radiograph == 3] = 0

    assert reconstructed_radiograph.shape == residue_radiograph.shape

    return reconstructed_radiograph, residue_radiograph


def generate_micro_calcification_score_list(images_tensor, classification_net, residue_radiograph_np, patch_size,
                                            prob_threshold, area_threshold):
    height, width = residue_radiograph_np.shape

    masked_residue_np = np.zeros_like(residue_radiograph_np)
    masked_residue_np[residue_radiograph_np > prob_threshold] = 1

    connected_components = measure.label(masked_residue_np)
    props = measure.regionprops(connected_components, coordinates='rc')

    micro_calcification_coordinate_list = list()
    micro_calcification_score_list = list()

    for prop in props:
        if prop.area < area_threshold:
            residue_radiograph_np[prop._label_image == 1] = 0
            continue
        else:
            micro_calcification_coordinate_list.append(prop.centroid)

            centroid_row_idx = prop.centroid[0]
            centroid_column_idx = prop.centroid[1]

            centroid_row_idx = np.clip(centroid_row_idx, patch_size[0], height - patch_size[0])
            centroid_column_idx = np.clip(centroid_column_idx, patch_size[1], width - patch_size[1])

            start_row_idx = centroid_row_idx - patch_size[0]
            end_row_idx = centroid_row_idx + patch_size[0]
            start_column_idx = centroid_column_idx - patch_size[1]
            end_column_idx = centroid_column_idx + patch_size[1]

            # crop this patch for model inference
            patch_image_tensor = images_tensor[:, :, start_row_idx:end_row_idx, start_column_idx:end_column_idx]

            # generate the positive class prediction probability
            classification_preds_tensor = classification_net(patch_image_tensor)
            positive_prob = classification_preds_tensor.cpu().detach().numpy().squeeze()[1]

            # generate the mean value of this connected component on the residue
            residue_mean = (residue_radiograph_np[prop._label_image == 1]).mean()

            micro_calcification_score_list.append(positive_prob * residue_mean)

    return micro_calcification_coordinate_list, micro_calcification_score_list, residue_radiograph_np


def label_2_coord_list(pixel_level_labels_tensor):
    if pixel_level_labels_tensor.device() != 'cpu':
        pixel_level_labels_tensor = pixel_level_labels_tensor.cpu()

    # remain micro calcifications and normal tissue label only
    pixel_level_label_np = pixel_level_labels_tensor.numpy().squeeze()
    pixel_level_label_np[pixel_level_label_np > 1] = 0

    label_connected_components = measure.label(pixel_level_label_np, connectivity=2)
    label_props = measure.regionprops(label_connected_components)

    label_coord_list = list()
    label_num = len(label_props)
    if label_num > 0:
        for idx in range(label_num):
            label_coord_list.append(np.array(label_props[idx].centroid))

    return label_coord_list


def save_tensor_in_png_and_nii_format(images_tensor, pixel_level_labels_tensor, residue_radiograph_np, filenames,
                                      prediction_saving_dir):
    # convert tensor into numpy and squeeze channel dimension
    image_np = images_tensor.cpu().detach().numpy().squeeze()
    pixel_level_label_np = pixel_level_labels_tensor.numpy().squeeze()

    assert image_np.shape == pixel_level_label_np.shape == residue_radiograph_np.shape

    filename = filenames[0]

    stacked_np = np.concatenate((np.expand_dims(image_np, axis=0),
                                 np.expand_dims(pixel_level_label_np, axis=0),
                                 np.expand_dims(residue_radiograph_np, axis=0)), axis=0)

    stacked_image = sitk.GetImageFromArray(stacked_np)
    sitk.WriteImage(stacked_image, os.path.join(prediction_saving_dir, filename.replace('png', 'nii')))

    image_np *= 255
    pixel_level_label_np *= 255
    residue_radiograph_np *= 255

    image_np = image_np.astype(np.uint8)
    pixel_level_label_np = pixel_level_label_np.astype(np.uint8)
    residue_radiograph_np = residue_radiograph_np.astype(np.uint8)

    cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_image.png')),
                image_np)
    cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pixel_level_label.png')),
                pixel_level_label_np)
    cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_post_processed_residue.png')),
                residue_radiograph_np)

    return


def TestMicroCalcificationRadiographLevelDetection(args):
    # start time of this dataset
    start_time_for_dataset = time()

    visualization_saving_dir = os.path.join(args.prediction_saving_dir, 'qualitative_results')

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(args.prediction_saving_dir):
        shutil.rmtree(args.prediction_saving_dir)
    os.mkdir(args.prediction_saving_dir)
    os.mkdir(visualization_saving_dir)

    # initialize logger
    logger = Logger(args.prediction_saving_dir, 'quantitative_results.txt')
    logger.write_and_print('Dataset: {}'.format(args.data_root_dir))
    logger.write_and_print('Dataset type: {}'.format(args.dataset_type))
    logger.write_and_print('Reconstruction model saving dir: {}'.format(args.reconstruction_model_saving_dir))
    logger.write_and_print('Reconstruction ckpt index: {}'.format(args.reconstruction_epoch_idx))
    logger.write_and_print('Classification model saving dir: {}'.format(args.classification_model_saving_dir))
    logger.write_and_print('Classification ckpt index: {}'.format(args.classification_epoch_idx))

    # define the reconstruction network
    reconstruction_net = VNet2d(num_in_channels=r_cfg.net.in_channels, num_out_channels=r_cfg.net.out_channels)
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

    # define the classification network
    classification_net = ResNet18(in_channels=c_cfg.net.in_channels, num_classes=c_cfg.net.num_classes)
    #
    # get the classification absolute ckpt path
    classification_ckpt_path = get_ckpt_path(args.classification_model_saving_dir, args.classification_epoch_idx)
    #
    # load ckpt and transfer net into gpu devices
    classification_net = torch.nn.DataParallel(classification_net).cuda()
    classification_net.load_state_dict(torch.load(classification_ckpt_path))
    classification_net = classification_net.eval()
    #
    logger.write_and_print('Load ckpt: {0}...'.format(classification_ckpt_path))

    # create dataset
    dataset = MicroCalcificationRadiographLevelDataset(data_root_dir=args.data_root_dir,
                                                       mode=args.dataset_type)
    # create data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=r_cfg.train.num_threads)

    # set up metrics object
    metrics = MetricsRadiographLevelDetection(args.distance_threshold, args.score_threshold_stride)

    for radiograph_idx, (
            images_tensor, pixel_level_labels_tensor, radiograph_level_labels_tensor, filenames) in enumerate(
        data_loader):
        # start time of this radiograph
        start_time_for_radiograph = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        _, residue_radiograph_np = generate_radiograph_level_reconstructed_and_residue_result(images_tensor,
                                                                                              reconstruction_net,
                                                                                              pixel_level_labels_tensor,
                                                                                              args.patch_size,
                                                                                              args.patch_stride)

        pred_coord_list, pred_score_list, residue_radiograph_np = generate_micro_calcification_score_list(images_tensor,
                                                                                                          classification_net,
                                                                                                          residue_radiograph_np,
                                                                                                          args.patch_size,
                                                                                                          args.prob_threshold,
                                                                                                          args.area_threshold)

        label_coord_list = label_2_coord_list(pixel_level_labels_tensor)

        detection_result_record_radiograph_level = metrics.metric_all_score_thresholds(pred_coord_list, pred_score_list,
                                                                                       label_coord_list)

        # print logging information of this radiograph
        logger.write_and_print('--------------------------------------------------------------------------------------')
        logger.write_and_print(
            'Radiograph: {}, consuming time: {:.4f}s'.format(radiograph_idx, time() - start_time_for_radiograph))
        detection_result_record_radiograph_level.print(logger)
        logger.write_and_print('--------------------------------------------------------------------------------------')

        save_tensor_in_png_and_nii_format(images_tensor, pixel_level_labels_tensor, residue_radiograph_np, filenames,
                                          visualization_saving_dir)

        logger.flush()

    # print logging information of this dataset
    logger.write_and_print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    logger.write_and_print(
        'Finished evaluating this dataset, consuming time: {:.4f}s'.format(time() - start_time_for_dataset))
    metrics.detection_result_record_dataset_level.print(logger)
    logger.write_and_print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationRadiographLevelDetection(args)
