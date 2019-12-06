import argparse
import copy
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

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-radiograph-level-roi-extracted-data-split-dataset/',
                        help='The source data dir.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset to be evaluated (training, validation, test).')
    parser.add_argument('--prediction_saving_dir',
                        type=str,
                        default='/data/lars/results/20191109_5764-uCs-micro_calcification_radiograph_level_detection_results_rec_dilatted_7_cls_pos_2_neg_0.5_areath_0.2_probth_0.1/',
                        help='The predicted results saving dir.')
    parser.add_argument('--reconstruction_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='The reconstruction model saved dir.')
    parser.add_argument('--reconstruction_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--classification_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_pos2neg_0.5_classification_CE_default/',
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
                        default=0.1,
                        help='residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=3.14 * 7 * 7 * 0.2,
                        help='Connected components whose area < area_threshold will be discarded.')
    parser.add_argument('--score_threshold_stride',
                        type=float,
                        default=0.05,
                        help='The score threshold stride for calculating recalls and FPs.')
    parser.add_argument('--distance_threshold',
                        type=int,
                        default=14,
                        help='Candidates whose distance between calcification < distance_threshold is a recalled one.')
    parser.add_argument('--slack_for_recall',
                        type=bool,
                        default=True,
                        help='The bool variable for slacking recall metric standard.')

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


def generate_coordinate_and_score_list(images_tensor, classification_net, pixel_level_label_np,
                                       raw_residue_radiograph_np, processed_residue_radiograph_np, filename, saving_dir,
                                       patch_size, mode='detected'):
    # mode must be either 'detected' or 'annotated'
    assert mode in ['detected', 'annotated']

    # make the related dirs
    patch_level_root_saving_dir = os.path.join(saving_dir, filename[:-4])
    patch_visualization_dir = os.path.join(patch_level_root_saving_dir, mode)
    if not os.path.exists(patch_level_root_saving_dir):
        os.mkdir(patch_level_root_saving_dir)
    os.mkdir(patch_visualization_dir)

    height, width = processed_residue_radiograph_np.shape

    if mode == 'detected':
        # mode: detected -> iterate each connected component on processed_residue_radiograph_np
        mask_np = copy.copy(processed_residue_radiograph_np)
        mask_np[processed_residue_radiograph_np > 0] = 1
    else:
        # mode: annotated -> iterate each connected component on pixel_level_label_np
        mask_np = copy.copy(pixel_level_label_np)
        # remain micro calcifications and normal tissue label only
        mask_np[mask_np > 1] = 0

    # generate information of each connected component
    connected_components = measure.label(mask_np)
    props = measure.regionprops(connected_components)

    # created for saving the coordinates and the detected score for this connected component
    coordinate_list = list()
    score_list = list()

    connected_idx = 0
    if len(props) > 0:
        for prop in props:
            connected_idx += 1

            # generate logical indexes for this connected component
            indexes = connected_components == connected_idx

            # record the centroid of this connected component
            coordinate_list.append(np.array(prop.centroid))

            # generate legal start and end idx for row and column
            centroid_row_idx = prop.centroid[0]
            centroid_column_idx = prop.centroid[1]
            #
            centroid_row_idx = np.clip(
                centroid_row_idx, patch_size[0] / 2, height - patch_size[0] / 2)
            centroid_column_idx = np.clip(
                centroid_column_idx, patch_size[1] / 2, width - patch_size[1] / 2)
            #
            start_row_idx = int(centroid_row_idx - patch_size[0] / 2)
            end_row_idx = int(centroid_row_idx + patch_size[0] / 2)
            start_column_idx = int(centroid_column_idx - patch_size[1] / 2)
            end_column_idx = int(centroid_column_idx + patch_size[1] / 2)

            # crop this patch for model inference
            patch_image_tensor = images_tensor[:, :, start_row_idx:end_row_idx, start_column_idx:end_column_idx]

            # generate the positive class prediction probability
            classification_preds_tensor = classification_net(patch_image_tensor)
            classification_preds_tensor = torch.softmax(classification_preds_tensor, dim=1)
            positive_prob = classification_preds_tensor.cpu().detach().numpy().squeeze()[1]

            # calculate the mean value of this connected component on the residue
            residue_mean = (processed_residue_radiograph_np[indexes]).mean()

            # calculate and record the score of this connected component
            score = positive_prob * residue_mean
            score_list.append(score)

            # process the visualization results
            image_patch_np = copy.copy(patch_image_tensor.cpu().detach().numpy().squeeze())
            #
            pixel_level_label_patch_np = copy.copy(
                pixel_level_label_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx])
            #
            raw_residue_patch_np = copy.copy(
                raw_residue_radiograph_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx])
            #
            processed_residue_patch_np = copy.copy(
                processed_residue_radiograph_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx])
            #
            stacked_np = np.concatenate((np.expand_dims(image_patch_np, axis=0),
                                         np.expand_dims(
                                             pixel_level_label_patch_np, axis=0),
                                         np.expand_dims(
                                             raw_residue_patch_np, axis=0),
                                         np.expand_dims(
                                             processed_residue_patch_np, axis=0)), axis=0)
            stacked_image = sitk.GetImageFromArray(stacked_np)
            #
            image_patch_np *= 255
            raw_residue_patch_np *= 255
            processed_residue_patch_np *= 255
            #
            pixel_level_label_patch_np[pixel_level_label_patch_np == 1] = 255
            pixel_level_label_patch_np[pixel_level_label_patch_np == 2] = 165
            pixel_level_label_patch_np[pixel_level_label_patch_np == 3] = 85
            #
            image_patch_np = image_patch_np.astype(np.uint8)
            raw_residue_patch_np = raw_residue_patch_np.astype(np.uint8)
            processed_residue_patch_np = processed_residue_patch_np.astype(np.uint8)
            pixel_level_label_patch_np = pixel_level_label_patch_np.astype(np.uint8)
            #
            prob_saving_image = np.zeros((patch_size[0], patch_size[1], 3), np.uint8)
            mean_residue_saving_image = np.zeros((patch_size[0], patch_size[1], 3), np.uint8)
            score_saving_image = np.zeros((patch_size[0], patch_size[1], 3), np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(prob_saving_image, '{:.4f}'.format(positive_prob), (0, 64), font, 1, (0, 255, 255), 2)
            cv2.putText(mean_residue_saving_image, '{:.4f}'.format(residue_mean), (0, 64), font, 1, (255, 0, 255), 2)
            cv2.putText(score_saving_image, '{:.4f}'.format(positive_prob * residue_mean), (0, 64), font, 1,
                        (255, 255, 0), 2)

            # saving
            cv2.imwrite(os.path.join(patch_visualization_dir,
                                     filename.replace('.png', '_patch_{:0>3d}_image.png'.format(connected_idx))),
                        image_patch_np)
            cv2.imwrite(os.path.join(patch_visualization_dir,
                                     filename.replace('.png', '_patch_{:0>3d}_mask.png'.format(connected_idx))),
                        pixel_level_label_patch_np)
            cv2.imwrite(os.path.join(patch_visualization_dir,
                                     filename.replace('.png', '_patch_{:0>3d}_raw_residue.png'.format(connected_idx))),
                        raw_residue_patch_np)
            cv2.imwrite(os.path.join(patch_visualization_dir,
                                     filename.replace('.png',
                                                      '_patch_{:0>3d}_processed_residue.png'.format(connected_idx))),
                        processed_residue_patch_np)
            cv2.imwrite(os.path.join(patch_visualization_dir,
                                     filename.replace('.png',
                                                      '_patch_{:0>3d}_positive_prob.png'.format(connected_idx))),
                        prob_saving_image)
            cv2.imwrite(os.path.join(patch_visualization_dir,
                                     filename.replace('.png', '_patch_{:0>3d}_mean_residue.png'.format(connected_idx))),
                        mean_residue_saving_image)
            cv2.imwrite(os.path.join(patch_visualization_dir,
                                     filename.replace('.png', '_patch_{:0>3d}_score.png'.format(connected_idx))),
                        score_saving_image)
            sitk.WriteImage(stacked_image,
                            os.path.join(patch_visualization_dir,
                                         filename.replace('.png', '_patch_{:0>3d}.nii'.format(connected_idx))))

    return coordinate_list, score_list


def save_radiograph_level_results(images_tensor, pixel_level_label_np, raw_residue_radiograph_np,
                                  processed_residue_radiograph_np, filename, saving_dir):
    # convert tensor into numpy and squeeze channel dimension
    image_np = images_tensor.cpu().detach().numpy().squeeze()

    assert image_np.shape == pixel_level_label_np.shape == raw_residue_radiograph_np.shape == \
           processed_residue_radiograph_np.shape

    # process the visualization results
    stacked_np = np.concatenate((np.expand_dims(image_np, axis=0),
                                 np.expand_dims(pixel_level_label_np, axis=0),
                                 np.expand_dims(processed_residue_radiograph_np, axis=0),
                                 np.expand_dims(raw_residue_radiograph_np, axis=0)), axis=0)
    stacked_image = sitk.GetImageFromArray(stacked_np)
    #
    image_np *= 255
    raw_residue_radiograph_np *= 255
    processed_residue_radiograph_np *= 255
    #
    # process pixel-level label                            # normal tissue: 0 (.png) <- 0 (tensor)
    # micro calcification: 255 (.png) <- 1 (tensor)
    pixel_level_label_np[pixel_level_label_np == 1] = 255
    # other lesion: 165 (.png) <- 2 (tensor)
    pixel_level_label_np[pixel_level_label_np == 2] = 165
    # background: 85 (.png) <- 3 (tensor)
    pixel_level_label_np[pixel_level_label_np == 3] = 85
    #
    image_np = image_np.astype(np.uint8)
    pixel_level_label_np = pixel_level_label_np.astype(np.uint8)
    raw_residue_radiograph_np = raw_residue_radiograph_np.astype(np.uint8)
    processed_residue_radiograph_np = processed_residue_radiograph_np.astype(np.uint8)

    # saving
    cv2.imwrite(os.path.join(saving_dir, filename.replace('.png', '_patch_image.png')), image_np)
    cv2.imwrite(os.path.join(saving_dir, filename.replace('.png', '_patch_pixel_level_label.png')),
                pixel_level_label_np)
    cv2.imwrite(os.path.join(saving_dir, filename.replace('.png', '_patch_raw_residue.png')),
                raw_residue_radiograph_np)
    cv2.imwrite(os.path.join(saving_dir, filename.replace('.png', '_patch_post_processed_residue.png')),
                processed_residue_radiograph_np)
    sitk.WriteImage(stacked_image, os.path.join(saving_dir, filename.replace('png', 'nii')))

    return


def TestMicroCalcificationRadiographLevelDetection(args):
    # start time of this dataset
    start_time_for_dataset = time()

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(args.prediction_saving_dir):
        shutil.rmtree(args.prediction_saving_dir)
    #
    visualization_saving_dir = os.path.join(args.prediction_saving_dir, 'qualitative_results')
    radiograph_level_visualization_saving_dir = os.path.join(visualization_saving_dir, 'radiograph_level')
    patch_level_visualization_saving_dir = os.path.join(visualization_saving_dir, 'patch_level')
    #
    os.mkdir(args.prediction_saving_dir)
    os.mkdir(visualization_saving_dir)
    os.mkdir(radiograph_level_visualization_saving_dir)
    os.mkdir(patch_level_visualization_saving_dir)

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
    dataset = MicroCalcificationRadiographLevelDataset(data_root_dir=args.data_root_dir, mode=args.dataset_type)

    # create data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=r_cfg.train.num_threads)

    # set up metrics object
    metrics = MetricsRadiographLevelDetection(args.distance_threshold, args.score_threshold_stride)

    for radiograph_idx, (images_tensor, pixel_level_labels_tensor, _, filenames) in enumerate(data_loader):
        filename = filenames[0]

        # logging
        logger.write_and_print('--------------------------------------------------------------------------------------')
        logger.write_and_print(
            'Start evaluating radiograph {} out of {}: {}...'.format(radiograph_idx + 1, dataset.__len__(), filename))

        # start time of this radiograph
        start_time_for_radiograph = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # transfer the tensor into ndarray format
        pixel_level_label_np = pixel_level_labels_tensor.cpu().numpy().squeeze()

        # generated raw radiograph-level residue
        _, raw_residue_radiograph_np = generate_radiograph_level_reconstructed_and_residue_result(images_tensor,
                                                                                                  reconstruction_net,
                                                                                                  pixel_level_label_np,
                                                                                                  args.patch_size,
                                                                                                  args.patch_stride)

        # post-process the raw radiograph-level residue
        processed_residue_radiograph_np = post_process_residue_radiograph(raw_residue_radiograph_np,
                                                                          pixel_level_label_np,
                                                                          args.prob_threshold,
                                                                          args.area_threshold)

        # generate coordinates and score list for the post-processed radiograph-level residue
        pred_coord_list, pred_score_list = generate_coordinate_and_score_list(images_tensor,
                                                                              classification_net,
                                                                              pixel_level_label_np,
                                                                              raw_residue_radiograph_np,
                                                                              processed_residue_radiograph_np,
                                                                              filename,
                                                                              patch_level_visualization_saving_dir,
                                                                              args.patch_size)

        # generate coordinates list for the mask
        label_coord_list, _ = generate_coordinate_and_score_list(images_tensor,
                                                                 classification_net,
                                                                 pixel_level_label_np,
                                                                 raw_residue_radiograph_np,
                                                                 processed_residue_radiograph_np,
                                                                 filename,
                                                                 patch_level_visualization_saving_dir,
                                                                 args.patch_size,
                                                                 mode='annotated')

        # evaluate based on the above three lists
        if args.slack_for_recall:
            detection_result_record_radiograph_level = metrics.metric_all_score_thresholds(pred_coord_list,
                                                                                           pred_score_list,
                                                                                           label_coord_list,
                                                                                           processed_residue_radiograph_np)
        else:
            detection_result_record_radiograph_level = metrics.metric_all_score_thresholds(pred_coord_list,
                                                                                           pred_score_list,
                                                                                           label_coord_list)
        # save radiograph-level visualization results
        save_radiograph_level_results(images_tensor, pixel_level_label_np, raw_residue_radiograph_np,
                                      processed_residue_radiograph_np, filename,
                                      radiograph_level_visualization_saving_dir)

        # logging
        # print logging information of this radiograph
        logger.write_and_print(
            'Finish evaluating radiograph: {}, consuming time: {:.4f}s'.format(radiograph_idx + 1,
                                                                               time() - start_time_for_radiograph))
        detection_result_record_radiograph_level.print(logger)
        logger.write_and_print('--------------------------------------------------------------------------------------')
        logger.flush()

    # print logging information of this dataset
    logger.write_and_print(
        '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    logger.write_and_print(
        'Finished evaluating this dataset, consuming time: {:.4f}s'.format(time() - start_time_for_dataset))
    metrics.detection_result_record_dataset_level.print(logger)
    logger.write_and_print(
        '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationRadiographLevelDetection(args)
