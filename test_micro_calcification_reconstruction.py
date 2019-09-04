import cv2
import numpy as np
import os
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn

from config.config_micro_calcification_reconstruction import cfg
from dataset.dataset_micro_calcification import MicroCalcificationDataset
from metrics.metrics_reconstruction import MetricsReconstruction
from net.vnet2d_v2 import VNet2d
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
cudnn.benchmark = True

model_saving_dir = '/data/lars/models/20190830_uCs_reconstruction_ttestloss_dilation_radius_14/'
epoch_idx = 470
dataset = 'test'  # 'training', 'validation' or 'test'

prob_threshold = 0.2  # residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1
area_threshold = 3.14 * 14 * 14 / 3  # connected components whose area < area_threshold will be discarded
distance_threshold = 14  # candidates whose distance between calcification < distance_threshold can be a recalled one

batch_size = 48


def save_tensor_in_png_and_nii_format(images_tensor, reconstructed_images_tensor, prediction_residues_tensor,
                                      post_process_preds_np, pixel_level_labels_tensor,
                                      pixel_level_labels_dilated_tensor, filenames, prediction_saving_dir):
    # convert tensor into numpy and squeeze channel dimension
    images_np = images_tensor.cpu().detach().numpy().squeeze(axis=1)
    reconstructed_images_np = reconstructed_images_tensor.cpu().detach().numpy().squeeze(axis=1)
    prediction_residues_np = prediction_residues_tensor.cpu().detach().numpy().squeeze(axis=1)
    pixel_level_labels_np = pixel_level_labels_tensor.numpy()
    pixel_level_labels_dilated_np = pixel_level_labels_dilated_tensor.numpy()

    # iterating each image of this batch
    for idx in range(images_np.shape[0]):
        image_np = images_np[idx, :, :]
        reconstructed_image_np = reconstructed_images_np[idx, :, :]
        prediction_residue_np = prediction_residues_np[idx, :, :]
        post_process_pred_np = post_process_preds_np[idx, :, :]
        pixel_level_label_np = pixel_level_labels_np[idx, :, :]
        pixel_level_label_dilated_np = pixel_level_labels_dilated_np[idx, :, :]
        filename = filenames[idx]

        stacked_np = np.concatenate((np.expand_dims(image_np, axis=0),
                                     np.expand_dims(reconstructed_image_np, axis=0),
                                     np.expand_dims(prediction_residue_np, axis=0),
                                     np.expand_dims(post_process_pred_np, axis=0),
                                     np.expand_dims(pixel_level_label_np, axis=0),
                                     np.expand_dims(pixel_level_label_dilated_np, axis=0)), axis=0)

        stacked_image = sitk.GetImageFromArray(stacked_np)
        sitk.WriteImage(stacked_image, os.path.join(prediction_saving_dir, filename.replace('png', 'nii')))

        image_np *= 255
        reconstructed_image_np *= 255
        prediction_residue_np *= 255
        post_process_pred_np *= 255
        pixel_level_label_np *= 255
        pixel_level_label_dilated_np *= 255

        image_np = image_np.astype(np.uint8)
        reconstructed_image_np = reconstructed_image_np.astype(np.uint8)
        prediction_residue_np = prediction_residue_np.astype(np.uint8)
        post_process_pred_np = post_process_pred_np.astype(np.uint8)
        pixel_level_label_np = pixel_level_label_np.astype(np.uint8)
        pixel_level_label_dilated_np = pixel_level_label_dilated_np.astype(np.uint8)

        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_image.png')),
                    image_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_reconstructed.png')),
                    reconstructed_image_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_residue.png')),
                    prediction_residue_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_post_processed_residue.png')),
                    post_process_pred_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pixel_level_label.png')),
                    pixel_level_label_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pixel_level_dilated_label.png')),
                    pixel_level_label_dilated_np)

    return


if __name__ == '__main__':
    prediction_saving_dir = os.path.join(model_saving_dir, 'results_dataset_{}_epoch_{}'.format(dataset, epoch_idx))
    if not os.path.exists(prediction_saving_dir):
        os.mkdir(prediction_saving_dir)

    # define the network
    net = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)

    # load the specified ckpt
    ckpt_dir = os.path.join(model_saving_dir, 'ckpt')
    ckpt_path = os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(epoch_idx))

    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(ckpt_path))
    net = net.eval()
    print('Load ckpt: {0}...'.format(ckpt_path))

    # create dataset and data loader
    dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                        mode=dataset,
                                        enable_random_sampling=False,
                                        pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                        image_channels=cfg.dataset.image_channels,
                                        cropping_size=cfg.dataset.cropping_size,
                                        dilation_radius=0,
                                        enable_data_augmentation=False)
    #
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    metrics = MetricsReconstruction(prob_threshold, area_threshold, distance_threshold)

    calcification_num = 0
    recall_num = 0
    FP_num = 0

    for batch_idx, (images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor,
                    image_level_labels_tensor, filenames) in enumerate(data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # network forward
        reconstructed_images_tensor, prediction_residues_tensor = net(images_tensor)

        # evaluation
        post_process_preds_np, calcification_num_batch_level, recall_num_batch_level, FP_num_batch_level = \
            metrics.metric_batch_level(prediction_residues_tensor, pixel_level_labels_tensor)

        calcification_num += calcification_num_batch_level
        recall_num += recall_num_batch_level
        FP_num += FP_num_batch_level

        print('The number of the annotated calcifications of this batch = {}'.format(calcification_num_batch_level))
        print('The number of the recalled calcifications of this batch = {}'.format(recall_num_batch_level))
        print('The number of the false positive calcifications of this batch = {}'.format(FP_num_batch_level))

        # print logging information
        print('batch: {}, consuming time: {:.4f}s'.format(batch_idx, time() - start_time_for_batch))
        print('-------------------------------------------------------------------------------------------------------')

        save_tensor_in_png_and_nii_format(images_tensor, reconstructed_images_tensor, prediction_residues_tensor,
                                          post_process_preds_np, pixel_level_labels_tensor,
                                          pixel_level_labels_dilated_tensor, filenames, prediction_saving_dir)

    print('The number of the annotated calcifications of this dataset = {}'.format(calcification_num))
    print('The number of the recalled calcifications of this dataset = {}'.format(recall_num))
    print('The number of the false positive calcifications of this dataset = {}'.format(FP_num))
