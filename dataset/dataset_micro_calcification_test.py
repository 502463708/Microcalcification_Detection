import cv2
import numpy as np
import os
import shutil
import argparse

from config.config_micro_calcification_reconstruction import cfg
from dataset.dataset_micro_calcification import MicroCalcificationDataset
from torch.utils.data import DataLoader

def ParseArguments():

    parser = argparse.ArgumentParser()


    parser.add_argument('--output_dir',

                        type=str,

                        default='/home/groupprofzli/data1/dwz/data/Inbreast-dataset-cropped-pathches-connected-component-1/',

                        help='Destination data dir.')

    parser.add_argument('--mode',
                        type=str,
                        default='training',
                        help='within training , validation or test')


    parser.add_argument('--num_epoch',
                        type=int,
                        default=5,
                        help='epoch for test')

    parser.add_argument('--batch_size',
                        type=int,
                        default=480,
                        help='the patch number in each batch')


    parser.add_argument('--num_workers',
                        type=int,
                        default=24,
                        help='')

    args = parser.parse_args()



    return args


def micro_calcification_reconstruction_dataset_test(args):
    # remove the existing folder with the same name
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)

    # create a new folder
    os.mkdir(args.output_dir)

    # create dataset for training
    training_dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                                 mode=args.mode,
                                                 enable_random_sampling=False,
                                                 pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                                 image_channels=cfg.dataset.image_channels,
                                                 cropping_size=cfg.dataset.cropping_size,
                                                 dilation_radius=cfg.dataset.dilation_radius,
                                                 enable_data_augmentation=cfg.dataset.augmentation.enable_data_augmentation,
                                                 enable_vertical_flip=cfg.dataset.augmentation.enable_vertical_flip,
                                                 enable_horizontal_flip=cfg.dataset.augmentation.enable_horizontal_flip)

    # create data loader for training
    training_data_loader = DataLoader(training_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)

    # enumerating
    for epoch_idx in range(args.num_epoch):
        # create folder for this epoch
        output_dir_epoch = os.path.join(args.output_dir, 'epoch_{0}'.format(epoch_idx))
        os.mkdir(output_dir_epoch)

        print('-------------------------------------------------------------------------------------------------------')
        print('Loading epoch {0}...'.format(epoch_idx))

        # the following two variables are used for counting positive and negative patch number in an epoch
        positive_patch_num_for_this_epoch = 0
        negative_patch_num_for_this_epoch = 0

        for batch_idx, (images_tensor, _, _, image_level_labels_tensor, filenames) in enumerate(training_data_loader):
            # create folder for this batch
            output_dir_batch = os.path.join(output_dir_epoch, 'batch_{0}'.format(batch_idx))
            os.mkdir(output_dir_batch)

            # create folder for saving positive patches
            output_dir_positive = os.path.join(output_dir_batch, 'positive')
            os.mkdir(output_dir_positive)

            # create folder for saving negative patches
            output_dir_negative = os.path.join(output_dir_batch, 'negative')
            os.mkdir(output_dir_negative)

            # the following two variables are used for counting positive and negative patch number in a batch
            positive_patch_num_for_this_batch = 0
            negative_patch_num_for_this_batch = 0

            images_np = images_tensor.cpu().numpy()
            image_level_labels_np = image_level_labels_tensor.cpu().numpy()

            for image_idx in range(images_np.shape[0]):
                image_np = images_np[image_idx, 0, :, :]
                label = image_level_labels_np[image_idx, 0]
                filename = filenames[image_idx]

                image_np *= 255
                image_np = image_np.astype(np.uint8)

                # label of each patch is either 0 or 1
                assert label in [0, 1]

                if label == 1:
                    cv2.imwrite(os.path.join(output_dir_positive, filename), image_np)
                    positive_patch_num_for_this_epoch += 1
                    positive_patch_num_for_this_batch += 1
                elif label == 0:
                    cv2.imwrite(os.path.join(output_dir_negative, filename), image_np)
                    negative_patch_num_for_this_epoch += 1
                    negative_patch_num_for_this_batch += 1

            print('----batch {0} loading finished; '
                  'positive patches: {1}, negative patches: {2}'.format(batch_idx,
                                                                        positive_patch_num_for_this_batch,
                                                                        negative_patch_num_for_this_batch))

        print('epoch {0} loading finished; '
              'positive patches: {1}, negative patches: {2}'.format(epoch_idx,
                                                                    positive_patch_num_for_this_epoch,
                                                                    negative_patch_num_for_this_epoch))

    return 0


if __name__ == '__main__':
    # saving results dir


    args= ParseArguments()

    micro_calcification_reconstruction_dataset_test(args)
