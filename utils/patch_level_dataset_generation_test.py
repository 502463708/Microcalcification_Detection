import argparse
import os
import shutil

from utils.patch_level_dataset_generation import crop_patches_and_labels, filter_and_save_patches_and_labels


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-roi-extracted-radiograph-level-split-dataset/',
                        help='The source data root dir.')

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-patch-level-split-dataset/',
                        help='The destination data root dir.')

    parser.add_argument('--patch_size',
                        type=tuple,
                        default=(112, 112),
                        help='The height and width of patch.')

    parser.add_argument('--stride',
                        type=int,
                        default=56,
                        help='The stride move from one patch to another.')

    parser.add_argument('--pixel_threshold',
                        type=int,
                        default=1,
                        help='Pixels whose intensity < pixel_threshold will be considered as background.')

    parser.add_argument('--training_area_threshold',
                        type=float,
                        default=0.6,
                        help='The maximum background area ratio endurance for training set.')

    parser.add_argument('--validation_area_threshold',
                        type=float,
                        default=0.6,
                        help='The maximum background area ratio endurance for validation set.')

    parser.add_argument('--test_area_threshold',
                        type=float,
                        default=0.1,
                        help='The maximum background area ratio endurance for test set.')

    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    for patch_type in ['positive_patches', 'negative_patches']:
        os.mkdir(os.path.join(args.dst_data_root_dir, patch_type))
        for dataset_type in ['training', 'validation', 'test']:
            os.mkdir(os.path.join(args.dst_data_root_dir, patch_type, dataset_type))
            for image_type in ['images', 'labels']:
                os.mkdir(os.path.join(args.dst_data_root_dir, patch_type, dataset_type, image_type))

    return args


def TestPatchLevelDatasetGeneration(args):
    for dataset_type in ['training', 'validation', 'test']:
        absolute_image_dir = os.path.join(args.src_data_root_dir, dataset_type, 'images')
        filename_list = os.listdir(absolute_image_dir)

        assert len(filename_list) > 0

        area_threshold = args.training_area_threshold
        if dataset_type == 'validation':
            area_threshold = args.validation_area_threshold
        elif dataset_type == 'test':
            area_threshold = args.test_area_threshold

        current_idx = 0
        pos_patch_count_dataset_level = 0
        neg_patch_count_dataset_level = 0
        other_lesion_patch_count_dataset_level = 0
        background_patch_count_dataset_level = 0

        for filename in filename_list:
            current_idx += 1
            print('---------------------------------------------------------------------------------------------------')
            print(
                'Processing {} out of {}: {} in {} set'.format(current_idx, len(filename_list), filename, dataset_type))

            absolute_image_path = os.path.join(absolute_image_dir, filename)
            absolute_label_path = absolute_image_path.replace('images', 'labels')

            patch_list, label_list = crop_patches_and_labels(absolute_image_path, absolute_label_path,
                                                             patch_size=args.patch_size, stride=args.stride)

            pos_patch_count_image_level, \
            neg_patch_count_image_level, \
            other_lesion_patch_count_image_level, \
            background_patch_count_image_level = filter_and_save_patches_and_labels(args.dst_data_root_dir,
                                                                                    dataset_type, patch_list,
                                                                                    label_list, filename,
                                                                                    pixel_threshold=args.pixel_threshold,
                                                                                    area_threshold=area_threshold)

            pos_patch_count_dataset_level += pos_patch_count_image_level
            neg_patch_count_dataset_level += neg_patch_count_image_level
            other_lesion_patch_count_dataset_level += other_lesion_patch_count_image_level
            background_patch_count_dataset_level += background_patch_count_image_level

        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('The {} set contains {} positive patches, {} negative patches, {} other_lesion_pathces, '
              '{} background_patches.'.format(dataset_type,
                                              pos_patch_count_dataset_level,
                                              neg_patch_count_dataset_level,
                                              other_lesion_patch_count_dataset_level,
                                              background_patch_count_dataset_level))
        print('Totally {} patches have been cropped.'.format(other_lesion_patch_count_dataset_level +
                                                             background_patch_count_dataset_level +
                                                             pos_patch_count_dataset_level +
                                                             neg_patch_count_dataset_level))
        print('Totally {} patches have been discarded.'.format(other_lesion_patch_count_dataset_level +
                                                               background_patch_count_dataset_level))
        print('Totally {} patches have been saved.'.format(pos_patch_count_dataset_level +
                                                           neg_patch_count_dataset_level))
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestPatchLevelDatasetGeneration(args)
