import argparse
import os
import shutil

from logger.logger import Logger
from utils.patch_level_sub_dataset_generation import filename_list_split, copy_data_from_src_2_dst


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/',
                        help='Source data dir.')
    parser.add_argument('--sub_dataset_number',
                        type=int,
                        default=2,
                        help='The number of the sub dataset gonna be created.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-sub-datasets/',
                        help='Destination data dir.')
    parser.add_argument('--random_seed',
                        type=int,
                        default=0,
                        help='Set random seed for reduplicating the results.'
                             '-1 -> do not set random seed.')
    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    for sub_dataset_idx in range(args.sub_dataset_number):
        sub_dataset_name = 'sub_dataset_{}'.format(sub_dataset_idx + 1)
        sub_dataset_dir = os.path.join(args.dst_data_root_dir, sub_dataset_name)
        os.mkdir(sub_dataset_dir)

        for patch_type in ['positive_patches', 'negative_patches']:
            patch_type_dir = os.path.join(sub_dataset_dir, patch_type)
            os.mkdir(patch_type_dir)

            for dataset_type in ['training', 'validation', 'test']:
                dataset_type_dir = os.path.join(patch_type_dir, dataset_type)
                os.mkdir(dataset_type_dir)
                os.mkdir(os.path.join(dataset_type_dir, 'images'))
                os.mkdir(os.path.join(dataset_type_dir, 'labels'))

    return args


def TestPatchLevelSubDatasetGeneration(args):
    # set up logger
    logger = Logger(args.dst_data_root_dir)
    for patch_type in ['positive_patches', 'negative_patches']:

        for dataset_type in ['training', 'validation', 'test']:
            src_dataset_type_dir = os.path.join(args.src_data_root_dir, patch_type, dataset_type)
            sub_filename_list_list = filename_list_split(src_dataset_type_dir, args.sub_dataset_number, patch_type,
                                                         args.random_seed, logger=logger)
            assert len(sub_filename_list_list) == args.sub_dataset_number

            for sub_dataset_idx in range(args.sub_dataset_number):
                sub_filename_list = sub_filename_list_list[sub_dataset_idx]

                sub_dataset_name = 'sub_dataset_{}'.format(sub_dataset_idx + 1)
                dst_dataset_type_dir = os.path.join(args.dst_data_root_dir, sub_dataset_name, patch_type, dataset_type)

                copy_data_from_src_2_dst(src_dataset_type_dir, dst_dataset_type_dir, sub_filename_list, sub_dataset_idx,
                                         dataset_type, patch_type, logger=logger)
    return


if __name__ == '__main__':
    args = ParseArguments()

    TestPatchLevelSubDatasetGeneration(args)