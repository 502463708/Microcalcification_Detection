import argparse
import os
import shutil

from utils.equal_patch_generation import SaveEqualPatch


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-patch-level-split-dataset/',
                        help='Source data root dir.')

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-splitted-data-with-pixel-level-labels/',
                        help='Destination data root dir.')

    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    for patch_type in ['positive_patches', 'negative_patches']:
        patch_type_dir = os.path.join(args.dst_data_root_dir, patch_type)
        os.mkdir(patch_type_dir)
        for dataset_type in ['training', 'validation', 'test']:
            dataset_type_dir = os.path.join(patch_type_dir, dataset_type)
            os.mkdir(dataset_type_dir)
            os.mkdir(os.path.join(dataset_type_dir, 'images'))
            os.mkdir(os.path.join(dataset_type_dir, 'labels'))

    return args


def TestEqualPatchGeneration(args):
    for mode in ['training', 'validation', 'test']:
        SaveEqualPatch(args.data_root_dir, args.dst_root_dir, mode)


if __name__ == '__main__':
    args = ParseArguments()

    TestEqualPatchGeneration(args)
