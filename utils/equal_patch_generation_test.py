import argparse
import os
import shutil

from utils.equal_patch_generation import SaveEqualPatch


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/models/20190920_uCs_reconstruction_connected_1_ttestlossv2_default_dilation_radius_14/',
                        help='Source data root dir.')

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-splitted-data-with-pixel-level-labels/',
                        help='Destination data root dir.')

    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset (training, validation, test).')

    args = parser.parse_args()

    assert os.path.exists(args.data_root_dir), 'Source data root dir does not exist.'
    positive_path = os.path.join(args.dst_data_root_dir, 'positive_patches', args.dataset_type)
    negative_path = positive_path.replace('positive', 'negative')

    if os.path.exists(positive_path):
        shutil.rmtree(positive_path)
    if os.path.exists(negative_path):
        shutil.rmtree(negative_path)
    if not os.path.exists(args.dst_data_root_dir):
        os.mkdir(args.dst_data_root_dir)
        os.mkdir(os.path.join(args.dst_data_root_dir, 'positive_patches'))
        os.mkdir(os.path.join(args.dst_data_root_dir, 'negative_patches'))
    os.mkdir(positive_path)
    os.mkdir(negative_path)

    return args


def TestEqualPatchGeneration(args):
    SaveEqualPatch(args.data_root_dir, args.dst_root_dir, mode=args.dataset_type)



if __name__=='__main__':
    args=ParseArguments()

    TestEqualPatchGeneration(args)
