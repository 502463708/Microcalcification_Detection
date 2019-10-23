import argparse

from utils.patch_level_dataset_generation import makedir, extract_patch, save_patch, load_image


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-dataset-cropped-pathches-connected-component-1/',
                        help='The source data dir.')

    parser.add_argument('--data_saving_dir',
                        type=str,
                        default='/data/lars/models/20190920_uCs_reconstruction_connected_1_ttestlossv2_default_dilation_radius_14/',
                        help='The dataset saved dir.')

    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset (training, validation, test).')

    parser.add_argument('--patch_size',
                        type=tuple,
                        default=(112, 112),
                        help='The height and width of patch.')

    parser.add_argument('--patch_stride',
                        type=int,
                        default=56,
                        help='The stride move from one patch to another.')

    parser.add_argument('--pixel_threshold',
                        type=int,
                        default=1,
                        help='patch pixel under threshold will be treated as background.')

    parser.add_argument('--training_area_threshold',
                        type=int,
                        default=112 * 112 * 0.4,
                        help='patch pixel background area under threshold will be discarded.')
    parser.add_argument('--validation_area_threshold',
                        type=int,
                        default=112 * 112 * 0.4,
                        help='patch pixel background area under threshold will be discarded.')
    parser.add_argument('--test_area_threshold',
                        type=int,
                        default=112 * 112 * 0.95,
                        help='patch pixel background area under threshold will be discarded.')

    args = parser.parse_args()

    return args


def TestPatchLevelSplit(args):
    mysave_dir = makedir(args.data_saving_dir)

    for mod in ['training', 'validation', 'test']:
        myname_list, myimage_list, mylabel_list, mymode = load_image(
            folder_dir=args.data_root_dir, mode=mod)
        if mod == 'training':
            area_threshold = args.training_area_threshold
        elif mod == 'validation':
            area_threshold = args.validation_area_threshold
        elif mod == 'test':
            area_threshold = args.test_area_threshold
        for idx in range(len(myname_list)):
            myimage_patch_list = extract_patch(myimage_list[idx], patch_size=args.patch_size, stride=args.patch_stride)
            mylabel_patch_list = extract_patch(mylabel_list[idx], patch_size=args.patch_size, stride=args.patch_stride)
            save_patch(mysave_dir, mymode, myimage_patch_list, mylabel_patch_list, myname_list[idx],
                       pixel_threshold=args.pixel_threshold, area_threshold=area_threshold)


if __name__ == '__main__':
    args = ParseArguments()

    TestPatchLevelSplit(args)
