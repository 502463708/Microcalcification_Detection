import argparse

from utils.patch_level_dataset_generation import makedir, ExtractPatch, save_patch, LoadImage


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-roi-data-with-pixel-level-labels-divided/',
                        help='The source data dir.')

    parser.add_argument('--data_saving_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-roi-data-divided-cropped-all-patches/',
                        help='The dataset saved dir.')

    parser.add_argument('--patch_size',
                        type=tuple,
                        default=(112, 112),
                        help='The height and width of patch.')

    parser.add_argument('--patch_stride',
                        type=int,
                        default=56,
                        help='The stride move from one patch to another ')

    parser.add_argument('--patch_threshold',
                        type=int,
                        default=30000,
                        help='patch which np.sum is lower than threshold will be discarded')

    args = parser.parse_args()

    return args


def TestPatchLevelSplit(args):
    mysave_dir = makedir(args.data_saving_dir)

    for mod in ['training', 'validation', 'test']:
        myname_list, myimage_list, mylabel_list, mymode = LoadImage(
            folder_dir=args.data_root_dir, mode=mod)
        for idx in range(len(myname_list)):
            myimage_patch_list = ExtractPatch(myimage_list[idx], patch_size=(112, 112), stride=56)
            mylabel_patch_list = ExtractPatch(mylabel_list[idx], patch_size=(112, 112), stride=56)
            save_patch(mysave_dir, mymode, myimage_patch_list, mylabel_patch_list, myname_list[idx], threshold=1000000)

            myimage_patch_list = ExtractPatch(myimage_list[idx], patch_size=args.patch_size, stride=args.patch_stride)
            mylabel_patch_list = ExtractPatch(mylabel_list[idx], patch_size=args.patch_size, stride=args.patch_stride)
            save_patch(mysave_dir, mymode, myimage_patch_list, mylabel_patch_list, myname_list[idx],
                       threshold=args.patch_threshold)


if __name__ == '__main__':
    args = ParseArguments()

    TestPatchLevelSplit(args)
