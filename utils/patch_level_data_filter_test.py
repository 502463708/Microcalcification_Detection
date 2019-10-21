import argparse

from utils.patch_level_data_filter import DatasetSplit


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-dataset-cropped-pathches/',
                        help='Source data dir.')
    parser.add_argument('--connected_component_threshold',
                        type=int,
                        default=1,
                        help='The threshold to select legal positive patches.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-dataset-cropped-pathches-connected-component-1/',
                        help='Destination data dir.')

    args = parser.parse_args()

    return args


def TestDadasetSplit(args):
    dataset_split_obj = DatasetSplit(args.data_root_dir, args.connected_component_threshold, args.output_dir)
    dataset_split_obj.run()

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestDadasetSplit(args)
