import argparse
import os
import shutil

from utils.image_level_dataset_partition import crop_process, image_level_datset_partition, saveimg


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-raw-data-with-pixel-level-labels/',
                        help='Source data root dir.')

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-roi-data-with-pixel-level-labels-divided/',
                        help='Destination data root dir.')

    parser.add_argument('--train_ratio',
                        type=float,
                        default=0.6,
                        help='the ratio of train dataset over all')

    parser.add_argument('--validation_ratio',
                        type=float,
                        default=0.2,
                        help='the ratio of validation dataset over all')

    parser.add_argument('--test_ratio',
                        type=float,
                        default=0.2,
                        help='the ratio of test dataset over all')

    parser.add_argument('--crop_size',
                        type=int,
                        default=600,
                        help='the maximum size of cropping')

    args = parser.parse_args()

    assert os.path.exists(args.data_root_dir), 'Source data root dir does not exist.'

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    for mode in ['training', 'validation', 'test']:
        mode_dir = os.path.join(args.dst_data_root_dir, mode)
        os.mkdir(mode_dir)
        os.mkdir(os.path.join(mode_dir, 'images'))
        os.mkdir(os.path.join(mode_dir, 'labels'))

    return args


def TestImageLevelSplit(args):
    image_data_root_dir = os.path.join(args.data_root_dir, 'images')
    label_data_root_dir = os.path.join(args.data_root_dir, 'labels')

    assert os.path.exists(image_data_root_dir), 'Source image data root dir does not exist.'
    assert os.path.exists(label_data_root_dir), 'Source label data root dir does not exist.'

    image_train, image_val, image_test = image_level_datset_partition(image_data_root_dir,
                                                                      train_ratio=args.train_ratio,
                                                                      validation_ratio=args.validation_ratio,
                                                                      test_ratio=args.test_ratio)

    saveimg(name_list=image_train, data_dir=image_data_root_dir, label_dir=label_data_root_dir,
            save_path=args.dst_data_root_dir, mode='training')
    saveimg(name_list=image_val, data_dir=image_data_root_dir, label_dir=label_data_root_dir,
            save_path=args.dst_data_root_dir, mode='validation')
    saveimg(name_list=image_test, data_dir=image_data_root_dir, label_dir=label_data_root_dir,
            save_path=args.dst_data_root_dir, mode='test')

    # data images crop
    for mode in ['training', 'validation', 'test']:
        my_dir = os.path.join(args.dst_data_root_dir, mode)
        img_dir = os.path.join(my_dir, 'images')
        lab_dir = os.path.join(my_dir, 'labels')
        crop_process(img_dir, lab_dir, crop_size=args.crop_size)
        crop_process(img_dir, lab_dir, crop_size=args.crop_size)
        crop_process(img_dir, lab_dir, crop_size=args.crop_size)

    print('finish dataset split')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestImageLevelSplit(args)
