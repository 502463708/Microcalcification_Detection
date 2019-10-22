import argparse
import os
import shutil

from utils.convert_xml_annotations_2_mask import image_with_xml2image_with_mask


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-raw-data-with-XML-annotations/',
                        help='Source data root dir.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-raw-data-with-pixel-level-labels/',
                        help='Destination data root dir.')
    parser.add_argument('--diameter_threshold_threshold',
                        type=float,
                        default=14,
                        help='The diameter threshold to filter large calcifications.')


    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)
    os.mkdir(os.path.join(args.dst_data_root_dir, 'images'))
    os.mkdir(os.path.join(args.dst_data_root_dir, 'labels'))

    return args


def TestConvertXml2Mask(args):
    src_image_dir = os.path.join(args.src_data_root_dir, 'images')
    src_xml_dir = os.path.join(args.src_data_root_dir, 'xml_annotations')

    dst_image_dir = os.path.join(args.dst_data_root_dir, 'images')
    dst_label_dir = os.path.join(args.dst_data_root_dir, 'labels')

    assert os.path.exists(src_image_dir)
    assert os.path.exists(src_xml_dir)

    image_filename_list = os.listdir(src_image_dir)

    current_idx = 0
    for image_filename in image_filename_list:
        current_idx += 1
        print('-------------------------------------------------------------------------------------------------------')
        print('Processing {} out of {}, filename: {}'.format(current_idx, len(image_filename_list), image_filename))

        xml_filename = image_filename.replace('png', 'xml')

        absolute_src_image_path = os.path.join(src_image_dir, image_filename)
        absolute_src_xml_path = os.path.join(src_xml_dir, xml_filename)
        absolute_dst_image_path = os.path.join(dst_image_dir, image_filename)
        absolute_dst_label_path = os.path.join(dst_label_dir, image_filename)

        image_with_xml2image_with_mask(absolute_src_image_path, absolute_src_xml_path, absolute_dst_image_path,
                                       absolute_dst_label_path, args.diameter_threshold)

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestConvertXml2Mask(args)
