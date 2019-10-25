import argparse
import os
import shutil

from utils.convert_xml_annotations_2_mask import image_with_xml2image_with_mask


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-radiograph-level-raw-images-with-XML-annotations-dataset/',
                        help='Source data root dir.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-radiograph-level-raw-images-with-pixel-level-labels-dataset/',
                        help='Destination data root dir.')
    parser.add_argument('--diameter_threshold',
                        type=float,
                        default=-1,
                        help='The calcifications whose diameter >= diameter_threshold will be discarded,'
                             '-1 -> the default diameter_threshold = 14.')

    args = parser.parse_args()

    # the source data root dir must exist
    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    # remove the destination data root dir if it already exists
    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)

    # create brand new destination data root dir
    os.mkdir(args.dst_data_root_dir)
    os.mkdir(os.path.join(args.dst_data_root_dir, 'images'))
    os.mkdir(os.path.join(args.dst_data_root_dir, 'labels'))

    return args


def TestConvertXml2Mask(args):
    src_image_dir = os.path.join(args.src_data_root_dir, 'images')
    src_xml_dir = os.path.join(args.src_data_root_dir, 'xml_annotations')

    dst_image_dir = os.path.join(args.dst_data_root_dir, 'images')
    dst_label_dir = os.path.join(args.dst_data_root_dir, 'labels')

    # the source data root dir must contain images and labels
    assert os.path.exists(src_image_dir)
    assert os.path.exists(src_xml_dir)

    image_filename_list = os.listdir(src_image_dir)

    # for statistical purpose
    qualified_calcification_count_dataset_level = 0
    outlier_calcification_count_dataset_level = 0
    other_lesion_count_dataset_level = 0

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

        qualified_calcification_count_image_level, outlier_calcification_count_image_level, \
        other_lesion_count_image_level = image_with_xml2image_with_mask(absolute_src_image_path, absolute_src_xml_path,
                                                                        absolute_dst_image_path,
                                                                        absolute_dst_label_path,
                                                                        args.diameter_threshold)

        qualified_calcification_count_dataset_level += qualified_calcification_count_image_level
        outlier_calcification_count_dataset_level += outlier_calcification_count_image_level
        other_lesion_count_dataset_level += other_lesion_count_image_level

    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('This dataset contains {} qualified calcifications.'.format(qualified_calcification_count_dataset_level))
    print('This dataset contains {} outlier calcifications.'.format(outlier_calcification_count_dataset_level))
    print('This dataset contains {} other lesions.'.format(other_lesion_count_dataset_level))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestConvertXml2Mask(args)
