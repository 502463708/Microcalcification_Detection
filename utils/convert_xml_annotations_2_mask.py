import cv2
import numpy as np
import os
import shutil
import xml.etree.ElementTree as ET

from skimage.draw import polygon
from skimage import measure


class Annotation(object):
    def __init__(self, xml_node):
        children_nodes = xml_node.getchildren()
        self.name = children_nodes[15].text
        self.area = float(children_nodes[1].text)
        self.coordinate_list = self.get_coordinate_list(children_nodes[21])
        self.pixel_number = len(self.coordinate_list)

        return

    def print_details(self):
        print('  name: {}'.format(self.name))
        print('  area: {}'.format(self.area))
        print('  pixel_number: {}'.format(self.pixel_number))
        print('  *****************************************************************************************************')

        return

    def get_coordinate_list(self, coordinate_root_node):
        coordinate_list = list()
        coordinate_child_nodes = coordinate_root_node.getchildren()
        for coordinate_child_node in coordinate_child_nodes:
            pixel_coordinate_text = coordinate_child_node.text
            pixel_coordinate_text = pixel_coordinate_text[1: -1]
            pixel_coordinates_text_list = pixel_coordinate_text.split(',')
            x = round(float(pixel_coordinates_text_list[0]))
            y = round(float(pixel_coordinates_text_list[1]))
            coordinate_list.append([x, y])

        return coordinate_list


class ImageLevelAnnotationCollection(object):
    def __init__(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        animNode = root.find('dict')
        animNode = animNode.find('array')
        animNode = animNode.find('dict')
        annotation_root_node = animNode.find('array')
        annotation_child_nodes = annotation_root_node.findall('dict')
        self.annotation_list = list()
        for annotation_child_node in annotation_child_nodes:
            annotation_obj = Annotation(annotation_child_node)
            self.annotation_list.append(annotation_obj)

        return


def generate_null_label(absolute_src_image_path, absolute_dst_label_path):
    image_np = cv2.imread(absolute_src_image_path, cv2.IMREAD_GRAYSCALE)
    label_np = np.zeros_like(image_np, dtype='uint8')
    cv2.imwrite(absolute_dst_label_path, label_np)

    return


def generate_label_according_to_xml(absolute_src_image_path, absolute_src_xml_path, absolute_dst_label_path,
                                    diameter_threshold=14):
    image_np = cv2.imread(absolute_src_image_path, cv2.IMREAD_GRAYSCALE)
    xml_obj = ImageLevelAnnotationCollection(absolute_src_xml_path)

    label_np = np.zeros_like(image_np)  # column, row
    calcification_mask_np = np.zeros_like(image_np)  # column, row
    other_lesion_mask_np = np.zeros_like(image_np)  # column, row

    # for statistical purpose
    qualified_calcification_count_image_level = 0
    outlier_calcification_count_image_level = 0
    other_lesion_count_image_level = 0

    # generate calcification_mask_np and other_lesion_mask_np
    for annotation in xml_obj.annotation_list:
        annotation.print_details()

        # convert the outline annotation into area annotation
        coordinate_list = np.array(annotation.coordinate_list) - 1
        cc, rr = polygon(coordinate_list[:, 0], coordinate_list[:, 1])

        # for the calcification annotations
        if annotation.name in ['Calcification', 'Calcifications', 'Unnamed', 'Point 1', 'Point 3']:
            # in case that only one pixel is annotated
            if len(rr) == 0:
                for coordinate in coordinate_list:
                    calcification_mask_np[coordinate[1], coordinate[0]] = 255
            else:
                # to avoid the situation that coordinate indexes get out of range
                height, width = image_np.shape
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)
                calcification_mask_np[rr, cc] = 255  # row ,column

        # for the other lesion annotations
        else:
            other_lesion_count_image_level += 1
            # in case that only one pixel is annotated
            if len(rr) == 0:
                for coordinate in coordinate_list:
                    other_lesion_mask_np[coordinate[1], coordinate[0]] = 255
            else:
                # to avoid the situation that coordinate indexes get out of range
                height, width = image_np.shape
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)
                other_lesion_mask_np[rr, cc] = 255  # row ,column

    # analyse the connected components for calcification_mask_np
    calcification_connected_components = measure.label(input=calcification_mask_np, connectivity=2)
    calcification_connected_component_props = measure.regionprops(calcification_connected_components)

    for prop in calcification_connected_component_props:
        # a large calcification is considered as an outlier calcification
        if prop.equivalent_diameter >= diameter_threshold:
            outlier_calcification_count_image_level += 1
            coordinates = prop.coords
            for coordinate in coordinates:
                hd = coordinate[0]
                wd = coordinate[1]
                label_np[hd][wd] = 125

        # a tiny calcification is considered as a qualified calcification
        else:
            qualified_calcification_count_image_level += 1
            coordinates = prop.coords
            for coordinate in coordinates:
                hd = coordinate[0]
                wd = coordinate[1]
                label_np[hd][wd] = 255

    label_np[other_lesion_mask_np == 255] = 125

    cv2.imwrite(absolute_dst_label_path, label_np)
    print('This image contains {} qualified calcifications.'.format(qualified_calcification_count_image_level))
    print('This image contains {} outlier calcifications.'.format(outlier_calcification_count_image_level))
    print('This image contains {} other lesions.'.format(other_lesion_count_image_level))

    return qualified_calcification_count_image_level, outlier_calcification_count_image_level, \
           other_lesion_count_image_level


def image_with_xml2image_with_mask(absolute_src_image_path, absolute_src_xml_path, absolute_dst_image_path,
                                   absolute_dst_label_path, diameter_threshold):
    # the source image must exist
    assert os.path.exists(absolute_src_image_path)

    # copy the image file into the destination image folder
    shutil.copyfile(absolute_src_image_path, absolute_dst_image_path)

    # for statistical purpose
    qualified_calcification_count_image_level = 0
    outlier_calcification_count_image_level = 0
    other_lesion_count_image_level = 0

    # if this image does not have its corresponding xml file -> generate a mask which is completely filled with 0
    if not os.path.exists(absolute_src_xml_path):
        print('This image does not have xml annotation.')
        generate_null_label(absolute_src_image_path, absolute_dst_label_path)
    # if this image has its corresponding xml file -> generate a mask according to its xml file
    else:
        print('This image has xml annotation.')
        if diameter_threshold == -1:
            qualified_calcification_count_image_level, outlier_calcification_count_image_level, \
            other_lesion_count_image_level = generate_label_according_to_xml(absolute_src_image_path,
                                                                             absolute_src_xml_path,
                                                                             absolute_dst_label_path)
        else:
            qualified_calcification_count_image_level, outlier_calcification_count_image_level, \
            other_lesion_count_image_level = generate_label_according_to_xml(absolute_src_image_path,
                                                                             absolute_src_xml_path,
                                                                             absolute_dst_label_path,
                                                                             diameter_threshold)

    return qualified_calcification_count_image_level, outlier_calcification_count_image_level, \
           other_lesion_count_image_level
