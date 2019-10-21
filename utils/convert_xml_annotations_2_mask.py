import cv2
import numpy as np
import os
import shutil
import xml.etree.ElementTree as ET

from skimage.draw import polygon


class Annotation(object):
    def __init__(self, xml_node):
        children_nodes = xml_node.getchildren()
        self.name = children_nodes[15].text
        self.area = float(children_nodes[1].text)
        self.coordinate_list = self.get_coordinate_list(children_nodes[21])
        self.pixel_number = len(self.coordinate_list)

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
                                    area_threshold=0.006):
    image_np = cv2.imread(absolute_src_image_path, cv2.IMREAD_GRAYSCALE)
    xml_obj = ImageLevelAnnotationCollection(absolute_src_xml_path)
    label_np = np.zeros_like(image_np)  # column, row

    for annotation in xml_obj.annotation_list:
        if annotation.name == 'Calcification':
            if annotation.area > area_threshold:
                coordinate_list = np.array(annotation.coordinate_list) - 1
                cc, rr = polygon(coordinate_list[:, 0], coordinate_list[:, 1])
                label_np[rr, cc] = 125
            else:
                coordinate_list = np.array(annotation.coordinate_list) - 1
                cc, rr = polygon(coordinate_list[:, 0], coordinate_list[:, 1])
                # row ,column
                if len(rr) == 0:
                    for i in coordinate_list:
                        label_np[i[1], i[0]] = 255
                else:
                    label_np[rr, cc] = 255

        elif annotation.name != 'Spiculated Region':
            coordinate_list = np.array(annotation.coordinate_list) - 1
            cc, rr = polygon(coordinate_list[:, 0], coordinate_list[:, 1])
            label_np[rr, cc] = 125

    cv2.imwrite(absolute_dst_label_path, label_np)

    return


def image_with_xml2image_with_mask(absolute_src_image_path, absolute_src_xml_path, absolute_dst_image_path,
                                   absolute_dst_label_path, area_threshold):
    assert os.path.exists(absolute_src_image_path)

    shutil.copyfile(absolute_src_image_path, absolute_dst_image_path)

    if not os.path.exists(absolute_src_xml_path):
        print('This image does not have xml annotation.')
        generate_null_label(absolute_src_image_path, absolute_dst_label_path)
    else:
        print('This image has xml annotation.')
        if area_threshold == -1:
            generate_label_according_to_xml(absolute_src_image_path, absolute_src_xml_path, absolute_dst_label_path)
        else:
            generate_label_according_to_xml(absolute_src_image_path, absolute_src_xml_path, absolute_dst_label_path,
                                            area_threshold)
    return
