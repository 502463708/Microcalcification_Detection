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

    # mask calcification on label images
    for annotation in xml_obj.annotation_list:
        cal_count = 0
        mass_count = 0
        if annotation.name == 'Calcification':
            cal_count += 1
            coordinate_list = np.array(annotation.coordinate_list) - 1
            cc, rr = polygon(coordinate_list[:, 0], coordinate_list[:, 1])
            if len(rr) == 0:
                for i in coordinate_list:
                    label_np[i[1], i[0]] = 255
            else:
                label_np[rr, cc] = 255  # row ,column

    # mask mass on label images
    for annotation in xml_obj.annotation_list:
        if annotation.name != 'Spiculated Region' and annotation.name != 'Calcification':
            mass_count += 1
            coordinate_list = np.array(annotation.coordinate_list) - 1
            cc, rr = polygon(coordinate_list[:, 0], coordinate_list[:, 1])
            label_np[rr, cc] = 125

    #
    region = measure.label(input=label_np, connectivity=2)
    props = measure.regionprops(region)
    out_cal = 0
    for prop in props:
        if prop.equivalent_diameter >= diameter_threshold:
            out_cal += 1
            crds = prop.coords
            for crd in crds:
                hd = crd[0]
                wd = crd[1]
                label_np[hd][wd] = 125

    cv2.imwrite(absolute_dst_label_path, label_np)
    print('-------------------------------------------------------------------------------------------------------')
    print('On xml file, there are {} Calcifications and {} Mass {}'.format(cal_count, mass_count))
    print('after filted {} calcifications, there are {} calcifications and {} mass'.format(out_cal, cal_count - out_cal,
                                                                                           mass_count + out_cal))

    return


def image_with_xml2image_with_mask(absolute_src_image_path, absolute_src_xml_path, absolute_dst_image_path,
                                   absolute_dst_label_path, diameter_threshold):
    assert os.path.exists(absolute_src_image_path)

    shutil.copyfile(absolute_src_image_path, absolute_dst_image_path)

    if not os.path.exists(absolute_src_xml_path):
        print('This image does not have xml annotation.')
        generate_null_label(absolute_src_image_path, absolute_dst_label_path)
    else:
        print('This image has xml annotation.')
        if diameter_threshold == -1:
            generate_label_according_to_xml(absolute_src_image_path, absolute_src_xml_path, absolute_dst_label_path)
        else:
            generate_label_according_to_xml(absolute_src_image_path, absolute_src_xml_path, absolute_dst_label_path,
                                            diameter_threshold)
    return
