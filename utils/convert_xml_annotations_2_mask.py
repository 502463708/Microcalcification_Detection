import xml.etree.ElementTree as ET
import cv2
import numpy as np
from skimage.draw import polygon
import os


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


def draw_label(xml_name, img_name, xml_dir, image_dir, label_dir):
    img = cv2.imread(os.path.join(image_dir, img_name), cv2.IMREAD_GRAYSCALE)
    xml_obj = ImageLevelAnnotationCollection(os.path.join(xml_dir, xml_name))
    label = np.zeros_like(img)  # column, row
    print(xml_name)
    for annotation in xml_obj.annotation_list:
        if annotation.name == 'Calcification':
            cal_list = np.array(annotation.coordinate_list) - 1  # row ,coumn
            for i in cal_list:
                try:
                    label[i[1], i[0]] = 255
                except:
                    print('wrong cor at cal in file{}'.format(xml_name))
                    label[i[0], i[1]] = 255

        elif annotation.name == 'Mass':
            other_list = np.array(annotation.coordinate_list) - 1
            try:
                cc, rr = polygon(other_list[:, 0], other_list[:, 1])
                label[rr, cc] = 100
            except:
                print('wrong cor at mass  in file{}'.format(xml_name))
                cc, rr = polygon(other_list[:, 0], other_list[:, 1])
                label[cc, rr] = 100
    cv2.imwrite(os.path.join(label_dir, img_name), label)

    return 'successfully save label'
