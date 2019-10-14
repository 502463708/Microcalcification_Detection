import xml.etree.ElementTree as ET


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


image_obj = ImageLevelAnnotationCollection('/data/lars/data/Inbreast-raw-data-with-XML-annotations/AllXML/20587612.xml')

for annotation in image_obj.annotation_list:
    print(annotation.name)
    print(annotation.area)
    print(annotation.pixel_number)
    print(annotation.coordinate_list)
