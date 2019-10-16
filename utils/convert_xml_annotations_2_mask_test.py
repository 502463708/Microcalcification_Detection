import cv2
import os
import numpy as np
from utils.convert_xml_annotations_2_mask import draw_label

mydir = r'C:\Users\75209\Desktop\data\Inbreast-raw-data-with-XML-annotations'
img_save_dir = r'C:\Users\75209\Desktop\data\Inbreast-raw-data-with-XML-annotations\Inbreast-png'

xml_dir = os.path.join(mydir, 'AllXML')
image_dir = os.path.join(mydir, 'Inbreast-png')
label_dir = os.path.join(mydir, 'labels-no-region')
if not os.path.isdir(label_dir):
    os.mkdir(label_dir)
xml_list = os.listdir(xml_dir)
image_list = os.listdir(image_dir)
xml_split_list = list()
image_split_list = list()

for xml in xml_list:
    xml_split = xml.split('.')
    xml_split_list.append(xml_split)

for image in image_list:
    image_split = image.split('.')
    image_split_list.append(image_split)

#generate null label images
def generate_null_label(dir,img_list,save_dir):
    for i in img_list:
        img = cv2.imread(os.path.join(dir, i), cv2.IMREAD_GRAYSCALE)
        no_label = np.zeros_like(img, dtype='uint8')
        cv2.imwrite(os.path.join(save_dir, i), no_label)

#generate_null_label(image_dir,image_list,r'C:\Users\75209\Desktop\data\Inbreast-raw-data-with-XML-annotations\null-labels')

if __name__ == '__main__':
    for xml_idx in range(len(xml_list)):
        for image_idx in range(len(image_list)):
            if xml_split_list[xml_idx][0] == image_split_list[image_idx][0]:
                draw_label(xml_list[xml_idx], image_list[image_idx], xml_dir, image_dir, label_dir)

# 22580341
# 22670094 3408 1743
#  22580706  3121 3659


# image_obj = ImageLevelAnnotationCollection(
#     r'C:\Users\75209\Desktop\data\Inbreast-raw-data-with-XML-annotations\AllXML\22580341.xml')
# label = np.zeros((3328, 2560))
# for annotation in image_obj.annotation_list:
#     print(annotation.name)
#     print(annotation.coordinate_list)
#     if annotation.name == 'Calcification':
#         cal_list = np.array(annotation.coordinate_list) - 1  # row ,coumn
#         for i in cal_list:
#             label[i[1], i[0]] = 255
#     if annotation.name != 'Calcification':
#         other_list = np.array(annotation.coordinate_list) - 1
#
#         cc, rr = polygon(other_list[:, 0], other_list[:, 1])
#         label[rr, cc] = 100
#     cv2.imwrite(r'C:\Users\75209\Desktop\data\test.png', label)
#
# max_0 = 0
# max_1 = 0
#
# for annotation in image_obj.annotation_list:
#     # print(annotation.name)
#     # print(annotation.coordinate_list)
#
#     for i in annotation.coordinate_list:
#         if i[0]>max_0:
#             max_0=i[0]
#         if i[1]>max_1:
#             max_1=i[1]
# print(max_0,max_1)


#          label[i[0],i[1]] = 255
# cv2.imwrite(r'C:\Users\75209\Desktop\data\test.png', label)
# img = cv2.imread(
#     r'C:\Users\75209\Desktop\data\Inbreast-raw-data-with-XML-annotations\Inbreast-raw-png\51048738_3f22cdda8da215e3_MG_R_ML_ANON.png',
#     cv2.IMREAD_GRAYSCALE)
# print(img.shape)
# #
# label=np.zeros_like(img)
# for annotation in image_obj.annotation_list:
#     if annotation.name == 'Calcification':
#         cal_list = np.array(annotation.coordinate_list)  # row ,column
#         for i in cal_list:
#             print(i)
#             label[i[0], i[1]] = 255
#     if annotation.name == 'Spiculated Region':

#         slist = np.array(annotation.coordinate_list)


# label = np.zeros_like(img)
# cc, rr = polygon(mlist[:, 0], mlist[:, 1])
# print(cc)
# print(rr)
# label[rr, cc] = 255
# cv2.imwrite(r'C:\Users\75209\Desktop\data\mtest.png', label)
