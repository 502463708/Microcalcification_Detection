import os

import cv2
import skimage
import numpy as np
import sys
# sys.exit(0)

import matplotlib.pyplot as plt


def show_images(images: list) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])

    plt.show(block=True)


def makedir(save_dir):
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
        for i in ['positive', 'negative']:
            os.mkdir(os.path.join(save_dir, i))
            for j in ['training', 'validation', 'test']:
                os.mkdir(os.path.join(save_dir, i, j))
                for k in ['labels', 'image']:
                    os.mkdir(os.path.join(save_dir, i, j, k))

    return save_dir


def LoadImage(folder_dir, mode='test'):
    image_dir = os.path.join(folder_dir, mode, 'image')
    label_dir = os.path.join(folder_dir, mode, 'labels')
    name_list = os.listdir(image_dir)
    image_list = list()
    label_list = list()

    for i in range(len(name_list)):
        img = cv2.imread(os.path.join(image_dir, name_list[i]), cv2.IMREAD_GRAYSCALE)
        image_list.append(img)

    for j in range(len(name_list)):
        label = cv2.imread(os.path.join(label_dir, name_list[j]), cv2.IMREAD_GRAYSCALE)
        label_list.append(label)

    return name_list, image_list, label_list, mode


def ExtractPatch(image, patch_size, stride):
    assert len(image.shape) == 2
    shape = image.shape  # H, W
    image_patch_list = list()

    for j in range(shape[1] // stride - 1):
        for i in range(shape[0] // stride - 1):
            image_patch = image[stride * i:patch_size[0] + stride * i, stride * j:patch_size[1] + stride * j]
            image_patch_list.append(image_patch)

            if i == (shape[0] // stride - 1):
                image_patch = image[shape[0] - patch_size[0]:shape[0], stride * j:patch_size[1] + stride * j]
                image_patch_list.append(image_patch)

        if j == (shape[1] // stride - 1):
            image_patch = image[stride * i:patch_size[0] + stride * i, shape[1] - patch_size[1]: shape[1]]

            image_patch_list.append(image_patch)

    return image_patch_list


def save_patch(save_dir, mode, image_patch_list, label_patch_list, image_name, threshold=10000):
    for idx in range(len(image_patch_list)):
        image_patch = image_patch_list[idx]
        label_patch = label_patch_list[idx]
        if np.sum(image_patch) >= threshold:
            if np.sum(label_patch) >= 0.01:
                dir_name = os.path.join(save_dir, 'positive', mode)
                save_name = 'positive' + str(idx) + image_name
                cv2.imwrite(os.path.join(dir_name, 'image', save_name), image_patch)
                cv2.imwrite(os.path.join(dir_name, 'labels', save_name), label_patch)

            elif np.sum(label_patch) <= 0.01:
                dir_name = os.path.join(save_dir, 'negative', mode)
                save_name = 'negative' + str(idx) + image_name
                cv2.imwrite(os.path.join(dir_name, 'image', save_name), image_patch)
                cv2.imwrite(os.path.join(dir_name, 'labels', save_name), label_patch)

    return('finish save patch')


if __name__ == '__main__':
    load_dir=r'C:\Users\75209\Desktop\Inbreat_Image_splitted_with_del'
    mysave_dir = makedir(r'C:\Users\75209\Desktop\Inbreat_patch_splitted_with_del')
    for mod in ['training', 'validation', 'test']:
        myname_list, myimage_list, mylabel_list, mymode = LoadImage(
            folder_dir=load_dir, mode=mod)
        for idx in range(len(myname_list)):
            myimage_patch_list = ExtractPatch(myimage_list[idx], patch_size=(112, 112), stride=56)
            mylabel_patch_list = ExtractPatch(mylabel_list[idx], patch_size=(112, 112), stride=56)
            save_patch(mysave_dir, mymode, myimage_patch_list, mylabel_patch_list, myname_list[idx], threshold=10000)

# test_img=cv2.imread(r'C:\Users\75209\Desktop\Inbreat-Image-splitted\test\image\test57.png',cv2.IMREAD_GRAYSCALE)
# test_label= cv2.imread(r'C:\Users\75209\Desktop\Inbreat-Image-splitted\test\labels\test57.png',cv2.IMREAD_GRAYSCALE)
# lalist=ExtractPatch(test_label,(112,112),56)
# plist=ExtractPatch(test_img,(112,112),56)
#img = cv2.imread(r'C:\Users\75209\Desktop\Inbreat-Image-splitted-patch-level\positive\test\labels\positive22test26.png',cv2.IMREAD_GRAYSCALE)
#region = measure.regionprops(img, connectivity=2)