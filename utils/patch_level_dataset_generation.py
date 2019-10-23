import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def show_images(images: list) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])

    plt.show(block=True)


def makedir(save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        for i in ['positive_patches', 'negative_patches']:
            os.mkdir(os.path.join(save_dir, i))
            for j in ['training', 'validation', 'test']:
                os.mkdir(os.path.join(save_dir, i, j))
                for k in ['labels', 'images']:
                    os.mkdir(os.path.join(save_dir, i, j, k))

    return save_dir


def load_image(folder_dir, mode='test'):
    image_dir = os.path.join(folder_dir, mode, 'images')
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


def extract_patch(image, patch_size, stride):
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


def save_patch(save_dir, mode, image_patch_list, label_patch_list, image_name, pixel_threshold=1,
               area_threshold=112 * 112 * 0.5):
    print('-------------------------------------------------------------------------------------------------------')
    print('Processing on {}'.format(image_name))
    pos_patch_count = 0
    neg_patch_count = 0
    for idx in range(len(image_patch_list)):
        image_patch = image_patch_list[idx]
        label_patch = label_patch_list[idx]
        if np.where(label_patch == 125)[0].shape[0] != 0 or np.sum(image_patch < pixel_threshold) >= area_threshold:
            continue
        elif np.sum(label_patch == 255) >= 0.01:
            pos_patch_count += 1
            dir_name = os.path.join(save_dir, 'positive_patches', mode)
            save_name = 'positive' + str(idx) + '_' + image_name
            cv2.imwrite(os.path.join(dir_name, 'images', save_name), image_patch)
            cv2.imwrite(os.path.join(dir_name, 'labels', save_name), label_patch)
        elif np.sum(label_patch) <= 0.01:
            neg_patch_count += 1
            dir_name = os.path.join(save_dir, 'negative_patches', mode)
            save_name = 'negative' + str(idx) + '_' + image_name
            cv2.imwrite(os.path.join(dir_name, 'images', save_name), image_patch)
            cv2.imwrite(os.path.join(dir_name, 'labels', save_name), label_patch)
    print('finish saving {} positive patches, {} negative patches. \n Totally {} patches have be saved '.format(
        pos_patch_count, neg_patch_count, pos_patch_count + neg_patch_count))

    return ('finish save patch')
