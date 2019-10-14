import numpy as np
from skimage.util import view_as_windows
from itertools import product
from typing import Tuple
import os
import skimage.io as io
import skimage.morphology as mp
import skimage.measure as measure
import sys
from skimage import img_as_int
import cv2
import copy
import skimage.draw as skdraw

root = '/home/groupprofzli/data1/dwz/nbreast-dataset-radiograph-level/Calcification_data'
root2 = '/home/groupprofzli/data1/dwz/Inbreast-dataset-radiograph-level/'
img_dir = 'removeother'
lbl_dir = 'imglabel'
txt_dir = 'ALLTXTall'
AREA = 49 * np.pi
PATCH_SIZE = (112, 112)
STRIDE = PATCH_SIZE[0] / 2
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
THRESHOLD = 10000


def train_val_test_splits():
    imgs = [i for i in os.listdir(
        os.path.join(root2, 'images')) if '.png' in i]
    np.random.seed(0)
    np.random.shuffle(imgs)
    train_num = round(len(imgs) * TRAIN_RATIO)
    val_num = round(len(imgs) * VAL_RATIO)
    test_num = len(imgs) - train_num - val_num
    train = imgs[0: train_num]
    val = imgs[train_num:train_num + val_num]
    test = imgs[train_num + val_num:]
    all_files = {'train': train,
                 'val': val,
                 'test': test}
    return all_files


def del_large(img, label):
    lbls = mp.label(label)
    props = measure.regionprops(lbls)
    cal_center = []
    for i in props:
        if i.area > AREA:
            img[lbls == i.label] = 0
            label[lbls == i.label] = 0
        else:
            cal_center += i.centroid
    return img, cal_center


def del_large_img(img, label):
    label[img == 0] = 0
    return img, label


def cal_center_patch(label):
    lbls = mp.label(label)
    if len(np.unique(lbls)) == 1:
        return None
    props = measure.regionprops(lbls)
    cal_center = []
    for i in props:
        cal_center.append(list(i.centroid))
    return cal_center


def cal_center_patch2global(row, col, h, stride, cal_center):
    cal_center_copy = copy.deepcopy(cal_center)
    for i in range(len(cal_center_copy)):
        cal_center_copy[i][0] += row * (h - stride)
        cal_center_copy[i][1] += col * (h - stride)
    return cal_center_copy


def create_dirs(file):
    dirname = os.path.dirname(file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def del_black(patch):
    if np.sum(patch != 0) >= THRESHOLD:
        return True
    return False


def gen_label(patch, cal_center):
    h, w = patch.shape
    lbl = np.zeros((h, w), dtype=np.uint8)
    if cal_center == None:
        return lbl
    for i in cal_center:
        rr, cc = skdraw.circle(int(i[0]), int(i[1]), 7, shape=lbl.shape)
        lbl[rr, cc] = 255
    return lbl


def gen_negative_train_val_list():
    train_patch_path = os.path.join(root, 'train', 'patch')

    val_patch_path = os.path.join(root, 'val', 'patch')
    train_positive = os.path.join(root, 'train', 'txt')
    val_positive = os.path.join(root, 'val', 'txt')

    train_list = [i for i in os.listdir(train_patch_path) if '.png' in i]
    train_poslist = [i.replace('.txt', '.png')
                     for i in os.listdir(train_positive) if '.txt' in i]
    train_negalist = list(set(train_list) - set(train_poslist))
    with open(os.path.join(root, 'train', 'negative.txt'), 'w') as f:
        for i in train_negalist:
            f.write(i + '\n')

    val_list = [i for i in os.listdir(val_patch_path) if '.png' in i]
    val_poslist = [i.replace('.txt', '.png')
                   for i in os.listdir(val_positive) if '.txt' in i]
    val_negalist = list(set(val_list) - set(val_poslist))
    with open(os.path.join(root, 'val', 'negative.txt'), 'w') as f:
        for i in val_negalist:
            f.write(i + '\n')


def save_patch(patch, cal_center, cal_center_global, img_name, row, col, splits):
    suffix_idx = img_name.find('.png')
    patch_name = img_name[:suffix_idx] + \
                 '_{}_{}'.format(row, col) + img_name[suffix_idx:]
    if cal_center:
        cal_center_file = os.path.join(
            root, splits, 'txt', patch_name.replace('.png', '.txt'))
        create_dirs(cal_center_file)
        with open(cal_center_file, 'w') as f:
            for i in range(len(cal_center)):
                f.write('{} {} {} {}\n'.format(
                    cal_center[i][0], cal_center[i][1], cal_center_global[i][0], cal_center_global[i][1]))
    patch_path = os.path.join(root, splits, 'patch', patch_name)
    lbl_path = os.path.join(root, splits, 'patch_lbl', patch_name)
    create_dirs(patch_path)
    create_dirs(lbl_path)
    patch_lbl = gen_label(patch, cal_center)
    print('Saving ', patch_path)
    cv2.imwrite(patch_path, patch)
    cv2.imwrite(lbl_path, patch_lbl)


def save_img(img, lbl, name, splits):
    img_path = os.path.join(root, splits, 'img', name)
    lbl_path = os.path.join(root, splits, 'lbl', name)
    create_dirs(img_path)
    create_dirs(lbl_path)
    print('Saving ', img_path)
    cv2.imwrite(img_path, img)
    cv2.imwrite(lbl_path, lbl)


def padding(img, lbl):
    h, w = img.shape
    padsize_h = PATCH_SIZE[0] - h % PATCH_SIZE[0]
    padsize_w = PATCH_SIZE[1] - w % PATCH_SIZE[1]
    new_img = np.pad(img, ((0, padsize_h), (0, padsize_w)),
                     'constant', constant_values=0)
    new_lbl = np.pad(lbl, ((0, padsize_h), (0, padsize_w)),
                     'constant', constant_values=0)
    return new_img, new_lbl


def main():
    img_path = os.path.join(root, img_dir)
    lbl_path = os.path.join(root, lbl_dir)
    img_path2 = os.path.join(root2, 'images')
    all_files = train_val_test_splits()

    for splits, img_list in all_files.items():
        for img_name in img_list:
            # 1 channel
            if os.path.exists(os.path.join(img_path, img_name)):
                img = cv2.imread(os.path.join(img_path, img_name),
                                 cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(os.path.join(img_path2, img_name),
                                 cv2.IMREAD_GRAYSCALE)

            if os.path.exists(os.path.join(lbl_path, img_name)):
                lbl = cv2.imread(os.path.join(
                    lbl_path, img_name), cv2.IMREAD_GRAYSCALE)
            else:
                h, w = img.shape
                lbl = np.zeros((h, w), dtype=img.dtype)
            img, lbl = padding(img, lbl)
            img, lbl = del_large_img(img, lbl)
            img, cal_center_img = del_large(img, lbl)
            save_img(img, lbl, img_name, splits)

            patches_img = patchify(img, PATCH_SIZE, int(STRIDE))
            patches_lbl = patchify(lbl, PATCH_SIZE, int(STRIDE))
            # row x col x patch_h x patch w
            for row in range(patches_img.shape[0]):
                for col in range(patches_img.shape[1]):
                    patch = patches_img[row, col]
                    patch_lbl = patches_lbl[row, col]

                    cal_center = cal_center_patch(patch_lbl)
                    if cal_center:
                        cal_center_global = cal_center_patch2global(
                            row, col, PATCH_SIZE[0], STRIDE, cal_center)
                    else:
                        cal_center_global = None

                    if del_black(patch):
                        save_patch(patch, cal_center,
                                   cal_center_global, img_name, row, col, splits)


def patchify(patches: np.ndarray, patch_size: Tuple[int, int], step: int = 1):
    return view_as_windows(patches, patch_size, step)


def unpatchify(patches: np.ndarray, imsize: Tuple[int, int]):
    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    # Calculat the overlap size in each axis
    o_w = (n_w * p_w - i_w) / (n_w - 1)
    o_h = (n_h * p_h - i_h) / (n_h - 1)

    # The overlap should be integer, otherwise the patches are unable to reconstruct into a image with given shape
    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i, j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor


def merge():
    pass


def count_ratio():
    train_patch_path = os.path.join(root, 'train', 'patch')
    val_patch_path = os.path.join(root, 'val', 'patch')
    train_positive = os.path.join(root, 'train', 'txt')
    val_positive = os.path.join(root, 'val', 'txt')
    test_patch_path = os.path.join(root, 'test', 'patch')
    test_positive = os.path.join(root, 'test', 'txt')

    train_list = [i for i in os.listdir(train_patch_path) if '.png' in i]
    train_poslist = [i for i in os.listdir(train_positive) if '.txt' in i]
    count_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in train_poslist:
        f = open(os.path.join(train_positive, i), 'r').readlines()
        if len(f) == 15:
            print(len(f), i)
        if len(f) <= 10:
            count_train[len(f) - 1] += 1
    count_train.insert(0, len(train_list) - len(train_poslist))

    val_list = [i for i in os.listdir(val_patch_path) if '.png' in i]
    val_poslist = [i for i in os.listdir(val_positive) if '.txt' in i]
    count_val = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in val_poslist:
        f = open(os.path.join(val_positive, i), 'r').readlines()
        if len(f) <= 10:
            count_val[len(f) - 1] += 1
    count_val.insert(0, len(val_list) - len(val_poslist))

    test_list = [i for i in os.listdir(test_patch_path) if '.png' in i]
    test_poslist = [i for i in os.listdir(test_positive) if '.txt' in i]
    count_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in test_poslist:
        f = open(os.path.join(test_positive, i), 'r').readlines()
        if len(f) <= 10:
            count_test[len(f) - 1] += 1
    count_test.insert(0, len(test_list) - len(test_poslist))

    print("Train ratio is ", count_train)
    print('Val ration is ', count_val)
    print('Test ration is ', count_test)


if __name__ == '__main__':
    main()
    #gen_negative_train_val_list()
    # count_ratio()
