import os
import random
import shutil

def LoadPatchDataset(path_root_dir, mode='training'):
    positive_dir = os.path.join(path_root_dir, 'positive_patches')
    mode_positive_dir = os.path.join(positive_dir, mode)
    mode_positive_image_dir = os.path.join(mode_positive_dir, 'images')
    mode_negative_image_dir = mode_positive_image_dir.replace('positive', 'negative')
    return mode_positive_image_dir, mode_negative_image_dir


def RandomChoosePatches(postive_dir, negative_dir):
    positive_list = os.listdir(postive_dir)
    negative_list = os.listdir(negative_dir)
    postive_num = len(positive_list)
    chose_negative_list = random.choices(negative_list, k=postive_num)

    return positive_list, chose_negative_list


def SaveEqualPatch(data_root_dir, dst_root_dir, mode='training'):
    positive_image_dir, negative_image_dir = LoadPatchDataset(data_root_dir, mode=mode)
    positive_list, chose_negative_list = RandomChoosePatches(positive_image_dir, negative_image_dir)
    for file in positive_list:
        file_image_dir=os.path.join(positive_image_dir,file)
        dst_file_image_dir=file_image_dir.replace(data_root_dir,dst_root_dir)
        shutil.copy(file_image_dir,dst_file_image_dir)
        file_label_dir=file_image_dir.replace('images','labels')
        dst_file_label_dir=dst_file_image_dir.replace('images','labels')
        shutil.copy(file_label_dir,dst_file_label_dir)
    for file in chose_negative_list:
        file_image_dir=os.path.join(negative_image_dir,file)
        dst_file_image_dir=file_image_dir.replace(data_root_dir,dst_root_dir)
        shutil.copy(file_image_dir,dst_file_image_dir)
        file_label_dir=file_image_dir.replace('images','labels')
        dst_file_label_dir=dst_file_image_dir.replace('images','labels')
        shutil.copy(file_label_dir,dst_file_label_dir)

