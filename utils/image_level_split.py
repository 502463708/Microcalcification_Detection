import cv2
import os

from skimage import measure
from sklearn.model_selection import train_test_split


# dataset split
def DatsetSplit(image_dir, train_ratio, validation_ratio, test_ratio):
    image_list = os.listdir(image_dir)
    image_train_and_val, image_test, label_train_and_val, label_test = train_test_split(image_list, image_list,
                                                                                        test_size=test_ratio)

    image_train, image_val, label_train, label_val = train_test_split(image_train_and_val, label_train_and_val,
                                                                      test_size=validation_ratio / (
                                                                              validation_ratio + train_ratio))
    return image_train, image_val, image_test


def crop_image(image, label, crop_size=500, threshold=1500):
    assert image.shape == label.shape
    for i in range(crop_size):
        if image[0, :].sum() < threshold:
            image = image[1:, :]
            label = label[1:, :]
        if image[-1, :].sum() < threshold:
            image = image[:image.shape[0] - 1, :]
            label = label[:label.shape[0] - 1, :]
        if image[:, 0].sum() < threshold:
            image = image[:, 1:]
            label = label[:, 1:]
        if image[:, -1].sum() < threshold:
            image = image[:, :image.shape[1] - 1]
            label = label[:, :label.shape[1] - 1]
    return image, label


def saveimg(name_list, data_dir, label_dir, save_path, large_threshold, mode='training'):
    for idx in range(len(name_list)):
        image_path = os.path.join(data_dir, name_list[idx])
        label_path = os.path.join(label_dir, name_list[idx])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        img, label = crop_image(img, label, threshold=5000)
        region = measure.label(input=label, connectivity=2)
        props = measure.regionprops(region)
        for prop in props:
            if prop.area >= large_threshold:
                crds = prop.coords
                for crd in crds:
                    hd = crd[0]
                    wd = crd[1]
                    label[hd][wd] == 0
        mode_path = os.path.join(save_path, mode)
        save_name = name_list[idx]
        assert img.shape == label.shape
        cv2.imwrite(os.path.join(mode_path, 'image', save_name), img)
        cv2.imwrite(os.path.join(mode_path, 'labels', save_name), label)

    return 'finished save {}'.format(mode)


def crop_process(imgdir, labdir, crop_size=10):
    image_list = os.listdir(imgdir)
    for i in image_list:
        img = cv2.imread(os.path.join(imgdir, i), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(labdir, i), cv2.IMREAD_GRAYSCALE)
        img, label = crop_image(img, label, crop_size)
        assert img.shape == label.shape
        cv2.imwrite(os.path.join(imgdir, i), img)
        cv2.imwrite(os.path.join(labdir, i), label)
    return 'finish cropped'



