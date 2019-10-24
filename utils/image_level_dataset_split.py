import cv2
import os

from sklearn.model_selection import train_test_split


def image_filename_list_split(image_dir, training_ratio, validation_ratio, test_ratio, random_seed):
    image_list = os.listdir(image_dir)
    assert len(image_list) > 0

    image_list_training_and_val, image_list_test, _, _ = train_test_split(image_list, image_list,
                                                                          test_size=test_ratio,
                                                                          random_state=random_seed if random_seed >= 0 else None)

    image_list_training, image_list_val, _, _ = train_test_split(image_list_training_and_val,
                                                                 image_list_training_and_val,
                                                                 test_size=validation_ratio / (
                                                                         validation_ratio + training_ratio),
                                                                 random_state=random_seed if random_seed >= 0 else None)

    print('***********************************************************************************************************')
    print('Training set contains {} images.'.format(len(image_list_training)))
    print('Validation set contains {} images.'.format(len(image_list_val)))
    print('Test set contains {} images.'.format(len(image_list_test)))
    print('***********************************************************************************************************')

    return image_list_training, image_list_val, image_list_test


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


def crop_and_save_data(filename_list, image_dir, label_dir, save_path, dataset_type):
    assert len(filename_list) > 0

    for filename in filename_list:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)

        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image_np, label_np = crop_image(image_np, label_np, threshold=5000)
        assert image_np.shape == label_np.shape

        dst_data_dir = os.path.join(save_path, dataset_type)

        cv2.imwrite(os.path.join(dst_data_dir, 'images', filename), image_np)
        cv2.imwrite(os.path.join(dst_data_dir, 'labels', filename), label_np)

    return


def crop_process(imgdir, labdir, crop_size=10):
    image_list = os.listdir(imgdir)
    for i in image_list:
        img = cv2.imread(os.path.join(imgdir, i), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(labdir, i), cv2.IMREAD_GRAYSCALE)
        img, label = crop_image(img, label, crop_size)
        assert img.shape == label.shape
        cv2.imwrite(os.path.join(imgdir, i), img)
        cv2.imwrite(os.path.join(labdir, i), label)
    return
