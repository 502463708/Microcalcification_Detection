from sklearn.model_selection import train_test_split
import os
import cv2
from skimage import measure

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2
large_threshold = 49 * 3.14

data_dir = r'C:\Users\75209\Desktop\Inbreast-dataset-radiograph-level\images'
label_dir = r'C:\Users\75209\Desktop\Inbreast-dataset-radiograph-level\labels'
save_path = r'C:\Users\75209\Desktop\Inbreat_Image_splitted_with_del'

image_list = os.listdir(data_dir)
image_train_and_val, image_test, label_train_and_val, label_test = train_test_split(image_list, image_list,
                                                                                    test_size=test_ratio)

image_train, image_val, label_train, label_val = train_test_split(image_train_and_val, label_train_and_val,
                                                                  test_size=validation_ratio / (
                                                                          validation_ratio + train_ratio))


def mkpath(save_path):
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
        for mode in ['training', 'validation', 'test']:
            mode_path = os.path.join(save_path, mode)
            os.mkdir(mode_path)
            os.mkdir(os.path.join(mode_path, 'image'))
            os.mkdir(os.path.join(mode_path, 'labels'))
    return save_path


mkpath(save_path)

for idx in range(len(image_train)):
    image_path = os.path.join(data_dir, image_train[idx])
    label_path = os.path.join(label_dir, image_train[idx])
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    region = measure.label(input=label, connectivity=2)
    props = measure.regionprops(region)
    for prop in props:
        if prop.area >= large_threshold:
            crds = prop.coords
            for crd in crds:
                hd = crd[0]
                wd = crd[1]
                label[hd][wd] == 0
    train_path = os.path.join(save_path, 'training')
    save_name = 'train' + str(idx) + '.png'
    cv2.imwrite(os.path.join(train_path, 'image', save_name), img)
    cv2.imwrite(os.path.join(train_path, 'labels', save_name), label)

for idx in range(len(image_val)):
    image_path = os.path.join(data_dir, image_val[idx])
    label_path = os.path.join(label_dir, image_val[idx])
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    region = measure.label(input=label, connectivity=2)
    props = measure.regionprops(region)
    for prop in props:
        if prop.area >= large_threshold:
            crds = prop.coords
            for crd in crds:
                hd = crd[0]
                wd = crd[1]
                label[hd][wd] == 0
    validation_path = os.path.join(save_path, 'validation')

    save_name = 'validation' + str(idx) + '.png'
    cv2.imwrite(os.path.join(validation_path, 'image', save_name), img)
    cv2.imwrite(os.path.join(validation_path, 'labels', save_name), label)

for idx in range(len(image_test)):
    image_path = os.path.join(data_dir, image_test[idx])
    label_path = os.path.join(label_dir, image_test[idx])
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    region = measure.label(input=label, connectivity=2)
    props = measure.regionprops(region)
    for prop in props:
        if prop.area >= large_threshold:
            crds = prop.coords
            for crd in crds:
                hd = crd[0]
                wd = crd[1]
                label[hd][wd] == 0
    test_path = os.path.join(save_path, 'test')
    save_name = 'test' + str(idx) + '.png'
    cv2.imwrite(os.path.join(test_path, 'image', save_name), img)
    cv2.imwrite(os.path.join(test_path, 'labels', save_name), label)

print('finish dataset split')
