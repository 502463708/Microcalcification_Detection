import cv2
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
from config.config_micro_calcification_image_level_classification import cfg
from dataset.dataset_micro_calcification import MicroCalcificationDataset
from metrics.metrics_image_level_classification import MetricsImageLevelClassification
from net.resnet18 import ResNet18
from torch.utils.data import DataLoader
from time import time
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.general.cuda_device_idx

cudnn.benchmark = True

model_saving_dir = 'data/lars/models/'
epoch_idx = 40
dataset = 'test'  # 'training', 'validation' or 'test'
unloader = transforms.ToPILImage()
batch_size = 48

features_blobs = []
def hook_feature(self,module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def returnCAM(self,feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def CAMProceed(net,image_np_input):
  net._modules.get('layer3').register_forward_hook(hook_feature)
  params = list(net.parameters())
  weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
  normalize = transforms.Normalize(
    mean=[0.485],
    std=[0.229]
  )
  preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    normalize
  ])
  img_pil = Image.fromarray(image_np_input.astype('uint8')).convert('L')
  img_pil.save('test.png')
  img_tensor = preprocess(img_pil)
  img_variable = Variable(img_tensor.unsqueeze(0))
  img_variable = img_variable.cuda()
  logit = net(img_variable)
  h_x = F.softmax(logit, dim=1).data.squeeze()
  probs, idx = h_x.sort(0, True)
  idx = idx.cpu().numpy()
  CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
  # render the CAM and output
  img = cv2.imread('test.png')
  height, width, _ = img.shape
  heatmap = cv2.applyColorMap(cv2.resize(
    CAMs[0], (width, height)), cv2.COLORMAP_JET)

  # Combine CAM and original map
  result = heatmap * 0.3 + img * 0.5
  return result


if __name__ == '__main__':
    start_time_for_epoch = time()

    prediction_saving_dir = os.path.join(
        model_saving_dir, 'results_dataset_{}_epoch_{}'.format(dataset, epoch_idx))
    TPs_saving_dir = os.path.join(prediction_saving_dir, 'TPs')
    TNs_saving_dir = os.path.join(prediction_saving_dir, 'TNs')
    FPs_saving_dir = os.path.join(prediction_saving_dir, 'FPs')
    FNs_saving_dir = os.path.join(prediction_saving_dir, 'FNs')

    if not os.path.exists(prediction_saving_dir):
        os.mkdir(prediction_saving_dir)
        os.mkdir(TPs_saving_dir)
        os.mkdir(TNs_saving_dir)
        os.mkdir(FPs_saving_dir)
        os.mkdir(FNs_saving_dir)

    # define the network
    net = ResNet18(in_channels=cfg.net.in_channels,
                   num_classes=cfg.net.num_classes)

    # load the specified ckpt
    ckpt_dir = os.path.join(model_saving_dir, 'ckpt')
    ckpt_path = os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(epoch_idx))

    net = net.cuda()
    net.load_state_dict(torch.load(ckpt_path))
    net = net.eval()
    print('Load ckpt: {0}...'.format(ckpt_path))

    # create dataset and data loader
    dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                        mode=dataset,
                                        enable_random_sampling=False,
                                        pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                        image_channels=cfg.dataset.image_channels,
                                        cropping_size=cfg.dataset.cropping_size,
                                        dilation_radius=0,
                                        enable_data_augmentation=False)

    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    metrics = MetricsImageLevelClassification(cfg.dataset.cropping_size)

    TPs_epoch_level = 0
    TNs_epoch_level = 0
    FPs_epoch_level = 0
    FNs_epoch_level = 0

    for batch_idx, (images_tensor, pixel_level_labels_tensor, _, image_level_labels_tensor, filenames) in enumerate(
            data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # reshape the label to meet the requirement of CrossEntropy
        # [B, C] -> [B]
        image_level_labels_tensor = image_level_labels_tensor.view(-1)
        # network forward
        preds_tensor = net(images_tensor)

        # evaluation
        _, classification_flag_np, TPs_batch_level, TNs_batch_level, FPs_batch_level, FNs_batch_level = \
            metrics.metric_batch_level(preds_tensor, image_level_labels_tensor)

        TPs_epoch_level += TPs_batch_level
        TNs_epoch_level += TNs_batch_level
        FPs_epoch_level += FPs_batch_level
        FNs_epoch_level += FNs_batch_level

        # print logging information
        print('The number of the TPs of this batch = {}'.format(TPs_batch_level))
        print('The number of the TNs of this batch = {}'.format(TNs_batch_level))
        print('The number of the FPs of this batch = {}'.format(FPs_batch_level))
        print('The number of the FNs of this batch = {}'.format(FNs_batch_level))
        print('batch: {}, consuming time: {:.4f}s'.format(
            batch_idx, time() - start_time_for_batch))
        print('-------------------------------------------------------------------------------------------------------')

        images_np = images_tensor.cpu().numpy()
        pixel_level_labels_np = pixel_level_labels_tensor.numpy()
        for patch_idx in range(images_tensor.shape[0]):
            image_np = images_np[patch_idx, 0, :, :]
            pixel_level_label_np = pixel_level_labels_np[patch_idx, :, :]
            filename = filenames[patch_idx]
            classification_flag = classification_flag_np[patch_idx]

            assert image_np.shape == pixel_level_label_np.shape
            assert len(image_np.shape) == 2

            image_np *= 255
            image_np = image_np.astype(np.uint8)

            pixel_level_label_np *= 255
            pixel_level_label_np = pixel_level_label_np.astype(np.uint8)

            flag_2_dir_mapping = {0: 'TPs', 1: 'TNs', 2: 'FPs', 3: 'FNs'}
            saving_dir_of_this_patch = os.path.join(
                prediction_saving_dir, flag_2_dir_mapping[classification_flag])

            #Function To generate cam
            if cfg.CAM:
              result = CAMProceed(net,image_np)

            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_image.png')),
                        image_np)
            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_pixel_level_label.png')),
                        pixel_level_label_np)
            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_cam.png')),
                        result)

            # print logging information
            print('The number of the TPs of this epoch = {}'.format(TPs_epoch_level))
            print('The number of the TNs of this epoch = {}'.format(TNs_epoch_level))
            print('The number of the FPs of this epoch = {}'.format(FPs_epoch_level))
            print('The number of the FNs of this epoch = {}'.format(FNs_epoch_level))
            print('consuming time: {:.4f}s'.format(
                time() - start_time_for_epoch))
            print('---------------------------------------------------------------------------------------------------')
