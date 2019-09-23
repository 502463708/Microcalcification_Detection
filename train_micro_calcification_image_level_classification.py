import importlib
import numpy as np
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
import visdom

from config.config_micro_calcification_image_level_classification import cfg
from dataset.dataset_micro_calcification import MicroCalcificationDataset
from metrics.metrics_image_level_classification import MetricsImageLevelClassification
from logger.logger import Logger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from time import time

# the environment related global variables are specified here
#
# specify the GPUs to be occupied
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.general.cuda_device_idx
cudnn.benchmark = True


def iterate_for_an_epoch(training, epoch_idx, data_loader, net, loss_func, metrics, visdom_obj, logger=None,
                         optimizer=None):
    # training == True -> training mode: backward gradients
    # training == False -> evaluation mode: do not backward gradients
    assert isinstance(training, bool)
    assert epoch_idx >= 0

    if training:
        assert optimizer is not None
        net = net.train()
        if logger is not None:
            logger.write('--------------------------------------------------------------------------------------------')
            logger.write('start training epoch: {}'.format(epoch_idx))
    else:
        net = net.eval()
        if logger is not None:
            logger.write('--------------------------------------------------------------------------------------------')
            logger.write('start evaluating epoch: {}'.format(epoch_idx))

    # this variable is created for recording loss of each batch
    loss_for_each_batch_list = list()

    # these variable is created for recording the annotated calcifications,
    # recalled calcifications and false positive calcifications
    TPs_epoch_level = 0
    TNs_epoch_level = 0
    FPs_epoch_level = 0
    FNs_epoch_level = 0

    # start time of this epoch
    start_time_for_epoch = time()

    # iterating through each batch
    for batch_idx, (images_tensor, pixel_level_labels_tensor, _, image_level_labels_tensor, _) in enumerate(
            data_loader):

        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()
        image_level_labels_tensor = image_level_labels_tensor.cuda()

        # reshape the label to meet the requirement of CrossEntropy
        image_level_labels_tensor = image_level_labels_tensor.view(-1)  # [B, C] -> [B]

        # network forward
        preds_tensor = net(images_tensor)  # the shape of preds_tensor: [B, 2]

        # calculate loss of this batch
        loss = loss_func(preds_tensor, image_level_labels_tensor)

        loss_for_each_batch_list.append(loss.item())

        # backward gradients only when training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics
        image_level_masks_np, _, TPs_batch_level, TNs_batch_level, FPs_batch_level, FNs_batch_level = \
            metrics.metric_batch_level(preds_tensor, image_level_labels_tensor)
        TPs_epoch_level += TPs_batch_level
        TNs_epoch_level += TNs_batch_level
        FPs_epoch_level += FPs_batch_level
        FNs_epoch_level += FNs_batch_level

        # print logging information
        if logger is not None:
            logger.write('epoch: {}, batch: {}, loss: {:.4f}, consuming time: {:.4f}s'
                         .format(epoch_idx, batch_idx, loss.item(), time() - start_time_for_batch))

        # update images display in visdom
        if batch_idx % cfg.visdom.update_batches == 0:
            try:
                visdom_obj.images(
                    images_tensor,
                    win='I{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='I{}'.format('T' if training else 'V'))
                )
                visdom_obj.images(
                    np.expand_dims(image_level_masks_np, axis=1),
                    win='ILM{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='ILM{}'.format('T' if training else 'V'))
                )
                visdom_obj.images(
                    pixel_level_labels_tensor.unsqueeze(dim=1),
                    win='PL{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='PL{}'.format('T' if training else 'V'))
                )
            except BaseException as err:
                print('Error message: ', err)

    # calculate loss of this epoch
    average_loss_of_this_epoch = np.array(loss_for_each_batch_list).mean()

    # calculate accuracy of this epoch
    accuracy_of_this_epoch = (TPs_epoch_level + TNs_epoch_level) / (
            TPs_epoch_level + TNs_epoch_level + FPs_epoch_level + FNs_epoch_level)

    # record metric on validation set for determining the best model to be saved
    if not training:
        metrics.determine_saving_metric_on_validation_list.append(accuracy_of_this_epoch)

    if logger is not None:
        logger.write('{} of epoch {} finished'.format('training' if training else 'evaluating', epoch_idx))
        logger.write('epoch: {}, loss: {:.4f}, consuming time: {:.4f}s'.format(epoch_idx,
                                                                               average_loss_of_this_epoch,
                                                                               time() - start_time_for_epoch))
        logger.write('--------------------------------------------------------------------------------------------')

    # update loss of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([average_loss_of_this_epoch]),
                    win='loss',
                    update='append',
                    name='{}_loss'.format('training' if training else 'validation'),
                    opts=dict(title='loss'))

    # update accuracy of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([accuracy_of_this_epoch]),
                    win='accuracy',
                    update='append',
                    name='{}_accuracy'.format('training' if training else 'validation'),
                    opts=dict(title='accuracy'))

    # update TPs of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([TPs_epoch_level]),
                    win='metrics_{}'.format('training' if training else 'validation'),
                    update='append',
                    name='TPs',
                    opts=dict(title='metrics_{}'.format('training' if training else 'validation')))

    # update TNs of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([TNs_epoch_level]),
                    win='metrics_{}'.format('training' if training else 'validation'),
                    update='append',
                    name='TNS',
                    opts=dict(title='metrics_{}'.format('training' if training else 'validation')))

    # update FPs of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([FPs_epoch_level]),
                    win='metrics_{}'.format('training' if training else 'validation'),
                    update='append',
                    name='FPs',
                    opts=dict(title='metrics_{}'.format('training' if training else 'validation')))

    # update FNs of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([FNs_epoch_level]),
                    win='metrics_{}'.format('training' if training else 'validation'),
                    update='append',
                    name='FNs',
                    opts=dict(title='metrics_{}'.format('training' if training else 'validation')))

    return


if __name__ == '__main__':
    # create a folder for saving purpose
    ckpt_dir = os.path.join(cfg.general.saving_dir, 'ckpt')
    if not os.path.exists(cfg.general.saving_dir):
        os.makedirs(cfg.general.saving_dir)
        os.makedirs(ckpt_dir)

        # copy related config and net .py file to the saving dir
        shutil.copyfile('./config/config_micro_calcification_image_level_classification.py',
                        os.path.join(cfg.general.saving_dir,
                                     'config_micro_calcification_image_level_classification.py'))
        shutil.copyfile('./net/{0}.py'.format(cfg.net.name),
                        os.path.join(cfg.general.saving_dir, '{0}.py'.format(cfg.net.name)))

    # initialize logger
    logger = Logger(cfg.general.saving_dir)

    # import the network package
    try:
        net_package = importlib.import_module('net.{}'.format(cfg.net.name))
    except BaseException:
        print('failed to import package: {}'.format('net.' + cfg.net.name))
    #
    # define the network
    net = net_package.ResNet18(in_channels=cfg.net.in_channels, num_classes=cfg.net.num_classes)

    # check whether the ckpt dir is empty
    ckpt_file_list = os.listdir(ckpt_dir)
    if len(ckpt_file_list) == 0:
        net = torch.nn.DataParallel(net).cuda()
        net_package.ApplyKaimingInit(net)
        logger.write('Training from scratch...')
    else:
        # find the latest saved ckpt
        latest_ckpt_idx = np.array([int(ckpt_file.split('_')[2].split('.')[0]) for ckpt_file in ckpt_file_list]).max()
        latest_ckpt_file = 'net_epoch_{0}.pth'.format(latest_ckpt_idx)
        checkpoint_path = os.path.join(ckpt_dir, latest_ckpt_file)

        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(checkpoint_path))
        logger.write('Load ckpt: {0}...'.format(latest_ckpt_file))

    # setup metrics
    metrics = MetricsImageLevelClassification(image_size=cfg.dataset.cropping_size)

    # setup Visualizer
    visdom_display_name = cfg.general.saving_dir.split('/')[-2]
    visdom_obj = visdom.Visdom(env=visdom_display_name, port=cfg.visdom.port)

    # create dataset and data loader for training
    training_dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                                 mode='training',
                                                 enable_random_sampling=True,
                                                 pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                                 image_channels=cfg.dataset.image_channels,
                                                 cropping_size=cfg.dataset.cropping_size,
                                                 dilation_radius=cfg.dataset.dilation_radius,
                                                 enable_data_augmentation=cfg.dataset.augmentation.enable_data_augmentation,
                                                 enable_vertical_flip=cfg.dataset.augmentation.enable_vertical_flip,
                                                 enable_horizontal_flip=cfg.dataset.augmentation.enable_horizontal_flip)

    training_data_loader = DataLoader(training_dataset, batch_size=cfg.train.batch_size,
                                      shuffle=True, num_workers=cfg.train.num_threads)

    # create dataset and data loader for validation
    validation_dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                                   mode='validation',
                                                   enable_random_sampling=False,
                                                   pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                                   image_channels=cfg.dataset.image_channels,
                                                   cropping_size=cfg.dataset.cropping_size,
                                                   dilation_radius=cfg.dataset.dilation_radius,
                                                   enable_data_augmentation=False)

    validation_data_loader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size,
                                        shuffle=True, num_workers=cfg.train.num_threads)

    # define loss function
    assert cfg.loss.name in ['CrossEntropyLoss']
    if cfg.loss.name == 'CrossEntropyLoss':
        loss_func = CrossEntropyLoss()

    # setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr_scheduler.lr)

    # learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler.step_size,
                                                   gamma=cfg.lr_scheduler.gamma)

    # iterating through each epoch
    for epoch_idx in range(0, cfg.train.num_epochs):
        iterate_for_an_epoch(training=True,
                             epoch_idx=epoch_idx,
                             data_loader=training_data_loader,
                             net=net,
                             loss_func=loss_func,
                             metrics=metrics,
                             visdom_obj=visdom_obj,
                             logger=logger,
                             optimizer=optimizer)

        iterate_for_an_epoch(training=False,
                             epoch_idx=epoch_idx,
                             data_loader=validation_data_loader,
                             net=net,
                             loss_func=loss_func,
                             metrics=metrics,
                             visdom_obj=visdom_obj,
                             logger=logger)
        lr_scheduler.step()

        logger.flush()

        # whether to save this model according to config
        if epoch_idx % cfg.train.save_epochs is 0:
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(epoch_idx)))

        # save this model in case that this is the currently best model on validation set
        is_best_ckpt_on_validation_set = metrics.determine_saving_metric_on_validation_list[-1] == max(
            metrics.determine_saving_metric_on_validation_list)
        if is_best_ckpt_on_validation_set:
            saved_ckpt_list = os.listdir(ckpt_dir)
            # remove the last saved best model
            for saved_ckpt_filename in saved_ckpt_list:
                if 'net_best_on_validation_set' in saved_ckpt_filename:
                    os.remove(os.path.join(ckpt_dir, saved_ckpt_filename))
            # save the current best model
            torch.save(net.state_dict(),
                       os.path.join(ckpt_dir, 'net_best_on_validation_set_epoch_{}.pth'.format(epoch_idx)))
