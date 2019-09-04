import os
import torch

from net.resnet18 import ResNet18

kMega = 1e6
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def TestResnet18(batch_size, num_classes, in_channels, dim_x, dim_y):
    assert batch_size > 0
    assert num_classes > 0

    model = ResNet18(num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / kMega))

    input_tensor = torch.zeros([batch_size, in_channels, dim_y, dim_x])
    input_tensor = input_tensor.cuda()

    output_tensor = model(input_tensor)
    assert output_tensor.shape[0] == batch_size
    assert output_tensor.shape[1] == num_classes

    print("input shape = ", input_tensor.shape)
    print("output shape = ", output_tensor.shape)


if __name__ == '__main__':
    run_time = 10
    batch_size = 2
    num_classes = 2
    in_channels = 1
    dim_x = 112
    dim_y = 112

    for idx in range(run_time):
        TestResnet18(batch_size,
                     num_classes,
                     in_channels,
                     dim_x,
                     dim_y)
