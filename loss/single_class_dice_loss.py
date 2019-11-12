import torch.nn as nn
import torch.nn.functional as F


class SingleClassDiceLoss(nn.Module):
    def __init__(self):
        super(SingleClassDiceLoss, self).__init__()

        return

    def forward(self, predictions, targets, activate=False):
        assert len(predictions.shape) == 4
        assert len(targets.shape) == 3
        assert predictions.shape[1] == 1

        batch_size = targets.shape[0]
        epsilon = 1

        if targets.device.type != 'cuda':
            targets = targets.cuda()

        if activate:
            predictions = F.sigmoid(predictions)

        predictions = predictions.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = (predictions * targets).sum(1)
        union = predictions.sum(1) + targets.sum(1)

        score = 2. * (intersection + epsilon) / (union + epsilon)
        score = 1 - score.sum() / batch_size

        return score
