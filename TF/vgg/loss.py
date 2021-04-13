import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dice_score = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return torch.tensor([1. - dice_score], requires_grad=True)
