import torch
import torch.nn.functional as F


def focal_loss(inputs, targets, alpha=0.5, gamma=2, reduction='mean'):
    logpt = F.cross_entropy(inputs, targets.long(), reduction='none')
    pt = torch.exp(-logpt)
    focal_loss = (1 - pt) ** gamma * logpt
    alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_weight * focal_loss

    if reduction == 'mean':
        return torch.mean(focal_loss)
    elif reduction == 'sum':
        return torch.sum(focal_loss)
    else:
        return focal_loss


def dice_loss(inputs, targets, epsilon=1e-7):
    targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=inputs.shape[1])
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
    inputs = F.softmax(inputs, dim=1)
    targets_one_hot = targets_one_hot.type(inputs.type())
    numerator = 2 * (inputs * targets_one_hot).sum(dim=(2,3))
    denominator = inputs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
    dice_coefficient = numerator / (denominator + epsilon)
    return 1 - dice_coefficient.mean()
