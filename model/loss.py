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
