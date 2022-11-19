import torch
import torch.nn.functional as F


def focal_loss(inputs, targets):
    gamma, weights = 2, torch.Tensor([10, 10]).to(targets.device)
    return F.nll_loss(
        (1 - F.softmax(inputs, dim=1)) ** gamma * F.log_softmax(inputs, dim=1),
        targets.long(),
        weight=weights
    )
