import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score, recall_score


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return precision_score(target.view(-1).cpu(), pred.view(-1).cpu())


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return recall_score(target.view(-1).cpu(), pred.view(-1).cpu())


def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1(target.view(-1).cpu(), pred.view(-1).cpu())
