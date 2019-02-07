"""
Code From https://github.com/teboli/gan_metrics
"""
import torch
import torch.nn.functional as F

import numpy as np

def _fast_hist(input, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    hist = num_classes * target[mask].astype(int) + input[mask].astype(int)
    hist = np.bincount(hist, minlength=num_classes**2).reshape(num_classes, num_classes)
    return hist


def label_score(input, target, num_classes):
    input  = input.cpu().numpy()
    target = target.cpu().numpy()
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(input, target):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    per_pixel_acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    per_class_acc = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    iou_acc = np.nanmean(iu)
    return per_pixel_acc, per_class_acc, iou_acc


if __name__ == "__main__":
    input  = torch.randint(low=0,high=21,size=(10,500,375))
    target = torch.randint(low=0,high=21,size=(10,500,375))
    num_classes = 21

    print(label_score(input, target, num_classes))
