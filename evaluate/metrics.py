"""
Code From https://github.com/teboli/gan_metrics
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms

import numpy as np
import math

from . import losses
from . import networks

import pdb

class InceptionScore(nn.Module):
    def __init__(self, state_dict=None, batch_size=32, splits=10):
        super(InceptionScore, self).__init__()
        self.batch_size = batch_size
        self.splits = splits
        self.name = 'inception'
        self.model = models.inception_v3(pretrained=True)  # model pretrained on ImageNet
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def forward(self, input):
        # Inception works only with 3x299x299 images. You can crop (like center crop or 10 crops)
        # the images or rescale them before feeding them into Inception.
        assert(input.shape[1] == 3 and input.shape[2] == 299 and input.shape[3] == 299)
        self.model.eval()

        n = input.shape[0]
        n_batches = int(math.ceil(float(n) / float(self.batch_size)))

        probs = []
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.batch_size
                end = min((i+1) * self.batch_size, n)
                input_split = input[start:end]
                probs.append(self.model(input_split).softmax(dim=1))
        probs = torch.cat(probs, dim=0)

        scores = []
        for i in range(self.splits):
            start = i * n // self.splits
            end = (i+1) * n // self.splits
            probs_split = probs[start:end]
            p = probs_split.mean(dim=0,keepdim=True).log()
            # kl = F.kl_div(probs_split, p, reduction='none')
            kl = probs_split*(probs_split.log() - p)
            kl = kl.sum(dim=1).mean()
            scores.append(kl.exp().item())

        return np.mean(scores), np.std(scores)


class FCNScore(nn.Module):
    def __init__(self, model=None, batch_size=10, num_classes=21):
        super(FCNScore, self).__init__()
        self.model = model
        self.name = 'fcn'
        self.batch_size = batch_size
        self.num_classes = num_classes

    def forward(self, input, target):
        self.model.eval()

        n = input.shape[0]
        n_batches = int(math.ceil(float(n) / float(self.batch_size)))

        labels = []
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.batch_size
                end   = min((i+1) * self.batch_size, n)
                input_split = input[start:end]
                labels.append(self.model(input_split).softmax(dim=1).argmax(dim=1))
        labels = torch.cat(labels, dim=0)

        return losses.label_score(labels,target,self.num_classes)


if __name__ == '__main__':
    # Toy example
    a = torch.rand(100,3,299,299)
    b = torch.rand(100,3,500,375)
    c = torch.randint(low=0,high=21,size=(100,500,375))

    # For a classical RGB image, don't forget to center the data with those means and stds.
    # Please use these mean and std values for pretrained Inception Score and FCN Score.
    # !!! Normalize works for 3*H*W images !!!
    # a = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(a)
    # b = transforms.Normalize(mean=[0.485,0.456,0.406], std=[1.,1.,1.])(b)

    fcn_criterion = FCNScore()
    inception_criterion  = InceptionScore()

    print(inception_criterion(a))
    print(fcn_criterion(b,c))
