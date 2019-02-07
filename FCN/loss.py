import torch
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    input = input.type(torch.cuda.FloatTensor)
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1,2).transpose(2,3).contiguous()
    log_p = log_p[target.view(n,h,w,1).repeat(1,1,1,c)>=0].view(-1,c)
    # target: (n*h*w,)
    mask = (target >= 0)
    target = target[mask]
    loss = F.nll_loss(log_p, target.type(torch.cuda.LongTensor), weight=weight, reduction='sum')
    if size_average:
        loss /= mask.type(torch.cuda.FloatTensor).data.sum()
    return loss

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets.type(torch.cuda.LongTensor))