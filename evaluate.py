import os
import torch
from .eval import metrics
from models import create_model

from .options.eval_options import EvalOptions

preprocessing = {
    "inception":
}

if __name__ == "__main__":
    opt = EvalOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    if opt.metric=='inception':
        metrics = [metrics.InceptionScore()]
    elif opt.metric=='fcn':
        metrics = [metrics.FCNScore()]
    elif opt.metric=='all':
        metrics = [metrics.InceptionScore(), metrics.FCNScore()]

    for metric in metrics:
        metric_scores = np.zeros(len(dataset))
        for i,data in enumerate(data_loader):
            metric_scores[i] = 0






