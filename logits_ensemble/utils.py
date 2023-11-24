import os
import json
import random

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import r2_score
from scipy.stats import kendalltau, spearmanr, rankdata


class Option(dict):
    def __init__(self, *args, **kwargs):
        args = [arg if isinstance(arg, dict) else json.loads(open(arg).read())
                for arg in args]
        super(Option, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = Option(v)
                    else:
                        self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = Option(v)
                else:
                    self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Option, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Option, self).__delitem__(key)
        del self.__dict__[key]


def seed_everything(seed: int = 29):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
        
        
def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result


def evaluation(out, labels, masks, rank_full=True):
    
    loss = F.mse_loss(out[masks["test"]],
                      labels[masks["test"]].reshape(-1, 1)).item()
    mae_loss = F.l1_loss(out[masks["test"]],
                         labels[masks["test"]].reshape(-1, 1)).item()
    
    #print("Labels: ", labels, "OUT: ", out)
    
    r2 = r2_score(labels[masks["test"]].reshape(-1, 1).tolist(),
                  out[masks["test"]].tolist())
    if rank_full:
        pred_acc = out.reshape(-1).tolist()
        true_acc = labels.reshape(-1).tolist()
    else:
        pred_acc = np.array(out[masks["test"]].tolist()).reshape(-1)
        true_acc = np.array(labels[masks["test"]].reshape(-1).tolist())

    pred_rank = rankdata(pred_acc)
    true_rank = rankdata(true_acc)
    tau, p1 = kendalltau(pred_rank, true_rank)
    coeff, p2 = spearmanr(pred_rank, true_rank)

    top_arc_pred = np.argsort(pred_acc)[::-1]
    top_arc_true = np.argsort(true_acc)[::-1]

    precision_at_1 = precision(top_arc_true[:1], top_arc_pred[:1], 1)
    precision_at_10 = precision(top_arc_true[:10], top_arc_pred[:10], 10)
    precision_at_50 = precision(top_arc_true[:50], top_arc_pred[:50], 50)
    precision_at_100 = precision(top_arc_true[:100], top_arc_pred[:100], 100)
    metric = {'knn test mse': loss, 'knn test mae': mae_loss, 'knn test r2': r2,
              'kendall tau': tau, 'spearmanr coeff': coeff, 'top_1_correct': precision_at_1,
              'p@10': precision_at_10, 'p@50': precision_at_50,
              'p@100': precision_at_100, 'top acc': true_acc[top_arc_pred[0]]}

    return metric