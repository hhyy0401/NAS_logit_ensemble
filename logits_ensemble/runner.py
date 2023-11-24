import os
import sys
import json
import time
import copy
import pickle

import torch
import numpy as np
from tqdm import trange
from torch_geometric.loader import DataLoader as GDataLoader

from utils import Option, seed_everything
from gnn import GIN_Predictor
from goat import Ensemble_Predictor

from scipy.stats import kendalltau

import warnings
warnings.filterwarnings("ignore")

data_path = "../data/"
logits_path = "../data/logits/"

predictor_map = {"baseline": GIN_Predictor, "logits": Ensemble_Predictor}

def get_data(perm, dataset, param, method, device=None, logits=None, spec=None):

    batch_size = param.batch_size

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_mask, val_mask, test_mask = perm[:param.num_train], perm[param.num_train:param.num_train+param.num_val], perm[param.num_train+param.num_val:]
    selected_data = [dataset[i] for i in perm]
    

    if method == "logits":
        fig = spec.fig
        softmax = spec.softmax
        m = torch.nn.Softmax(dim=1)        
        for idx, data in enumerate(selected_data):
            for j in range(fig):
                if softmax:
                    data['logits_'+str(j)] = m(logits[idx][j])                    
                else: 
                    data['logits_'+str(j)] = logits[idx][j]    


    train_data = selected_data[:param.num_train]
    val_data = selected_data[param.num_train:param.num_train+param.num_val]
    test_data = selected_data[param.num_train+param.num_val:]       
    
    masks = {"train": torch.tensor(train_mask).long().to(device),
             "val": torch.tensor(val_mask).long().to(device),
             "test": torch.tensor(test_mask).long().to(device)}

    #if method == "baseline" or method == "logits":
    train_data = GDataLoader(train_data, shuffle=True, batch_size=batch_size["train"])
    val_data = GDataLoader(val_data, shuffle=False, batch_size=batch_size["val"])
    test_data = GDataLoader(test_data, shuffle=False, batch_size=batch_size["test"])
    full_data = GDataLoader(selected_data, shuffle=False, batch_size=batch_size["full"])

    data = {"train" : train_data,
            "val" : val_data,
            "test" : test_data,
            "full" : full_data
           }
    
    return masks, data


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config_fname = sys.argv[1]
    device = sys.argv[2]
    config = Option(config_fname)
    config.dataset.data_path = data_path
    method = str(config.method)
    
    # load data
    perms = np.load(data_path + "permutation.npy")
    dataset = torch.load(data_path + "nas201_sample.pt")
    labels = torch.load(data_path + "labels.pt")
        
    if config.method == "logits":
        logits = torch.load(logits_path + "random_logits.pt")[:, :, :config.model.spec.fig, :]
        #logits = torch.load(data_path + "logits_test_batch.pt").to(device)
        #logits = torch.load(data_path + "logits/logits.pt")[:, :, :config.model.spec.fig, :]
        #img_idx = torch.load(logits_path + "cifar10_idx.pt")[:, :config.model.spec.logits]
        #img_idx = torch.load(data_path + "logits/img.pt")[:, :config.model.spec.fig] 
        #img_emb = torch.load(data_path + "logits/cifar10_emb.pt")
        
    #TODO: signature 정하기
    signature = str(config.method)
    print("===================================================")
    print(signature)
    print("===================================================")
    

    results = {}
    for run in trange(10):
        curr = {}
        perm = perms[run]
        seed_everything()
        
        predictor = predictor_map[method](config, labels, device=device)
        if config.method == "logits":
            masks, data = get_data(perm, dataset, config.dataset, method, device, logits[run], config.model.spec)
        else:
            masks, data = get_data(perm, dataset, config.dataset, method, device, "", config.model.spec)

        e_t_hist, e_v_hist = predictor.fit(masks, data)
        metrics = predictor.query(masks, data['full'])
        
        for key in metrics:
            curr[key] = metrics[key]
            if key not in results:
                results[key] = [metrics[key]]
            else:
                results[key].append(metrics[key])
        for key in results:
            print(key, ": {:.4f}({:.4f}), curr: {:.4f} max: {:.4f}".format(np.mean(results[key]), np.std(results[key]), curr[key], np.max(results[key])))
