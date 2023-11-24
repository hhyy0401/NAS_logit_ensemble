import math

import torch

from modules import GIN
from utils import evaluation
from predictor import Predictor
from model_utils import gnn_train, gnn_eval
from tqdm import tqdm

class GIN_Predictor(Predictor):
    def __init__(self, params, labels=None, device=None, ss_type=None, encoding_type=None):
        super().__init__(labels, device, ss_type, encoding_type)
        self.params = params
        self.model = GIN(self.params.model.spec, task="graph").to(self.device)

    def fit(self, masks, loaders):
        train_loss_hist, val_loss_hist = self.gnn_fit(loaders["train"],
                                                          loaders["val"],
                                                          self.params.model.training)

        return train_loss_hist, val_loss_hist

    def query(self, masks, loaders):
        out, label = self.gnn_out(loaders)
        print(out[:10], self.labels[:10])
        
        metric = evaluation(out, label, masks, rank_full=False)
        return metric

    def gnn_fit(self, train_loader, val_loader, params):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=params.lr,
                                     weight_decay=params.weight_decay)

        best_val_loss = math.inf
        best_val_model = self.model.state_dict()
        train_loss_hist = []
        val_loss_hist = []
        for epoch in tqdm(range(params.epoch)):
            train_loss = gnn_train(self.model, train_loader,
                                   optimizer, self.device,
                                   multi=True, loss_type=params.val_loss_type)
            train_loss_hist.append(train_loss)
            val_loss = gnn_eval(self.model, val_loader, self.device,
                                multi=True, loss_type=params.val_loss_type)
            val_loss_hist.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_model = self.model.state_dict()

        self.model.load_state_dict(best_val_model)
        return train_loss_hist, val_loss_hist

    def gnn_out(self, loader):
        out, labels = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                emb = self.model(data.x, data.edge_index, data.batch, embedding=False)[:,-1]   
                out.extend(emb.tolist())
                labels.extend(data.y.float().reshape(emb.shape[0], -1)[:, -1].tolist())
        
        out = torch.tensor(out).squeeze().to(self.device)
        labels = torch.tensor(labels).to(self.device)
        
        return out, labels
    
