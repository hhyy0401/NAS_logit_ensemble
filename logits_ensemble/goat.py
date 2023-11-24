import math

import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected
from tqdm import tqdm
from utils import evaluation
from modules import GIN, LoGIN, IndependentPredictor
from predictor import Predictor
from model_utils import gnn_train, gnn_eval

from torch.utils.data import TensorDataset, DataLoader

class Ensemble_Predictor(Predictor):
    def __init__(self, params, labels=None, device=None, ss_type=None, encoding_type=None):
        super().__init__(labels, device, ss_type, encoding_type)
        self.params = params
        self.logits = (params.method == "logits")
        self.fig = params.model.spec.fig
        self.models = {}
        
        channels = params.model.spec.out_channels
        params.model.spec.out_channels = 10

        self.models["logits"] = [LoGIN(self.params.model.spec, task="graph").to(self.device) for _ in range(self.fig)]
        self.models["accs"] = IndependentPredictor(self.models["logits"], self.fig)

        params.model.spec.out_channels = channels
        
    def fit(self, masks, loaders):
        enc_train_loss_hist, enc_val_loss_hist = {}, {}

        for idx, encoder in enumerate(self.models['logits']):

            self.encoder_fit(encoder, loaders["train"], loaders["val"], self.params.model.training, idx)

            #for param in encoder.parameters():
            #    param.requires_grad = False
            #for param in encoder.lin3.parameters():
            #    param.requires_grad = True


        enc_train_loss_hist["logits_mlp"] = self.mlp_fit(self.models['accs'], 
                                                          loaders["train"],
                                                          self.params.model.training)

        

        return enc_train_loss_hist, enc_val_loss_hist
    
    '''
    def query(self, masks, loader=None):
        out = []
        labels = []
        encoder = self.models['accs']
               
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                emb = encoder(data.x, data.edge_index, data.batch)[:,-1]   
                out.extend(emb.tolist())
                labels.extend(data.y.float().reshape(emb.shape[0], -1)[:, -1].tolist())

        
        out = torch.tensor(out).squeeze().to(self.device)
        labels = torch.tensor(labels).to(self.device)
        print(labels[:10], out[:10])
        metric = evaluation(out, labels, masks, rank_full=False)
        
        return metric
        '''

    
    def encoder_fit(self, encoder, train_loader, val_loader, params, idx=-1):
        optimizer = torch.optim.Adam(encoder.parameters(),
                                     lr=params.lr,
                                     weight_decay=params.weight_decay)
    
        
        best_val_loss = math.inf
        best_val_model = encoder.state_dict()
        train_loss_hist = []
        val_loss_hist = []

        for epoch in tqdm(range(params.pre_epoch)):

            train_loss = gnn_train(encoder, train_loader,
                                    optimizer, self.device,
                                    params.loss_type, idx)
            train_loss_hist.append(train_loss)
        
            val_loss = gnn_eval(encoder, val_loader, self.device,
                                params.val_loss_type, idx)
            val_loss_hist.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_model = encoder.state_dict()

                
        encoder.load_state_dict(best_val_model)

        return train_loss_hist, val_loss_hist

    def mlp_fit(self, encoder, train_loader, params):
        
        train_loss_hist = []
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), 
                                     lr=params.lr, weight_decay=params.weight_decay)
        
        best_val_loss = math.inf
        best_val_model = encoder.state_dict()
        train_loss_hist = []

        for epoch in tqdm(range(params.epoch)):
            total_loss = 0.0
            for data in train_loader:
                data = data.to(self.device)
                
                # Generate predictions from self.predictors
                predictor_outputs = [model(data.x, data.edge_index, data.batch, False) 
                                     for model in encoder.predictors]

                # Compute concatenated predictions
                concatenated_predictions = torch.cat(predictor_outputs, dim=1).to("cpu")
        
                # Forward pass through the fusion MLP
                output = encoder.mlp(concatenated_predictions).squeeze()
                true = data.y.float().reshape(output.shape[0], -1)[:, -1].to("cpu")

                #pairs_list = list(zip(output, true)) 
                #pairs_list = [(a.item(), b.item()) for a, b in pairs_list]
                #print(pairs_list)
                # Compute the loss
                loss = torch.norm(output - true, p=2)  # Using the first set of fake labels for simplicity

                # Backpropagation and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * len(data)
             
            train_loss_hist.append(total_loss)

        return train_loss_hist

    def query(self, masks, loader=None):
        out = []
        labels = []
        encoder = self.models['accs']
               
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                predictor_outputs = [model(data.x, data.edge_index, data.batch, False) 
                                     for model in encoder.predictors]
                
                concatenated_predictions = torch.cat(predictor_outputs, dim=1).to("cpu")
                
                output = encoder.mlp(concatenated_predictions).squeeze()
                out.extend(output.tolist())
                labels.extend(data.y.float().reshape(output.shape[0], -1)[:, -1].tolist())

        
        out = torch.tensor(out).squeeze().to(self.device)
        labels = torch.tensor(labels).to(self.device)
        print(labels[:10], out[:10])
        metric = evaluation(out, labels, masks, rank_full=False)
        
        return metric
