import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

from utils import Option

SOS_ID = 0
EOS_ID = 0

class GIN(nn.Module):
    def __init__(self, params, task="node"):
        """
        :param n_feat: int, dim of input feature
        :param n_hidden: dim of hidden
        :param num_layers: number of GIN Layers
        :param task: string, task type, node/graph
        """
        super().__init__()
        n_feat, n_hidden, n_emb, n_layers = params.in_channels, params.hidden_dim, params.emb_dim, params.n_layers
        n_class = params.out_channels

        self.conv1 = GINConv(
            Sequential(Linear(n_feat, n_hidden), BatchNorm1d(n_hidden), ReLU(),
                       Linear(n_hidden, n_hidden), ReLU()))

        self.convs = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(Linear(n_hidden, n_hidden),
                               BatchNorm1d(n_hidden),
                               ReLU(),
                               Linear(n_hidden, n_hidden),
                               ReLU()))
            )
        self.pool = False if task == "node" else True

        self.lin1 = Linear(n_hidden, n_emb)
        self.lin2 = Linear(n_emb, n_class)

    def forward(self, x, edge_index, batch=None, embedding=False):
        x = x.float()
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        if self.pool:
            x = global_add_pool(x, batch)

        x = self.lin1(x).relu()
        x = self.lin2(x)
        
        return x.sigmoid()

class LoGIN(nn.Module):
    def __init__(self, params, task="node"):
        """
        :param n_feat: int, dim of input feature
        :param n_hidden: dim of hidden
        :param num_layers: number of GIN Layers
        :param task: string, task type, node/graph
        """
        super().__init__()
        n_feat, n_hidden, n_emb, n_layers = params.in_channels, params.hidden_dim, params.emb_dim, params.n_layers
        n_class = params.out_channels
        self.conv1 = GINConv(
            Sequential(Linear(n_feat, n_hidden), BatchNorm1d(n_hidden), ReLU(),
                       Linear(n_hidden, n_hidden), ReLU()))

        self.convs = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(Linear(n_hidden, n_hidden),
                               BatchNorm1d(n_hidden),
                               ReLU(),
                               Linear(n_hidden, n_hidden),
                               ReLU()))
            )
        self.pool = False if task == "node" else True

        self.lin1 = Linear(n_hidden, n_emb)
        self.lin2 = Linear(n_emb, n_class)

        self.softmax = params.softmax
    
    def forward(self, x, edge_index, batch=None, embedding=False):

        x = x.float()
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        if self.pool:
            x = global_add_pool(x, batch)    
        
        x = self.lin1(x).relu()
        x = self.lin2(x)
        
        if not self.softmax:
            return x
        else:
            m = torch.nn.Softmax(dim=1)
            x = m(x)
            return torch.log(x)  


class IndependentPredictor(nn.Module):
    def __init__(self, encoder, fig):
        super(IndependentPredictor, self).__init__()

        # Create your independent predictors (assuming you have a class for them)
        self.predictors = nn.ModuleList(encoder)
        
        # Define the MLP for fusion
        self.mlp = nn.Sequential(
            nn.Linear(10*fig, 128),  # Input size: flattened outputs from 'fig' predictors
            nn.ReLU(),
            nn.Linear(128, 64),   # Adding more layers
            nn.ReLU(),
            nn.Linear(64, 32),    # Adding more layers
            nn.ReLU(),
            nn.Linear(32, 1)      # Output a single scalar
        )

    def forward(self, concatenated_predictions):
        output_scalar = self.mlp(concatenated_predictions)
        return output_scalar.sigmoid()
        
        
'''
class GCN(nn.Module):
    def __init__(self, params, task="node"):
        """
        :param n_feat: int, dim of input feature
        :param n_hidden: dim of hidden
        :param num_layers: number of GIN Layers
        :param task: string, task type, node/graph
        """
        super().__init__()
        n_feat, n_hidden, n_emb, n_layers = params.in_channels, params.hidden_dim, params.emb_dim, params.n_layers
        n_class = params.out_channels
        self.lstm = params.lstm
    
        self.conv1 = GCNConv(n_feat, n_hidden)

        self.convs = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.convs.append(
                GCNConv(n_hidden, n_hidden)
            )
        self.pool = False if task == "node" else True
        
        
        self.lin1 = Linear(n_hidden, n_emb)
        if self.lstm: n_emb += 32
        self.lin2 = Linear(n_emb, n_class)
        
        self.conv2 = GCNConv(n_hidden, n_class)
        self.sigmoid = params.sigmoid

    def forward(self, x, edge_index, batch=None, embedding=False):
        x = x.float()
        
        if self.lstm: y = x
        
        x = self.conv1(x, edge_index).relu()
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        if self.pool:
            x = global_add_pool(x, batch)



        x = self.lin1(x).relu()
        
        return self.conv2(x, edge_index).sigmoid()
        
        if embedding:
            return x
        if self.lstm: 
            x = torch.concat((x, y), dim=1)
        x = self.lin2(x)

        if not self.sigmoid:
            return x
        return x.sigmoid()

'''