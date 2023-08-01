import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from base.base_net import BaseNet

import numpy as np


################################################################################################
## Nonlinear structure
################################################################################################

class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h

class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))

    
class Linear_BN_ReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_ReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.relu(self.bn(self.linear(x)))    


class Reuters_mlp(BaseNet):

    def __init__(self, x_dim, h_dims=[32, 64, 128], rep_dim=10, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_ReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)
    
    def Sample_Z(self , z_mu , z_log_var):    
        eps = torch.randn(z_log_var.shape).normal_().type(z_mu.type())          
        sam_z = z_mu + (torch.exp(z_log_var / 2)) * eps    
        return sam_z , eps


    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        
        return self.code(x)
        


class Reuters_mlp_Decoder(BaseNet):
    def __init__(self, x_dim, h_dims=[128, 64, 32], rep_dim=10, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_ReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        return self.output_activation(x)


class Reuters_mlp_ae(BaseNet):

    def __init__(self, x_dim=26147, h_dims=[32, 64, 128], rep_dim=10, bias=False):
        super().__init__()
        self.rep_dim = rep_dim
        self.encoder = Reuters_mlp(x_dim, h_dims, rep_dim, bias)
        self.decoder = Reuters_mlp_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

        