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

    
class RSRLayer(nn.Module):
    def __init__(self, latent_dim: int, h: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.h = h
        self.A = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(latent_dim, h)))

    def forward(self, z):
        # z is the output from the encoder
        z_hat = self.A @ z.view(z.size(0), self.h, 1)
        return z_hat.squeeze(2)


class MNIST_mlp_Encoder(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)


    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        
        return self.code(x)
        


class MNIST_mlp_Decoder(BaseNet):
    def __init__(self, x_dim, h_dims=[64, 128], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        return self.output_activation(x)


class MNIST_mlp_RSRAE(nn.Module):
    """
    Variational Autoencoder (VAE) (Kingma and Welling, 2013) model consisting of an encoder-decoder pair for which
    a variational distribution is fitted to the encoder.
    Also known as the M1 model in (Kingma et al., 2014)

    :param  dims: dimensions of the networks given by [input_dim, latent_dim, [hidden_dims]]. Encoder and decoder
    are build symmetrically.
    """
    def __init__(self, x_dim=784, h_dims=[128, 64], rep_dim=32, bias=False):
        super(MNIST_mlp_RSRAE,self).__init__()

        self.rep_dim = rep_dim
        self.encoder = MNIST_mlp_Encoder(x_dim, h_dims[:-1], h_dims[-1], bias)
        
        self.decoder = MNIST_mlp_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)
        self.rsr = RSRLayer(latent_dim = rep_dim, h = h_dims[-1])

        