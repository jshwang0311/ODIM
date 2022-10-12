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


class CIFAR10_mlp_Encoder(BaseNet):

    def __init__(self, x_dim, h_dims=[768, 384], rep_dim=128, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        #self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)
        self.q_z_mean = nn.Linear(h_dims[-1], self.rep_dim, bias=False)
        self.q_z_logvar = NonLinear(h_dims[-1], self.rep_dim, activation=nn.Hardtanh(min_val=-6.,max_val=2.))
    
    def Sample_Z(self , z_mu , z_log_var):    
        eps = torch.randn(z_log_var.shape).normal_().type(z_mu.type())          
        sam_z = z_mu + (torch.exp(z_log_var / 2)) * eps    
        return sam_z , eps


    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)

        #return self.code(x)
        return z_q_mean,z_q_logvar


class CIFAR10_mlp_Decoder(BaseNet):
    def __init__(self, x_dim, h_dims=[384, 768], rep_dim=128, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        #self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.recon_mean = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.recon_logvar = NonLinear(h_dims[-1], x_dim, activation=nn.Hardtanh(min_val=-6.,max_val=2.))
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        #x = self.reconstruction(x)
        #return self.output_activation(x)
        recon_mean = self.recon_mean(x)
        recon_logvar = self.recon_logvar(x)
        return self.output_activation(recon_mean), recon_logvar


class CIFAR10_mlp_Vae_gaussian(nn.Module):
    """
    Variational Autoencoder (VAE) (Kingma and Welling, 2013) model consisting of an encoder-decoder pair for which
    a variational distribution is fitted to the encoder.
    Also known as the M1 model in (Kingma et al., 2014)

    :param  dims: dimensions of the networks given by [input_dim, latent_dim, [hidden_dims]]. Encoder and decoder
    are build symmetrically.
    """
    def __init__(self, x_dim=3072, h_dims=[768, 384], rep_dim=128, bias=False):
        super(CIFAR10_mlp_Vae_gaussian,self).__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10_mlp_Encoder(x_dim, h_dims, rep_dim, bias)
        self.decoder = CIFAR10_mlp_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

        