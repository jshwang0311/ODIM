import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from base.base_net import BaseNet

import numpy as np
#=======================================================================================================================
def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)

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


class AD_mlp_Encoder(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
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


class AD_mlp_Decoder(BaseNet):
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
    
class AD_mlp_Vamp(nn.Module):
    """
    Variational Autoencoder (VAE) (Kingma and Welling, 2013) model consisting of an encoder-decoder pair for which
    a variational distribution is fitted to the encoder.
    Also known as the M1 model in (Kingma et al., 2014)

    :param  dims: dimensions of the networks given by [input_dim, latent_dim, [hidden_dims]]. Encoder and decoder
    are build symmetrically.
    """
    def __init__(self, x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False, n_cluster=1):
        super(AD_mlp_Vamp,self).__init__()

        self.rep_dim = rep_dim
        self.n_cluster = n_cluster
        self.encoder = AD_mlp_Encoder(x_dim, h_dims, rep_dim, bias)
        self.decoder = AD_mlp_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

        ##############################################
        ## pseudo input
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.means = NonLinear(n_cluster, x_dim, bias=False, activation=nonlinearity)

        ## initialization
        normal_init(self.means.linear, 0.05, 0.01)

        ## identity matrix
        self.idle_input = (torch.eye(n_cluster, n_cluster))

        ##############################################
        ## cluster probability vector
        self.p_cluster = nn.Parameter(torch.zeros(n_cluster) , requires_grad=True)

        # Init linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def Pseudo_input(self):        
        self.idle_input = self.idle_input.cuda()
        return self.means(self.idle_input)
    
    def P_cluster(self):
        return nn.Softmax(0)(self.p_cluster)

        