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

class CIFAR10_LeNet_Encoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        #self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.q_z_mean = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.q_z_logvar = NonLinear(128 * 4 * 4, self.rep_dim, activation=nn.Hardtanh(min_val=-6.,max_val=2.))
        
    def Sample_Z(self , z_mu , z_log_var):    
        eps = torch.randn(z_log_var.shape).normal_().type(z_mu.type())          
        sam_z = z_mu + (torch.exp(z_log_var / 2)) * eps    
        return sam_z , eps

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        #x = self.fc1(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)

        #return x
        return z_q_mean,z_q_logvar


class CIFAR10_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class CIFAR10_LeNet_Vamp(nn.Module):
    """
    Variational Autoencoder (VAE) (Kingma and Welling, 2013) model consisting of an encoder-decoder pair for which
    a variational distribution is fitted to the encoder.
    Also known as the M1 model in (Kingma et al., 2014)

    :param  dims: dimensions of the networks given by [input_dim, latent_dim, [hidden_dims]]. Encoder and decoder
    are build symmetrically.
    """

    def __init__(self, rep_dim=128,n_cluster=1):
        super(CIFAR10_LeNet_Vamp, self).__init__()

        self.rep_dim = rep_dim
        self.n_cluster = n_cluster

        self.encoder = CIFAR10_LeNet_Encoder(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)
        

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        ##############################################
        ## pseudo input
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.means = NonLinear(n_cluster, 3072, bias=False, activation=nonlinearity)

        ## initialization
        normal_init(self.means.linear, 0.05, 0.01)

        ## identity matrix
        self.idle_input = (torch.eye(n_cluster, n_cluster))

        ##############################################
        ## cluster probability vector
        self.p_cluster = nn.Parameter(torch.zeros(n_cluster) , requires_grad=True)
        '''
        if learnable_cluster_prob_ox:
            self.p_cluster = nn.Parameter(torch.zeros(n_cluster) , requires_grad=True)
        else:
            self.p_cluster = torch.zeros(n_cluster).cuda()
        '''

    def Pseudo_input(self):        
        self.idle_input = self.idle_input.cuda()
        return self.means(self.idle_input)
    
    def P_cluster(self):
        return nn.Softmax(0)(self.p_cluster)
    
'''
class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''