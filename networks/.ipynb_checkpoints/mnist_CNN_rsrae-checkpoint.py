import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


#=======================================================================================================================
# WEIGHTS INITS
#=======================================================================================================================
def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

################################################################################################
## Gated Dense structure
################################################################################################

class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

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
#=======================================================================================================================
class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

#=======================================================================================================================
class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None, bias=True):
        super(Conv2d, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

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


################################################################################################
## Encoder structure
################################################################################################
    
class MNIST_cnn_Encoder(torch.nn.Module):
    def __init__(self, X_dim):
        ## n_hidden must be larger than 0!
        super(MNIST_cnn_Encoder, self).__init__()
        # encoder: q(z | x)
        self.q_z_layers = nn.Sequential(
            GatedConv2d(1, 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)
        
    
    #def forward(self, x,option):
    def forward(self , x):
        x = x.view(-1, 1, 28, 28)
        h = self.q_z_layers(x)
        h = h.view(x.size(0),-1)

        return h
               
################################################################################################
## Decoder structure
################################################################################################

class MNIST_cnn_Decoder(torch.nn.Module):
    def __init__(self, Z_dim, X_dim , n_hidden=2):
        ## n_hidden must be larger than 0!
        super(MNIST_cnn_Decoder, self).__init__()
        # decoder: p(x | z)
        self.p_x_layers = nn.Sequential(
            GatedDense(Z_dim, 300),
            GatedDense(300, 784)
        )
        
        # decoder: p(x | z)
        act = nn.ReLU(True)
        # joint
        self.p_x_layers_joint = nn.Sequential(
            GatedConv2d(1, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
        )
        self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
                
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)
        
    def forward(self, z):
        z = self.p_x_layers(z)

        z = z.view(-1, 1, 28, 28)
        z = self.p_x_layers_joint(z)
        
        
        x_mean = self.p_x_mean(z).view(-1,784)

        return x_mean
    


class MNIST_cnn_RSRAE(nn.Module):
    """
    Variational Autoencoder (VAE) (Kingma and Welling, 2013) model consisting of an encoder-decoder pair for which
    a variational distribution is fitted to the encoder.
    Also known as the M1 model in (Kingma et al., 2014)

    :param  dims: dimensions of the networks given by [input_dim, latent_dim, [hidden_dims]]. Encoder and decoder
    are build symmetrically.
    """
    def __init__(self, x_dim=784, h_dims=294, rep_dim=40, bias=False):
        super(MNIST_cnn_RSRAE,self).__init__()

        self.rep_dim = rep_dim
        self.encoder = MNIST_cnn_Encoder(x_dim)
        self.decoder = MNIST_cnn_Decoder(rep_dim, x_dim, bias)
        self.rsr = RSRLayer(latent_dim = rep_dim, h = h_dims)
         