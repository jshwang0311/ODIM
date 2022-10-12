import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
import numpy as np
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


#=======================================================================================================================
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

#=======================================================================================================================
def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

class CIFAR10_pixelcnn(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()
        self.rep_dim = rep_dim
        self.conv = nn.Sequential(
            GatedConv2d(3, 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )

        #self.fc1 = nn.Linear(128 * 3, 128, bias=False)
        self.q_z_mean = nn.Linear(128 * 3, 128, bias=False)
        self.q_z_logvar = NonLinear(128 * 3, 128, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv(x)
        x = x.view(int(x.size(0)), -1)
        #x = self.fc1(x)
        #return x
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)

        return z_q_mean,z_q_logvar
    
    def Sample_Z(self , z_mu , z_log_var):    
        eps = torch.randn(z_log_var.shape).normal_().type(z_mu.type())          
        sam_z = z_mu + (torch.exp(z_log_var / 2)) * eps    
        return sam_z , eps


    
class CIFAR10_pixelcnn_Decoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()
        self.rep_dim = rep_dim
        self.p_x_layers = nn.Sequential(
            GatedDense(128, 300),
            GatedDense(300, 3072)
        )
        
        
        # PixelCNN
        act = nn.ReLU(True)
        self.pixelcnn = nn.Sequential(
            MaskedConv2d('A', 6, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,######
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act
        )

        #self.p_x_mean = Conv2d(64, 3, 1, 1, 0, activation=nn.Sigmoid())
        self.recon_mean = Conv2d(64, 3, 1, 1, 0, activation=nn.Sigmoid())
        self.recon_logvar = Conv2d(64, 3, 1, 1, 0, activation=nn.Hardtanh(min_val=-6.,max_val=2.))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)


    def forward(self, x,z):
        x = x.view(-1, 3, 32, 32)
         # processing z2
        z = self.p_x_layers(z)
        z = z.view(-1, 3,32,32)
        # concatenate x and z1 and z2
        h = torch.cat((x,z), 1)
        # pixelcnn part of the decoder
        h_pixelcnn = self.pixelcnn(h)
        #x_mean = self.p_x_mean(h_pixelcnn).view(-1,3,32,32)
        #return x_mean
        
        recon_mean = self.recon_mean(h_pixelcnn).view(-1,3,32,32)
        recon_logvar = self.recon_logvar(h_pixelcnn)
        return recon_mean, recon_logvar


class CIFAR10_pixelcnn_Vae_gaussian(BaseNet):

    def __init__(self):
        super().__init__()

        self.encoder = CIFAR10_pixelcnn()
        self.decoder = CIFAR10_pixelcnn_Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(x,z)
        return x


