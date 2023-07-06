from .mnist_mlp_vae import MNIST_mlp_Vae
from .mnist_mlp_vae_gaussian import MNIST_mlp_Vae_gaussian
from .reuters_mlp_vae_gaussian import Reuters_mlp_Vae_gaussian
from .AD_mlp_vae_gaussian import AD_mlp_Vae_gaussian


def build_network(net_name,n_cluster=None):
    """Builds the neural network."""

    implemented_networks = ('mnist_mlp_vae','mnist_mlp_vae_gaussian', 
                            'reuters_mlp_vae_256_128_64_gaussian', 
                            'arrhythmia_mlp_vae_gaussian', 'cardio_mlp_vae_gaussian', 'satellite_mlp_vae_gaussian', 
                            'satimage-2_mlp_vae_gaussian', 'shuttle_mlp_vae_gaussian',
                            'annthyroid_mlp_vae_gaussian', 'breastw_mlp_vae_gaussian', 'cover_mlp_vae_gaussian', 'glass_mlp_vae_gaussian', 'ionosphere_mlp_vae_gaussian', 'letter_mlp_vae_gaussian', 'mammography_mlp_vae_gaussian', 'musk_mlp_vae_gaussian', 'optdigits_mlp_vae_gaussian', 'pendigits_mlp_vae_gaussian', 'pima_mlp_vae_gaussian', 'speech_mlp_vae_gaussian', 'vertebral_mlp_vae_gaussian', 'vowels_mlp_vae_gaussian', 'wbc_mlp_vae_gaussian','thyroid_mlp_vae_gaussian',
                            
                            '1_ALOI_mlp_vae_gaussian', '3_backdoor_mlp_vae_gaussian', '5_campaign_mlp_vae_gaussian', '7_Cardiotocography_mlp_vae_gaussian', '8_celeba_mlp_vae_gaussian',
                            '9_census_mlp_vae_gaussian', '11_donors_mlp_vae_gaussian', '13_fraud_mlp_vae_gaussian', '19_landsat_mlp_vae_gaussian', '22_magic.gamma_mlp_vae_gaussian', 
                            '27_PageBlocks_mlp_vae_gaussian', '33_skin_mlp_vae_gaussian', '35_SpamBase_mlp_vae_gaussian', '41_Waveform_mlp_vae_gaussian'
                           )

    assert net_name in implemented_networks

    net = None


    if net_name == 'mnist_mlp_vae':
        net = MNIST_mlp_Vae()
    if net_name == 'mnist_mlp_vae_gaussian':
        net = MNIST_mlp_Vae_gaussian()
    

        
    if net_name == 'reuters_mlp_vae_256_128_64_gaussian':
        net = Reuters_mlp_Vae_gaussian(h_dims=[256, 128], rep_dim=64)
        
    
    if net_name == '1_ALOI_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=27, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '3_backdoor_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=196, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == '5_campaign_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=62, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '7_Cardiotocography_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '8_celeba_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=39, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '9_census_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=500, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == '11_donors_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '13_fraud_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=29, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '19_landsat_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '22_magic.gamma_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '27_PageBlocks_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '33_skin_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=3, h_dims=[32, 16], rep_dim=2, bias=False)
    if net_name == '35_SpamBase_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=57, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '41_Waveform_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)
        
    
    
    
    if net_name == 'arrhythmia_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'cardio_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'satellite_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'satimage-2_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'shuttle_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'thyroid_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'annthyroid_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
        #net = AD_mlp_Vae_gaussian(x_dim=6, h_dims=[2], rep_dim=1, bias=False)
    if net_name == 'breastw_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'cover_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'glass_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'ionosphere_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=33, h_dims=[32, 16], rep_dim=8, bias=False)
        #net = AD_mlp_Vae_gaussian(x_dim=33, h_dims=[64,32], rep_dim=16, bias=False)
    if net_name == 'letter_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=32, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'mammography_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'musk_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=166, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'optdigits_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=64, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'pendigits_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=16, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'pima_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=8, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'speech_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=400, h_dims=[256, 128], rep_dim=64, bias=False)
    if net_name == 'vertebral_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'vowels_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=12, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'wbc_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=30, h_dims=[32, 16], rep_dim=8, bias=False)


    return net

