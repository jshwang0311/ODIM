from .mnist_mlp_vae import MNIST_mlp_Vae
from .mnist_mlp_vae_gaussian import MNIST_mlp_Vae_gaussian
from .reuters_mlp_vae_gaussian import Reuters_mlp_Vae_gaussian
from .AD_mlp_vae_gaussian import AD_mlp_Vae_gaussian
from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .reuters_mlp import Reuters_mlp, Reuters_mlp_ae
from .AD_mlp import MLP, MLP_Autoencoder


def build_network(net_name,n_cluster=None):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet','mnist_mlp',
                            'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                            'thyroid_mlp',
                            'annthyroid_mlp', 'breastw_mlp', 'cover_mlp', 'ecoli_mlp', 'glass_mlp', 
                            'ionosphere_mlp', 'letter_mlp', 'lympho_mlp', 'mammography_mlp', 'musk_mlp', 
                            'optdigits_mlp', 'pendigits_mlp', 'pima_mlp', 'speech_mlp', 'vertebral_mlp', 
                            'vowels_mlp', 'wbc_mlp', 'wine_mlp',
                            '1_ALOI_mlp', '3_backdoor_mlp', '5_campaign_mlp', '7_Cardiotocography_mlp', 
                            '8_celeba_mlp',
                            '9_census_mlp', '11_donors_mlp', '13_fraud_mlp', '19_landsat_mlp', 
                            '22_magic.gamma_mlp', 
                            '27_PageBlocks_mlp', '33_skin_mlp', '35_SpamBase_mlp', '41_Waveform_mlp',
                            'AD_VResNet_mlp', 'AD_ViT_mlp','AD_ViT512_mlp','AD_ViT256_mlp',
                            'AD_BERT_mlp','AD_BERT512_mlp','AD_BERT256_mlp',
                            'AD_RoBERTa_mlp','AD_RoBERTa512_mlp','AD_RoBERTa256_mlp',
                            'mnist_mlp_vae','mnist_mlp_vae_gaussian', 
                            'reuters_mlp_vae_256_128_64_gaussian', 
                            'arrhythmia_mlp_vae_gaussian', 'cardio_mlp_vae_gaussian', 'satellite_mlp_vae_gaussian', 
                            'satimage-2_mlp_vae_gaussian', 'shuttle_mlp_vae_gaussian',
                            'annthyroid_mlp_vae_gaussian', 'breastw_mlp_vae_gaussian', 'cover_mlp_vae_gaussian', 'glass_mlp_vae_gaussian', 
                            'ionosphere_mlp_vae_gaussian', 'letter_mlp_vae_gaussian', 'mammography_mlp_vae_gaussian', 'musk_mlp_vae_gaussian', 
                            'optdigits_mlp_vae_gaussian', 'pendigits_mlp_vae_gaussian', 'pima_mlp_vae_gaussian', 'speech_mlp_vae_gaussian', 'vertebral_mlp_vae_gaussian', 
                            'vowels_mlp_vae_gaussian', 'wbc_mlp_vae_gaussian','thyroid_mlp_vae_gaussian',
                            '1_ALOI_mlp_vae_gaussian', '3_backdoor_mlp_vae_gaussian', '5_campaign_mlp_vae_gaussian', '7_Cardiotocography_mlp_vae_gaussian', 
                            '8_celeba_mlp_vae_gaussian',
                            '9_census_mlp_vae_gaussian', '11_donors_mlp_vae_gaussian', '13_fraud_mlp_vae_gaussian', '19_landsat_mlp_vae_gaussian', 
                            '22_magic.gamma_mlp_vae_gaussian', 
                            '27_PageBlocks_mlp_vae_gaussian', '33_skin_mlp_vae_gaussian', '35_SpamBase_mlp_vae_gaussian', '41_Waveform_mlp_vae_gaussian',
                            'AD_VResNet_mlp_vae_gaussian', 'AD_ViT_mlp_vae_gaussian','AD_ViT512_mlp_vae_gaussian','AD_ViT256_mlp_vae_gaussian',
                            'AD_BERT_mlp_vae_gaussian','AD_BERT512_mlp_vae_gaussian','AD_BERT256_mlp_vae_gaussian',
                            'AD_RoBERTa_mlp_vae_gaussian','AD_RoBERTa512_mlp_vae_gaussian','AD_RoBERTa256_mlp_vae_gaussian'
                           )

    assert net_name in implemented_networks

    net = None


    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()
    if net_name == 'reuters_mlp':
        net = Reuters_mlp(x_dim = 26147)
        
    if net_name == 'mnist_mlp_vae':
        net = MNIST_mlp_Vae()
    if net_name == 'mnist_mlp_vae_gaussian':
        net = MNIST_mlp_Vae_gaussian()
    

        
    if net_name == 'reuters_mlp_vae_256_128_64_gaussian':
        net = Reuters_mlp_Vae_gaussian(h_dims=[256, 128], rep_dim=64)
        
    
    if (net_name == 'AD_BERT_mlp_vae_gaussian') or (net_name == 'AD_RoBERTa_mlp_vae_gaussian'):
        net = AD_mlp_Vae_gaussian(x_dim=768, h_dims=[128, 64], rep_dim=32, bias=False)
    if (net_name == 'AD_BERT512_mlp_vae_gaussian') or (net_name == 'AD_RoBERTa512_mlp_vae_gaussian'):
        net = AD_mlp_Vae_gaussian(x_dim=768, h_dims=[512, 256], rep_dim=128, bias=False)
    if (net_name == 'AD_BERT256_mlp_vae_gaussian') or (net_name == 'AD_RoBERTa256_mlp_vae_gaussian'):
        net = AD_mlp_Vae_gaussian(x_dim=768, h_dims=[256, 128], rep_dim=64, bias=False)
        
    if net_name == 'AD_VResNet_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=512, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'AD_ViT_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=1000, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'AD_ViT512_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=1000, h_dims=[512, 256], rep_dim=128, bias=False)
    if net_name == 'AD_ViT256_mlp_vae_gaussian':
        net = AD_mlp_Vae_gaussian(x_dim=1000, h_dims=[256, 128], rep_dim=64, bias=False)
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
        
        
        
    if net_name == 'mnist_mlp':
        net = MLP(x_dim=784, h_dims=[128, 64], rep_dim=32, bias=False)
        
    if net_name == 'arrhythmia_mlp':
        net = MLP(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == 'cardio_mlp':
        net = MLP(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satellite_mlp':
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage-2_mlp':
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'shuttle_mlp':
        net = MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'thyroid_mlp':
        net = MLP(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
        
    if net_name == 'annthyroid_mlp':
        net = MLP(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)   
    if net_name == 'breastw_mlp':
        net = MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'cover_mlp':
        net = MLP(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'ecoli_mlp':
        net = MLP(x_dim=7, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'glass_mlp':
        net = MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'ionosphere_mlp':
        net = MLP(x_dim=33, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'letter_mlp':
        net = MLP(x_dim=32, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'lympho_mlp':
        net = MLP(x_dim=18, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'mammography_mlp':
        net = MLP(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'musk_mlp':
        net = MLP(x_dim=166, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'optdigits_mlp':
        net = MLP(x_dim=64, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'pendigits_mlp':
        net = MLP(x_dim=16, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'pima_mlp':
        net = MLP(x_dim=8, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'speech_mlp':
        net = MLP(x_dim=400, h_dims=[256, 128], rep_dim=64, bias=False)
    if net_name == 'vertebral_mlp':
        net = MLP(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'vowels_mlp':
        net = MLP(x_dim=12, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'wbc_mlp':
        net = MLP(x_dim=30, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'wine_mlp':
        net = MLP(x_dim=13, h_dims=[32, 16], rep_dim=8, bias=False)
    
    if (net_name == 'AD_BERT_mlp') or (net_name == 'AD_RoBERTa_mlp'):
        net = MLP(x_dim=768, h_dims=[128, 64], rep_dim=32, bias=False)
    if (net_name == 'AD_BERT512_mlp') or (net_name == 'AD_RoBERTa512_mlp'):
        net = MLP(x_dim=768, h_dims=[512, 256], rep_dim=128, bias=False)
    if (net_name == 'AD_BERT256_mlp') or (net_name == 'AD_RoBERTa256_mlp'):
        net = MLP(x_dim=768, h_dims=[256, 128], rep_dim=64, bias=False)
        
    if net_name == 'AD_VResNet_mlp':
        net = MLP(x_dim=512, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'AD_ViT_mlp':
        net = MLP(x_dim=1000, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'AD_ViT512_mlp':
        net = MLP(x_dim=1000, h_dims=[512, 256], rep_dim=128, bias=False)
    if net_name == 'AD_ViT256_mlp':
        net = MLP(x_dim=1000, h_dims=[256, 128], rep_dim=64, bias=False)
    if net_name == '1_ALOI_mlp':
        net = MLP(x_dim=27, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '3_backdoor_mlp':
        net = MLP(x_dim=196, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == '5_campaign_mlp':
        net = MLP(x_dim=62, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '7_Cardiotocography_mlp':
        net = MLP(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '8_celeba_mlp':
        net = MLP(x_dim=39, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '9_census_mlp':
        net = MLP(x_dim=500, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == '11_donors_mlp':
        net = MLP(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '13_fraud_mlp':
        net = MLP(x_dim=29, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '19_landsat_mlp':
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '22_magic.gamma_mlp':
        net = MLP(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '27_PageBlocks_mlp':
        net = MLP(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '33_skin_mlp':
        net = MLP(x_dim=3, h_dims=[32, 16], rep_dim=2, bias=False)
    if net_name == '35_SpamBase_mlp':
        net = MLP(x_dim=57, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '41_Waveform_mlp':
        net = MLP(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)


    return net




def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'mnist_mlp',
                            'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                            'thyroid_mlp',
                           'annthyroid_mlp', 'breastw_mlp', 'cover_mlp', 'ecoli_mlp', 'glass_mlp', 'ionosphere_mlp', 'letter_mlp',
                            'lympho_mlp', 'mammography_mlp', 'musk_mlp', 'optdigits_mlp', 'pendigits_mlp', 'pima_mlp', 'speech_mlp', 
                            'vertebral_mlp', 'vowels_mlp', 'wbc_mlp', 'wine_mlp', 'reuters_mlp',
                           '1_ALOI_mlp', '3_backdoor_mlp', '5_campaign_mlp', '7_Cardiotocography_mlp', 
                            '8_celeba_mlp',
                            '9_census_mlp', '11_donors_mlp', '13_fraud_mlp', '19_landsat_mlp', 
                            '22_magic.gamma_mlp', 
                            '27_PageBlocks_mlp', '33_skin_mlp', '35_SpamBase_mlp', '41_Waveform_mlp',
                            'AD_VResNet_mlp', 'AD_ViT_mlp','AD_ViT512_mlp','AD_ViT256_mlp',
                            'AD_BERT_mlp','AD_BERT512_mlp','AD_BERT256_mlp',
                            'AD_RoBERTa_mlp','AD_RoBERTa512_mlp','AD_RoBERTa256_mlp')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_mlp':
        ae_net = MLP_Autoencoder(x_dim=784, h_dims=[128, 64], rep_dim=32, bias=False)
        
    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()
    if net_name == 'reuters_mlp':
        ae_net = Reuters_mlp_ae()
    if net_name == 'arrhythmia_mlp':
        ae_net = MLP_Autoencoder(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == 'cardio_mlp':
        ae_net = MLP_Autoencoder(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satellite_mlp':
        ae_net = MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage-2_mlp':
        ae_net = MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'shuttle_mlp':
        ae_net = MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'thyroid_mlp':
        ae_net = MLP_Autoencoder(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    
    if net_name == 'annthyroid_mlp':
        ae_net = MLP_Autoencoder(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'breastw_mlp':
        ae_net = MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'cover_mlp':
        ae_net = MLP_Autoencoder(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'ecoli_mlp':
        ae_net = MLP_Autoencoder(x_dim=7, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'glass_mlp':
        ae_net = MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'ionosphere_mlp':
        ae_net = MLP_Autoencoder(x_dim=33, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'letter_mlp':
        ae_net = MLP_Autoencoder(x_dim=32, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'lympho_mlp':
        ae_net = MLP_Autoencoder(x_dim=18, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'mammography_mlp':
        ae_net = MLP_Autoencoder(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'musk_mlp':
        ae_net = MLP_Autoencoder(x_dim=166, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'optdigits_mlp':
        ae_net = MLP_Autoencoder(x_dim=64, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'pendigits_mlp':
        ae_net = MLP_Autoencoder(x_dim=16, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'pima_mlp':
        ae_net = MLP_Autoencoder(x_dim=8, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'speech_mlp':
        ae_net = MLP_Autoencoder(x_dim=400, h_dims=[256, 128], rep_dim=64, bias=False)
    if net_name == 'vertebral_mlp':
        ae_net = MLP_Autoencoder(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    if net_name == 'vowels_mlp':
        ae_net = MLP_Autoencoder(x_dim=12, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'wbc_mlp':
        ae_net = MLP_Autoencoder(x_dim=30, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == 'wine_mlp':
        ae_net = MLP_Autoencoder(x_dim=13, h_dims=[32, 16], rep_dim=8, bias=False)


    if (net_name == 'AD_BERT_mlp') or (net_name == 'AD_RoBERTa_mlp'):
        ae_net =MLP_Autoencoder(x_dim=768, h_dims=[128, 64], rep_dim=32, bias=False)
    if (net_name == 'AD_BERT512_mlp') or (net_name == 'AD_RoBERTa512_mlp'):
        ae_net =MLP_Autoencoder(x_dim=768, h_dims=[512, 256], rep_dim=128, bias=False)
    if (net_name == 'AD_BERT256_mlp') or (net_name == 'AD_RoBERTa256_mlp'):
        ae_net =MLP_Autoencoder(x_dim=768, h_dims=[256, 128], rep_dim=64, bias=False)
        
    if net_name == 'AD_VResNet_mlp':
        ae_net =MLP_Autoencoder(x_dim=512, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'AD_ViT_mlp':
        ae_net =MLP_Autoencoder(x_dim=1000, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == 'AD_ViT512_mlp':
        ae_net =MLP_Autoencoder(x_dim=1000, h_dims=[512, 256], rep_dim=128, bias=False)
    if net_name == 'AD_ViT256_mlp':
        ae_net =MLP_Autoencoder(x_dim=1000, h_dims=[256, 128], rep_dim=64, bias=False)
    if net_name == '1_ALOI_mlp':
        ae_net =MLP_Autoencoder(x_dim=27, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '3_backdoor_mlp':
        ae_net =MLP_Autoencoder(x_dim=196, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == '5_campaign_mlp':
        ae_net =MLP_Autoencoder(x_dim=62, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '7_Cardiotocography_mlp':
        ae_net =MLP_Autoencoder(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '8_celeba_mlp':
        ae_net =MLP_Autoencoder(x_dim=39, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '9_census_mlp':
        ae_net =MLP_Autoencoder(x_dim=500, h_dims=[128, 64], rep_dim=32, bias=False)
    if net_name == '11_donors_mlp':
        ae_net =MLP_Autoencoder(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '13_fraud_mlp':
        ae_net =MLP_Autoencoder(x_dim=29, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '19_landsat_mlp':
        ae_net =MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '22_magic.gamma_mlp':
        ae_net =MLP_Autoencoder(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '27_PageBlocks_mlp':
        ae_net =MLP_Autoencoder(x_dim=10, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '33_skin_mlp':
        ae_net =MLP_Autoencoder(x_dim=3, h_dims=[32, 16], rep_dim=2, bias=False)
    if net_name == '35_SpamBase_mlp':
        ae_net =MLP_Autoencoder(x_dim=57, h_dims=[32, 16], rep_dim=8, bias=False)
    if net_name == '41_Waveform_mlp':
        ae_net =MLP_Autoencoder(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

        
    return ae_net
