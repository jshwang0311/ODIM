import click
import torch
import logging
import random
import numpy as np
import time

from datasets.main import load_dataset
from optim.prop_trainer import *
import matplotlib.pyplot as plt
from PIL import Image
import os


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score


import argparse

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS 
# # # # # # # # # # #

parser = argparse.ArgumentParser(description='ODIM Experiment')
# arguments for optimization
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--dataset_name', type=str, default='42_WBC_all')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--filter_net_name', type=str, default='')
parser.add_argument('--data_path', type=str, default='')

# python calculate_AUC_light.py --dataset_name "CIFAR10_0"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT_mlp_vae_gaussian" 

args = parser.parse_args()
# args = parser.parse_args([])

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if __name__ == "__main__":
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # parameter setting
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    use_cuda = args.use_cuda
    gpu_num = args.gpu_num
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    ratio_known_normal = 0.0
    ratio_known_outlier = 0.0
    n_known_outlier_classes = 0
    
    data_path = '../data'
    if dataset_name == 'mnist':
        train_option = 'IWAE_alpha1._binarize'
        filter_net_name = 'mnist_mlp_vae'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4,5,6,7,8,9]
        patience_thres = 100
    elif dataset_name == 'fmnist':
        train_option = 'IWAE_alpha1._gaussian'
        filter_net_name = 'mnist_mlp_vae_gaussian'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4,5,6,7,8,9]
        patience_thres = 100
    elif dataset_name == 'wafer_scale':
        train_option = 'IWAE_alpha1._binarize'
        filter_net_name = 'mnist_mlp_vae'
        ratio_pollution = 0.1
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'annthyroid':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'annthyroid_mlp_vae_gaussian'
        ratio_pollution = 0.07
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'breastw':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'breastw_mlp_vae_gaussian'
        ratio_pollution = 0.35
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'cover':
        train_option = 'IWAE_alpha1.'
        filter_net_name =  'cover_mlp_vae_gaussian'
        ratio_pollution = 0.009
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'glass':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'glass_mlp_vae_gaussian'
        ratio_pollution = 0.042
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'ionosphere':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'ionosphere_mlp_vae_gaussian'
        ratio_pollution = 0.36
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'letter':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'letter_mlp_vae_gaussian'
        ratio_pollution = 0.0625
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'mammography':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'mammography_mlp_vae_gaussian'
        ratio_pollution = 0.0232
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'musk':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'musk_mlp_vae_gaussian'
        ratio_pollution = 0.032
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'optdigits':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'optdigits_mlp_vae_gaussian'
        ratio_pollution = 0.029
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'pendigits':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'pendigits_mlp_vae_gaussian'
        ratio_pollution = 0.0227
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'pima':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'pima_mlp_vae_gaussian'
        ratio_pollution = 0.34
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'speech':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'speech_mlp_vae_gaussian'
        ratio_pollution = 0.0165
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'vertebral':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'vertebral_mlp_vae_gaussian'
        ratio_pollution = 0.125
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'vowels':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'vowels_mlp_vae_gaussian'
        ratio_pollution = 0.034
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'wbc':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'wbc_mlp_vae_gaussian'
        ratio_pollution = 0.056
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'arrhythmia':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'arrhythmia_mlp_vae_gaussian'
        ratio_pollution = 0.14
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'cardio':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'cardio_mlp_vae_gaussian'
        ratio_pollution = 0.09
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'satellite':
        train_option = 'IWAE_alpha1.'
        filter_net_name =  'satellite_mlp_vae_gaussian'
        ratio_pollution = 0.31
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'satimage-2':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'satimage-2_mlp_vae_gaussian'
        ratio_pollution = 0.01
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'shuttle':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'shuttle_mlp_vae_gaussian'
        ratio_pollution = 0.07
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'thyroid':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'thyroid_mlp_vae_gaussian'
        ratio_pollution = 0.02
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'reuters':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'reuters_mlp_vae_256_128_64_gaussian'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4]
        patience_thres = 300
    

    elif '1_ALOI' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '1_ALOI_mlp_vae_gaussian'
        ratio_pollution = 0.0304
        normal_class_list = [0]
        patience_thres = 100
    elif '2_annthyroid' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'annthyroid_mlp_vae_gaussian'
        ratio_pollution = 0.07
        normal_class_list = [0]
        patience_thres = 100
    elif '3_backdoor' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '3_backdoor_mlp_vae_gaussian'
        ratio_pollution = 0.0244
        normal_class_list = [0]
        patience_thres = 100
    elif '4_breastw' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'breastw_mlp_vae_gaussian'
        ratio_pollution = 0.35
        normal_class_list = [0]
        patience_thres = 100
    elif '5_campaign' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '5_campaign_mlp_vae_gaussian'
        ratio_pollution = 0.11265
        normal_class_list = [0]
        patience_thres = 100
    elif '6_cardio' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'cardio_mlp_vae_gaussian'
        ratio_pollution = 0.0961
        normal_class_list = [0]
        patience_thres = 100
    elif '7_Cardiotocography' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '7_Cardiotocography_mlp_vae_gaussian'
        ratio_pollution = 0.2204
        normal_class_list = [0]
        patience_thres = 100
    elif '8_celeba' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '8_celeba_mlp_vae_gaussian'
        ratio_pollution = 0.0224
        normal_class_list = [0]
        patience_thres = 100
    elif '9_census' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '9_census_mlp_vae_gaussian'
        ratio_pollution = 0.062
        normal_class_list = [0]
        patience_thres = 100
    elif '10_cover' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name =  'cover_mlp_vae_gaussian'
        ratio_pollution = 0.009
        normal_class_list = [0]
        patience_thres = 100
    elif '11_donors' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '11_donors_mlp_vae_gaussian'
        ratio_pollution = 0.05925
        normal_class_list = [0]
        patience_thres = 100
    elif '13_fraud' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '13_fraud_mlp_vae_gaussian'
        ratio_pollution = 0.0017
        normal_class_list = [0]
        patience_thres = 100
    elif '14_glass' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '14_glass_mlp_vae_gaussian'
        ratio_pollution = 0.042
        normal_class_list = [0]
        patience_thres = 100
    elif '18_Ionosphere' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '18_Ionosphere_mlp_vae_gaussian'
        ratio_pollution = 0.36
        normal_class_list = [0]
        patience_thres = 100
    elif '19_landsat' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '19_landsat_mlp_vae_gaussian'
        ratio_pollution = 0.2071
        normal_class_list = [0]
        patience_thres = 100
    elif '20_letter' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'letter_mlp_vae_gaussian'
        ratio_pollution = 0.0625
        normal_class_list = [0]
        patience_thres = 100
    elif '22_magic.gamma' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '22_magic.gamma_mlp_vae_gaussian'
        ratio_pollution = 0.3516
        normal_class_list = [0]
        patience_thres = 100
    elif '23_mammography' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'mammography_mlp_vae_gaussian'
        ratio_pollution = 0.0232
        normal_class_list = [0]
        patience_thres = 100
    elif '25_musk' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'musk_mlp_vae_gaussian'
        ratio_pollution = 0.0317
        normal_class_list = [0]
        patience_thres = 100
    elif '26_optdigits' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'optdigits_mlp_vae_gaussian'
        ratio_pollution = 0.0288
        normal_class_list = [0]
        patience_thres = 100
    elif '27_PageBlocks' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '27_PageBlocks_mlp_vae_gaussian'
        ratio_pollution = 0.0946
        normal_class_list = [0]
        patience_thres = 100
    elif '28_pendigits' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'pendigits_mlp_vae_gaussian'
        ratio_pollution = 0.0227
        normal_class_list = [0]
        patience_thres = 100
    elif '29_Pima' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'pima_mlp_vae_gaussian'
        ratio_pollution = 0.34
        normal_class_list = [0]
        patience_thres = 100
    elif '30_satellite' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name =  'satellite_mlp_vae_gaussian'
        ratio_pollution = 0.31
        normal_class_list = [0]
        patience_thres = 100
    elif '31_satimage-2' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'satimage-2_mlp_vae_gaussian'
        ratio_pollution = 0.01
        normal_class_list = [0]
        patience_thres = 100
    elif '32_shuttle' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'shuttle_mlp_vae_gaussian'
        ratio_pollution = 0.07
        normal_class_list = [0]
        patience_thres = 100
    elif '33_skin' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '33_skin_mlp_vae_gaussian'
        ratio_pollution = 0.2075
        normal_class_list = [0]
        patience_thres = 100
    elif '35_SpamBase' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '35_SpamBase_mlp_vae_gaussian'
        ratio_pollution = 0.3991
        normal_class_list = [0]
        patience_thres = 100
    elif '36_speech' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'speech_mlp_vae_gaussian'
        ratio_pollution = 0.0165
        normal_class_list = [0]
        patience_thres = 100
    elif '38_thyroid' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'thyroid_mlp_vae_gaussian'
        ratio_pollution = 0.02
        normal_class_list = [0]
        patience_thres = 100
    elif '39_vertebral' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'vertebral_mlp_vae_gaussian'
        ratio_pollution = 0.125
        normal_class_list = [0]
        patience_thres = 100
    elif '40_vowels' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'vowels_mlp_vae_gaussian'
        ratio_pollution = 0.034
        normal_class_list = [0]
        patience_thres = 100
    elif '41_Waveform' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '41_Waveform_mlp_vae_gaussian'
        ratio_pollution = 0.029
        normal_class_list = [0]
        patience_thres = 100
    elif '42_WBC' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '42_WBC_mlp_vae_gaussian'
        ratio_pollution = 0.0448
        normal_class_list = [0]
        patience_thres = 100
    
    
    
    elif '12_fault' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '12_fault_mlp_vae_gaussian'
        ratio_pollution = 0.3467
        normal_class_list = [0]
        patience_thres = 100
    elif '15_Hepatitis' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '15_Hepatitis_mlp_vae_gaussian'
        ratio_pollution = 0.1625
        normal_class_list = [0]
        patience_thres = 100
        batch_size = 32
    elif '16_http' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '16_http_mlp_vae_gaussian'
        ratio_pollution = 0.003896
        normal_class_list = [0]
        patience_thres = 100
    elif '17_InternetAds' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '17_InternetAds_mlp_vae_gaussian'
        ratio_pollution = 0.1872
        normal_class_list = [0]
        patience_thres = 100
    elif '21_Lymphography' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '21_Lymphography_mlp_vae_gaussian'
        ratio_pollution = 0.0405
        normal_class_list = [0]
        patience_thres = 100
    elif '34_smtp' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '34_smtp_mlp_vae_gaussian'
        ratio_pollution = 0.0003
        normal_class_list = [0]
        patience_thres = 100
    elif '37_Stamps' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '37_Stamps_mlp_vae_gaussian'
        ratio_pollution = 0.0912
        normal_class_list = [0]
        patience_thres = 100
    elif '43_WDBC' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '43_WDBC_mlp_vae_gaussian'
        ratio_pollution = 0.0272
        normal_class_list = [0]
        patience_thres = 100
    elif '44_Wilt' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '44_Wilt_mlp_vae_gaussian'
        ratio_pollution = 0.0533
        normal_class_list = [0]
        patience_thres = 100
    elif '45_wine' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '45_wine_mlp_vae_gaussian'
        ratio_pollution = 0.0775
        normal_class_list = [0]
        patience_thres = 100
    elif '46_WPBC' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '46_WPBC_mlp_vae_gaussian'
        ratio_pollution = 0.2374
        normal_class_list = [0]
        patience_thres = 100
    elif '47_yeast' in dataset_name:
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '47_yeast_mlp_vae_gaussian'
        ratio_pollution = 0.3416
        normal_class_list = [0]
        patience_thres = 100

    
    
    elif 'CIFAR10' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100
    elif 'MNIST-C' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100
    elif 'MVTec-AD' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100
    elif 'SVHN' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100

        
    elif '20news' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100    
    elif 'agnews' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100    
    elif 'amazon' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100
    elif 'imdb' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100  
    elif 'yelp' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100  
        
    
        


    data_seed_list = [110,120,130,140,150]
    start_model_seed = 1234
    n_ens = 10
    normal_class_idx = 0
    for normal_class_idx in range(len(normal_class_list)):
        normal_class = normal_class_list[normal_class_idx]
        known_outlier_class = 0


        # Default device to 'cpu' if cuda is not available
        if not torch.cuda.is_available():
            device = 'cpu'

        torch.cuda.set_device(gpu_num)
        print('Current number of the GPU is %d'%torch.cuda.current_device())


        seed_idx = 0
        nu = 0.1
        num_threads = 0
        n_jobs_dataloader = 0
        
        row_name_list = []
        for seed_idx in range(len(data_seed_list)):
            row_name = f'Class{normal_class}_simulation{seed_idx+1}'
            row_name_list.append(row_name)
        row_name = f'Average'
        row_name_list.append(row_name)
        row_name = f'Std'
        row_name_list.append(row_name)
        train_auc_list = []
        train_ap_list = []
        test_auc_list = []
        test_ap_list = []
        seed_idx = 0
        for seed_idx in range(len(data_seed_list)):
            seed = data_seed_list[seed_idx]

            save_metric_dir = f'Results/{dataset_name}'
            os.makedirs(save_metric_dir, exist_ok=True)
            save_dir = os.path.join(f'Results/{dataset_name}/ODIM_light{batch_size}_{filter_net_name}',f'log{seed}')
            os.makedirs(save_dir, exist_ok=True)
            save_score_dir = os.path.join(f'Results/{dataset_name}/ODIM_light{batch_size}_{filter_net_name}',f'score{seed}')
            os.makedirs(save_score_dir, exist_ok=True)
            


            # Set up logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log_file = save_dir + '/log_'+dataset_name+'_trainOption'+ train_option + '_normal' + str(normal_class) +'.txt'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info('-----------------------------------------------------------------')
            logger.info('-----------------------------------------------------------------')
            # Print paths
            logger.info('Log file is %s' % log_file)
            logger.info('Data path is %s' % data_path)
            # Print experimental setup
            logger.info('Dataset: %s' % dataset_name)
            logger.info('Normal class: %s' % normal_class)
            logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
            logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
            logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
            if n_known_outlier_classes == 1:
                logger.info('Known anomaly class: %d' % known_outlier_class)
            else:
                logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
            logger.info('Network: %s' % filter_net_name)


            # Print model configuration
            logger.info('nu-parameter: %.2f' % nu)

            # Set seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # Load data
            dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                                   ratio_known_normal, ratio_known_outlier, ratio_pollution,
                                   random_state=np.random.RandomState(seed))
            # Log random sample of known anomaly classes if more than 1 class
            if n_known_outlier_classes > 1:
                logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

            # Train Filter model 
            train_loader = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)


            ## extract train ys and idxs
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            train_ys = []
            train_idxs = []
            for (_, outputs, idxs) in train_loader:
                train_ys.append(outputs.data.numpy())
                train_idxs.append(idxs.data.numpy())

            train_ys = np.hstack(train_ys)
            train_idxs = np.hstack(train_idxs)
            train_idxs_ys = pd.DataFrame({'idx' : train_idxs,'y' : train_ys})
            logger.info(f'Sample Size : {len(train_ys)}')



            ens_train_me_losses = 0
            ens_st_train_me_losses = 0

            ## patience index
            train_n = train_ys.shape[0]
            check_iter = np.min(np.array([10, (train_n // batch_size)]))
            
            patience = np.ceil(patience_thres / check_iter).astype('int')
            loss_column = ['idx','ens_value','ens_st_value','y']
            for model_iter in range(n_ens):
                model_seed = start_model_seed+(model_iter*10)

                logger.info('Set model seed to %d.' % (model_seed))
                ## step 1
                train_idxs_losses, _, running_time = odim_light(filter_net_name, train_loader, train_loader, check_iter, patience, model_seed,seed, logger, train_option)
                train_me_losses = (train_idxs_losses.to_numpy())[:,1]
                st_train_me_losses = (train_me_losses - train_me_losses.mean())/train_me_losses.std()
                train_idxs_losses['st_loss'] = st_train_me_losses
                add_label_idx_losses = pd.merge(train_idxs_losses, train_idxs_ys, on ='idx')
                fpr, tpr, thresholds = metrics.roc_curve(np.array(add_label_idx_losses['y']), np.array(add_label_idx_losses['loss']), pos_label=1)
                roc_auc = metrics.auc(fpr, tpr)
                logger.info('\n...Train_AUC value- VAE: %0.4f' %(roc_auc))



                if model_iter == 0:
                    ens_loss = add_label_idx_losses
                    ens_loss.columns = loss_column

                else:
                    merge_data = pd.merge(ens_loss, train_idxs_losses, on = 'idx')
                    merge_data['ens_value'] = merge_data['ens_value'] + merge_data['loss']
                    merge_data['ens_st_value'] = merge_data['ens_st_value'] + merge_data['st_loss']
                    ens_loss = merge_data[loss_column]


                train_auc = roc_auc_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))
                train_ap = average_precision_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))
                logger.info('\n ...Train_AUC value- Ens VAE: %0.4f' % train_auc)
                logger.info('\n ...Train_PRAUC value- Ens VAE: %0.4f' % train_ap)


            logger.info('\n ...Final Train_AUC value of Ens VAE: %0.4f' %(train_auc))
            logger.info('\n ...Final Train_PRAUC value of Ens VAE: %0.4f' %(train_ap))
            train_auc_list.append(train_auc)
            train_ap_list.append(train_ap)


            ens_loss.to_csv(os.path.join(save_score_dir,'score_data.csv'),index=False)
            logger.removeHandler(file_handler)
        
        
        train_auc_list.append(np.mean(train_auc_list))
        train_auc_list.append(np.std(train_auc_list))
        train_ap_list.append(np.mean(train_ap_list))
        train_ap_list.append(np.std(train_ap_list))

        class_train_df = pd.DataFrame({
            'row_names' : row_name_list,
            'train_auc' : train_auc_list,
            'train_ap' : train_ap_list
        })
        class_train_df.set_index(keys = 'row_names', inplace = True)
        try:
            train_df = pd.concat([train_df, class_train_df], axis = 0)
        except:
            train_df = class_train_df

        train_df.to_csv(os.path.join(save_metric_dir,f'ODIM_light{batch_size}_{filter_net_name}_train_result.csv'))
        



            
    
