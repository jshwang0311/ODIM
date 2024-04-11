#!/bin/bash

python calculate_AUC_light_all.py --dataset_name "1_ALOI_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "3_backdoor_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "5_campaign_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "7_Cardiotocography_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "19_landsat_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "22_magic.gamma_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "27_PageBlocks_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "35_SpamBase_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "41_Waveform_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "8_celeba_all" --gpu_num 0 #
python calculate_AUC_light_all.py --dataset_name "9_census_all" --gpu_num 0 #



python calculate_AUC_light_all.py --dataset_name "CIFAR10_0_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian" 
python calculate_AUC_light_all.py --dataset_name "CIFAR10_1_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "CIFAR10_2_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "CIFAR10_3_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "CIFAR10_4_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "CIFAR10_5_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "CIFAR10_6_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "CIFAR10_7_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "CIFAR10_8_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "CIFAR10_9_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"


python calculate_AUC_light_all.py --dataset_name "20news_0_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "20news_1_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "20news_2_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "20news_3_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "20news_4_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "20news_5_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
