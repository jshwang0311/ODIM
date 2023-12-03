#!/bin/bash

python calculate_AUC_fully_fit.py --dataset_name "mnist" --gpu_num 0 
python calculate_AUC_fully_fit.py --dataset_name "fmnist" --gpu_num 0
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_0"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian" 
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_1"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_2"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_3"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_4"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_5"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_6"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_7"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_8"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "CIFAR10_9"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_brightness"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_canny_edges"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_dotted_line"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_fog"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_glass_blur"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_identity"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_impulse_noise"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_motion_blur"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_rotate"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_scale"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_shear"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_shot_noise"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_spatter"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_stripe"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_translate"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MNIST-C_zigzag"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT256_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_bottle"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_cable"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_capsule"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_carpet"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_grid"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_hazelnut"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_leather"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_metal_nut"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_pill"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_screw"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_tile"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_toothbrush"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_transistor"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_wood"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "MVTec-AD_zipper"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"

python calculate_AUC_fully_fit.py --dataset_name "20news_0"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "20news_1"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "20news_2"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "20news_3"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "20news_4"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "20news_5"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "agnews_0"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "agnews_1"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "agnews_2"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_fully_fit.py --dataset_name "agnews_3"  --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"

python calculate_AUC_fully_fit.py --dataset_name "arrhythmia" --gpu_num 0 
python calculate_AUC_fully_fit.py --dataset_name "cardio" --gpu_num 0 
python calculate_AUC_fully_fit.py --dataset_name "satellite" --gpu_num 0
python calculate_AUC_fully_fit.py --dataset_name "satimage-2" --gpu_num 0
python calculate_AUC_fully_fit.py --dataset_name "shuttle" --gpu_num 0
python calculate_AUC_fully_fit.py --dataset_name "thyroid" --gpu_num 0
