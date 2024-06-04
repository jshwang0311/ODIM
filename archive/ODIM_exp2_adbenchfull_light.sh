#!/bin/bash

python calculate_AUC_light_all.py --dataset_name "12_fault_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "15_Hepatitis_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "17_InternetAds_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "21_Lymphography_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "34_smtp_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "37_Stamps_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "43_WDBC_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "44_Wilt_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "45_wine_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "46_WPBC_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "47_yeast_all" --gpu_num 1 
python calculate_AUC_light_all.py --dataset_name "16_http_all" --gpu_num 1 #
python calculate_AUC_light_all.py --dataset_name "33_skin_all" --gpu_num 1 #


python calculate_AUC_light_all.py --dataset_name "MNIST-C_brightness_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_canny_edges_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_dotted_line_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_fog_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_glass_blur_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_identity_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_impulse_noise_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_motion_blur_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_rotate_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_scale_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_shear_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_shot_noise_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_spatter_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_stripe_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_translate_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MNIST-C_zigzag_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"

python calculate_AUC_light_all.py --dataset_name "agnews_0_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "agnews_1_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "agnews_2_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "agnews_3_all" --gpu_num 1 --batch_size 512 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --filter_net_name "AD_RoBERTa512_mlp_vae_gaussian"
