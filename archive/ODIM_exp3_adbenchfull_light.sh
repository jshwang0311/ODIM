#!/bin/bash


python calculate_AUC_light_all.py --dataset_name "2_annthyroid_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "4_breastw_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "6_cardio_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "14_glass_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "18_Ionosphere_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "20_letter_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "23_mammography_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "25_musk_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "26_optdigits_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "28_pendigits_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "29_Pima_all" --gpu_num 0 
python calculate_AUC_light_all.py --dataset_name "10_cover_all" --gpu_num 0 #


python calculate_AUC_light_all.py --dataset_name "MVTec-AD_bottle_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_cable_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_capsule_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_carpet_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_grid_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_hazelnut_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_leather_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_metal_nut_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_pill_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_screw_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_tile_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_toothbrush_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_transistor_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_wood_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
python calculate_AUC_light_all.py --dataset_name "MVTec-AD_zipper_all" --gpu_num 0 --batch_size 512 --data_path "../ADBench/datasets/CV_by_ViT" --filter_net_name "AD_ViT512_mlp_vae_gaussian"
