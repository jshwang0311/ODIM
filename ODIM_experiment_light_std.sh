#!/bin/bash


python calculate_AUC_light_std.py --dataset_name "1_ALOI_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "3_backdoor_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "5_campaign_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "7_Cardiotocography_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "8_celeba_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "9_census_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "11_donors_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "13_fraud_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "19_landsat_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "22_magic.gamma_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "27_PageBlocks_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "33_skin_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "35_SpamBase_std" --gpu_num 0 --batch_size 512 
python calculate_AUC_light_std.py --dataset_name "41_Waveform_std" --gpu_num 0 --batch_size 512 

