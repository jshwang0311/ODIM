#!/bin/bash

python caculate_AUC_conventional.py --dataset_name "1_ALOI" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "0_backdoor" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "5_campaign" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "7_Cardiotocography" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "8_celeba" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "9_census" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "11_donors" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "10_fraud" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "19_landsat" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "22_magic.gamma" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "27_PageBlocks" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "00_skin" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "05_SpamBase" --gpu_num 0
python caculate_AUC_conventional.py --dataset_name "41_Waveform" --gpu_num 0

