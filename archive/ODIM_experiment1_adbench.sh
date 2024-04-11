#!/bin/bash

python calculate_AUC.py --dataset_name "1_ALOI" --gpu_num 3
python calculate_AUC.py --dataset_name "3_backdoor" --gpu_num 3
python calculate_AUC.py --dataset_name "5_campaign" --gpu_num 3
python calculate_AUC.py --dataset_name "7_Cardiotocography" --gpu_num 3
python calculate_AUC.py --dataset_name "8_celebagraphy" --gpu_num 3
python calculate_AUC.py --dataset_name "9_census" --gpu_num 3
python calculate_AUC.py --dataset_name "11_donors" --gpu_num 3
python calculate_AUC.py --dataset_name "13_fraud" --gpu_num 3
python calculate_AUC.py --dataset_name "19_landsat" --gpu_num 3
python calculate_AUC.py --dataset_name "22_magic.gamma" --gpu_num 3
python calculate_AUC.py --dataset_name "27_PageBlocks" --gpu_num 3
python calculate_AUC.py --dataset_name "33_skin" --gpu_num 3
python calculate_AUC.py --dataset_name "35_SpamBase" --gpu_num 3
python calculate_AUC.py --dataset_name "41_Waveform" --gpu_num 3

