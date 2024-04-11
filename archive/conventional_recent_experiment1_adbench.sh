#!/bin/bash

python caculate_AUC_conventional_recent.py --dataset_name "1_ALOI" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "3_backdoor" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "5_campaign" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "7_Cardiotocography" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "8_celeba" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "9_census" --gpu_num 0
