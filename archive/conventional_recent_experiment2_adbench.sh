#!/bin/bash

python caculate_AUC_conventional_recent.py --dataset_name "19_landsat" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "22_magic.gamma" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "27_PageBlocks" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "33_skin" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "35_SpamBase" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "41_Waveform" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "11_donors" --gpu_num 0
python caculate_AUC_conventional_recent.py --dataset_name "13_fraud" --gpu_num 0

