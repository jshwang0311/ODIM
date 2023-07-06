#!/bin/bash

python calculate_AUC_light.py --dataset_name "3_backdoor" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "5_campaign" --gpu_num 1 --batch_size 512
python calculate_AUC.py --dataset_name "11_donors" --gpu_num 1 


