#!/bin/bash

# python calculate_AUC_deepSVDD.py --dataset_name "cardio" --gpu_num 0 
# python calculate_AUC_deepSVDD.py --dataset_name "cover" --gpu_num 0
# python calculate_AUC_deepSVDD.py --dataset_name "satellite" --gpu_num 0
# python calculate_AUC_deepSVDD.py --dataset_name "satimage-2" --gpu_num 0
# python calculate_AUC_deepSVDD.py --dataset_name "shuttle" --gpu_num 0
# python calculate_AUC_deepSVDD.py --dataset_name "thyroid" --gpu_num 0

python calculate_AUC_deepSVDD.py --dataset_name "speech" --gpu_num 0
python calculate_AUC_deepSVDD.py --dataset_name "vertebral" --gpu_num 0

