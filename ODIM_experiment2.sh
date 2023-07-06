#!/bin/bash

python calculate_AUC_light.py --dataset_name "annthyroid" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "breastw" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "glass" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "ionosphere" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "mammography" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "musk" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "pendigits" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "pima" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "vowels" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "wbc" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "arrhythmia" --gpu_num 1 --batch_size 512
python calculate_AUC_light.py --dataset_name "cardio" --gpu_num 1 --batch_size 512
python calculate_AUC.py --dataset_name "cover" --gpu_num 1
python calculate_AUC.py --dataset_name "satellite" --gpu_num 1
python calculate_AUC.py --dataset_name "satimage-2" --gpu_num 1
python calculate_AUC.py --dataset_name "shuttle" --gpu_num 1
python calculate_AUC.py --dataset_name "thyroid" --gpu_num 1
