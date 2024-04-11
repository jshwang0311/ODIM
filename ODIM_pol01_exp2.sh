#!/bin/bash

python calculate_AUC_light_01.py --dataset_name "mnist" --gpu_num 1
python calculate_AUC_light_01.py --dataset_name "fmnist" --gpu_num 1

