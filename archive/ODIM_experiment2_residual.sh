#!/bin/bash

python calculate_AUC_light.py --dataset_name "vowels" --gpu_num 1 --batch_size 512
python calculate_AUC.py --dataset_name "vowels" --gpu_num 1
