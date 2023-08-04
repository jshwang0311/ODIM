#!/bin/bash



python calculate_AUC_deepSVDD_ens.py --dataset_name "mnist" --gpu_num 0 
python calculate_AUC_deepSVDD_ens.py --dataset_name "fmnist" --gpu_num 0 

python calculate_AUC_deepSVDD_ens.py --dataset_name "13_fraud" --gpu_num 0
python calculate_AUC_deepSVDD_ens.py --dataset_name "19_landsat" --gpu_num 0
python calculate_AUC_deepSVDD_ens.py --dataset_name "22_magic.gamma" --gpu_num 0
python calculate_AUC_deepSVDD_ens.py --dataset_name "27_PageBlocks" --gpu_num 0
python calculate_AUC_deepSVDD_ens.py --dataset_name "33_skin" --gpu_num 0
python calculate_AUC_deepSVDD_ens.py --dataset_name "35_SpamBase" --gpu_num 0
python calculate_AUC_deepSVDD_ens.py --dataset_name "41_Waveform" --gpu_num 0


python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_0"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp" 
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_1"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_2"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_3"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_4"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_5"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_6"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_7"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_8"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "CIFAR10_9"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_brightness"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_canny_edges"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_dotted_line"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_fog"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_glass_blur"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_identity"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_impulse_noise"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_motion_blur"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_rotate"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_scale"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_shear"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_shot_noise"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_spatter"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_stripe"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_translate"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MNIST-C_zigzag"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_bottle"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_cable"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_capsule"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_carpet"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_grid"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_hazelnut"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_leather"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_metal_nut"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_pill"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_screw"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_tile"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_toothbrush"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_transistor"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_wood"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "MVTec-AD_zipper"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_0"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_1"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_2"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_3"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_4"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_5"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_6"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_7"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_8"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "SVHN_9"  --gpu_num 0 --data_path "../ADBench/datasets/CV_by_ResNet18" --net_name "AD_VResNet_mlp"


python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_0"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_1"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_2"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_3"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_4"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_5"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_0"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_1"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_2"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_3"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "amazon"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "imdb"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "yelp"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa256_mlp"


python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_0"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_1"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_2"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_3"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_4"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_5"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_0"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_1"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_2"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_3"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "amazon"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "imdb"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "yelp"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa512_mlp"

python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_0"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_1"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_2"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_3"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_4"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "20news_5"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_0"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_1"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_2"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "agnews_3"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "amazon"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "imdb"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"
python calculate_AUC_deepSVDD_ens.py --dataset_name "yelp"  --gpu_num 0 --data_path "../ADBench/datasets/NLP_by_RoBERTa" --net_name "AD_RoBERTa_mlp"

