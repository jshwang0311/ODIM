import os


def gen_hyperparams(data_root, dataset_name, dataset_name_option):
    
    if (('all' in dataset_name_option) | ('adbench' in dataset_name_option)):
        cand_dataset_list = [
            '1_ALOI', '2_annthyroid', '3_backdoor', '4_breastw', '5_campaign', '6_cardio', '7_Cardiotocography', '8_celeba', '9_census', '10_cover', 
            '11_donors', '12_fault','13_fraud', '14_glass', '15_Hepatitis', '16_http', '17_InternetAds', '18_Ionosphere', '19_landsat', '20_letter', 
            '21_Lymphography', '22_magic.gamma', '23_mammography', '25_musk', '26_optdigits', '27_PageBlocks', '28_pendigits', '29_Pima', '30_satellite', 
            '31_satimage-2', '32_shuttle', '33_skin', '34_smtp', '35_SpamBase', '36_speech', '37_Stamps', '38_thyroid', '39_vertebral', 
            '40_vowels', '41_Waveform', '42_WBC', '43_WDBC', '44_Wilt', '45_wine', '46_WPBC', '47_yeast',
            'CIFAR10_0', 'CIFAR10_1', 'CIFAR10_2', 'CIFAR10_3', 'CIFAR10_4', 'CIFAR10_5', 'CIFAR10_6', 'CIFAR10_7', 'CIFAR10_8', 'CIFAR10_9', 
            'MNIST-C_brightness', 'MNIST-C_canny_edges', 'MNIST-C_dotted_line', 'MNIST-C_fog', 'MNIST-C_glass_blur', 
            'MNIST-C_identity', 'MNIST-C_impulse_noise', 'MNIST-C_motion_blur', 'MNIST-C_rotate', 'MNIST-C_scale',
            'MNIST-C_shear', 'MNIST-C_shot_noise', 'MNIST-C_spatter', 'MNIST-C_stripe', 'MNIST-C_translate', 
            'MNIST-C_zigzag', 'MVTec-AD_bottle', 'MVTec-AD_cable', 'MVTec-AD_capsule', 'MVTec-AD_carpet', 
            'MVTec-AD_grid', 'MVTec-AD_hazelnut', 'MVTec-AD_leather', 'MVTec-AD_metal_nut', 'MVTec-AD_pill',
            'MVTec-AD_screw', 'MVTec-AD_tile', 'MVTec-AD_toothbrush', 'MVTec-AD_transistor', 'MVTec-AD_wood', 'MVTec-AD_zipper',
            'SVHN_0', 'SVHN_1', 'SVHN_2', 'SVHN_3', 'SVHN_4', 'SVHN_5', 'SVHN_6', 'SVHN_7', 'SVHN_8', 'SVHN_9',
            '20news_0', '20news_1', '20news_2', '20news_3', '20news_4', '20news_5', 
            'agnews_0', 'agnews_1', 'agnews_2', 'agnews_3', 
            'amazon', 'imdb', 'yelp',
            # about another embedding space dataset
            'CIFAR10_0', 'CIFAR10_1', 'CIFAR10_2', 'CIFAR10_3', 'CIFAR10_4', 'CIFAR10_5', 'CIFAR10_6', 'CIFAR10_7', 'CIFAR10_8', 'CIFAR10_9', 
            'MNIST-C_brightness', 'MNIST-C_canny_edges', 'MNIST-C_dotted_line', 'MNIST-C_fog', 'MNIST-C_glass_blur', 
            'MNIST-C_identity', 'MNIST-C_impulse_noise', 'MNIST-C_motion_blur', 'MNIST-C_rotate', 'MNIST-C_scale',
            'MNIST-C_shear', 'MNIST-C_shot_noise', 'MNIST-C_spatter', 'MNIST-C_stripe', 'MNIST-C_translate', 
            'MNIST-C_zigzag', 'MVTec-AD_bottle', 'MVTec-AD_cable', 'MVTec-AD_capsule', 'MVTec-AD_carpet', 
            'MVTec-AD_grid', 'MVTec-AD_hazelnut', 'MVTec-AD_leather', 'MVTec-AD_metal_nut', 'MVTec-AD_pill',
            'MVTec-AD_screw', 'MVTec-AD_tile', 'MVTec-AD_toothbrush', 'MVTec-AD_transistor', 'MVTec-AD_wood', 'MVTec-AD_zipper',
            'SVHN_0', 'SVHN_1', 'SVHN_2', 'SVHN_3', 'SVHN_4', 'SVHN_5', 'SVHN_6', 'SVHN_7', 'SVHN_8', 'SVHN_9',
            '20news_0', '20news_1', '20news_2', '20news_3', '20news_4', '20news_5', 
            'agnews_0', 'agnews_1', 'agnews_2', 'agnews_3', 
            'amazon', 'imdb', 'yelp',
        ]
        if dataset_name_option == 'adbench_all':
            for i in range(len(cand_dataset_list)):
                cand_data = cand_dataset_list[i]
                cand_data = cand_data + '_all'
                cand_dataset_list[i] = cand_data
        if dataset_name_option == 'all':
            cand_dataset_list.append('minst')
            cand_dataset_list.append('fminst')
            cand_dataset_list.append('wafer_scale')
                
    else:
        cand_dataset_list = [dataset_name]
    
    data_path_list = []
    train_option_list = []
    filter_net_name_list = []
    ratio_pollution_list = []
    normal_class_list_list = []
    patience_thres_list = []
    check_dataset_list = []
    for dataset in cand_dataset_list:
        if dataset == 'mnist':
            data_path = data_root
            train_option = 'IWAE_alpha1._binarize'
            filter_net_name = 'mnist_mlp_vae'
            ratio_pollution = 0.1
            normal_class_list = [0,1,2,3,4,5,6,7,8,9]
            patience_thres = 100
        elif dataset == 'fmnist':
            data_path = data_root
            train_option = 'IWAE_alpha1._gaussian'
            filter_net_name = 'mnist_mlp_vae_gaussian'
            ratio_pollution = 0.1
            normal_class_list = [0,1,2,3,4,5,6,7,8,9]
            patience_thres = 100
        elif dataset == 'wafer_scale':
            data_path = data_root
            train_option = 'IWAE_alpha1._binarize'
            filter_net_name = 'mnist_mlp_vae'
            ratio_pollution = 0.1
            normal_class_list = [0]
            patience_thres = 100


        elif '1_ALOI' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '1_ALOI_mlp_vae_gaussian'
            ratio_pollution = 0.0304
            normal_class_list = [0]
            patience_thres = 100
        elif '2_annthyroid' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'annthyroid_mlp_vae_gaussian'
            ratio_pollution = 0.07
            normal_class_list = [0]
            patience_thres = 100
        elif '3_backdoor' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '3_backdoor_mlp_vae_gaussian'
            ratio_pollution = 0.0244
            normal_class_list = [0]
            patience_thres = 100
        elif '4_breastw' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'breastw_mlp_vae_gaussian'
            ratio_pollution = 0.35
            normal_class_list = [0]
            patience_thres = 100
        elif '5_campaign' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '5_campaign_mlp_vae_gaussian'
            ratio_pollution = 0.11265
            normal_class_list = [0]
            patience_thres = 100
        elif '6_cardio' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'cardio_mlp_vae_gaussian'
            ratio_pollution = 0.0961
            normal_class_list = [0]
            patience_thres = 100
        elif '7_Cardiotocography' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '7_Cardiotocography_mlp_vae_gaussian'
            ratio_pollution = 0.2204
            normal_class_list = [0]
            patience_thres = 100
        elif '8_celeba' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '8_celeba_mlp_vae_gaussian'
            ratio_pollution = 0.0224
            normal_class_list = [0]
            patience_thres = 100
        elif '9_census' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '9_census_mlp_vae_gaussian'
            ratio_pollution = 0.062
            normal_class_list = [0]
            patience_thres = 100
        elif '10_cover' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name =  'cover_mlp_vae_gaussian'
            ratio_pollution = 0.009
            normal_class_list = [0]
            patience_thres = 100
        elif '11_donors' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '11_donors_mlp_vae_gaussian'
            ratio_pollution = 0.05925
            normal_class_list = [0]
            patience_thres = 100
        elif '12_fault' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '12_fault_mlp_vae_gaussian'
            ratio_pollution = 0.3467
            normal_class_list = [0]
            patience_thres = 100    
        elif '13_fraud' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '13_fraud_mlp_vae_gaussian'
            ratio_pollution = 0.0017
            normal_class_list = [0]
            patience_thres = 100
        elif '14_glass' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '14_glass_mlp_vae_gaussian'
            ratio_pollution = 0.042
            normal_class_list = [0]
            patience_thres = 100
        elif '15_Hepatitis' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '15_Hepatitis_mlp_vae_gaussian'
            ratio_pollution = 0.1625
            normal_class_list = [0]
            patience_thres = 100
            batch_size = 32
        elif '16_http' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '16_http_mlp_vae_gaussian'
            ratio_pollution = 0.003896
            normal_class_list = [0]
            patience_thres = 100
        elif '17_InternetAds' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '17_InternetAds_mlp_vae_gaussian'
            ratio_pollution = 0.1872
            normal_class_list = [0]
            patience_thres = 100    
        elif '18_Ionosphere' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '18_Ionosphere_mlp_vae_gaussian'
            ratio_pollution = 0.36
            normal_class_list = [0]
            patience_thres = 100
        elif '19_landsat' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '19_landsat_mlp_vae_gaussian'
            ratio_pollution = 0.2071
            normal_class_list = [0]
            patience_thres = 100
        elif '20_letter' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'letter_mlp_vae_gaussian'
            ratio_pollution = 0.0625
            normal_class_list = [0]
            patience_thres = 100
        elif '21_Lymphography' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '21_Lymphography_mlp_vae_gaussian'
            ratio_pollution = 0.0405
            normal_class_list = [0]
            patience_thres = 100    
        elif '22_magic.gamma' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '22_magic.gamma_mlp_vae_gaussian'
            ratio_pollution = 0.3516
            normal_class_list = [0]
            patience_thres = 100
        elif '23_mammography' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'mammography_mlp_vae_gaussian'
            ratio_pollution = 0.0232
            normal_class_list = [0]
            patience_thres = 100



        elif '25_musk' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'musk_mlp_vae_gaussian'
            ratio_pollution = 0.0317
            normal_class_list = [0]
            patience_thres = 100
        elif '26_optdigits' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'optdigits_mlp_vae_gaussian'
            ratio_pollution = 0.0288
            normal_class_list = [0]
            patience_thres = 100
        elif '27_PageBlocks' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '27_PageBlocks_mlp_vae_gaussian'
            ratio_pollution = 0.0946
            normal_class_list = [0]
            patience_thres = 100
        elif '28_pendigits' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'pendigits_mlp_vae_gaussian'
            ratio_pollution = 0.0227
            normal_class_list = [0]
            patience_thres = 100
        elif '29_Pima' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'pima_mlp_vae_gaussian'
            ratio_pollution = 0.34
            normal_class_list = [0]
            patience_thres = 100
        elif '30_satellite' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name =  'satellite_mlp_vae_gaussian'
            ratio_pollution = 0.31
            normal_class_list = [0]
            patience_thres = 100
        elif '31_satimage-2' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'satimage-2_mlp_vae_gaussian'
            ratio_pollution = 0.01
            normal_class_list = [0]
            patience_thres = 100
        elif '32_shuttle' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'shuttle_mlp_vae_gaussian'
            ratio_pollution = 0.07
            normal_class_list = [0]
            patience_thres = 100
        elif '33_skin' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '33_skin_mlp_vae_gaussian'
            ratio_pollution = 0.2075
            normal_class_list = [0]
            patience_thres = 100
        elif '34_smtp' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '34_smtp_mlp_vae_gaussian'
            ratio_pollution = 0.0003
            normal_class_list = [0]
            patience_thres = 100    
        elif '35_SpamBase' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '35_SpamBase_mlp_vae_gaussian'
            ratio_pollution = 0.3991
            normal_class_list = [0]
            patience_thres = 100
        elif '36_speech' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'speech_mlp_vae_gaussian'
            ratio_pollution = 0.0165
            normal_class_list = [0]
            patience_thres = 100
        elif '37_Stamps' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '37_Stamps_mlp_vae_gaussian'
            ratio_pollution = 0.0912
            normal_class_list = [0]
            patience_thres = 100         
        elif '38_thyroid' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'thyroid_mlp_vae_gaussian'
            ratio_pollution = 0.02
            normal_class_list = [0]
            patience_thres = 100
        elif '39_vertebral' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'vertebral_mlp_vae_gaussian'
            ratio_pollution = 0.125
            normal_class_list = [0]
            patience_thres = 100
        elif '40_vowels' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = 'vowels_mlp_vae_gaussian'
            ratio_pollution = 0.034
            normal_class_list = [0]
            patience_thres = 100
        elif '41_Waveform' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '41_Waveform_mlp_vae_gaussian'
            ratio_pollution = 0.029
            normal_class_list = [0]
            patience_thres = 100
        elif '42_WBC' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '42_WBC_mlp_vae_gaussian'
            ratio_pollution = 0.0448
            normal_class_list = [0]
            patience_thres = 100
        elif '43_WDBC' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '43_WDBC_mlp_vae_gaussian'
            ratio_pollution = 0.0272
            normal_class_list = [0]
            patience_thres = 100
        elif '44_Wilt' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '44_Wilt_mlp_vae_gaussian'
            ratio_pollution = 0.0533
            normal_class_list = [0]
            patience_thres = 100
        elif '45_wine' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '45_wine_mlp_vae_gaussian'
            ratio_pollution = 0.0775
            normal_class_list = [0]
            patience_thres = 100
        elif '46_WPBC' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '46_WPBC_mlp_vae_gaussian'
            ratio_pollution = 0.2374
            normal_class_list = [0]
            patience_thres = 100
        elif '47_yeast' in dataset:
            data_path = os.path.join(data_root, 'ADBench/datasets/Classical')
            train_option = 'IWAE_alpha1.'
            filter_net_name = '47_yeast_mlp_vae_gaussian'
            ratio_pollution = 0.3416
            normal_class_list = [0]
            patience_thres = 100



        elif (('CIFAR10' in dataset) | ('MNIST-C' in dataset) | ('MVTec-AD' in dataset) | ('SVHN' in dataset)):
            if dataset not in check_dataset_list:
                data_path = os.path.join(data_root, 'ADBench/datasets/CV_by_ViT')
                train_option = 'IWAE_alpha1.'
                filter_net_name = 'AD_ViT512_mlp_vae_gaussian'
                ratio_pollution = 0.05
                normal_class_list = [0]
                patience_thres = 100
            else:
                data_path = os.path.join(data_root, 'ADBench/datasets/CV_by_ResNet18')
                train_option = 'IWAE_alpha1.'
                filter_net_name = 'AD_VResNet_mlp_vae_gaussian'
                ratio_pollution = 0.05
                normal_class_list = [0]
                patience_thres = 100


        elif (('20news' in dataset) | ('agnews' in dataset) | ('amazon' in dataset) | ('imdb' in dataset) | ('yelp' in dataset)):
            if dataset not in check_dataset_list:
                data_path = os.path.join(data_root, 'ADBench/datasets/NLP_by_RoBERTa')
                train_option = 'IWAE_alpha1.'
                filter_net_name = 'AD_RoBERTa512_mlp_vae_gaussian'
                ratio_pollution = 0.05
                normal_class_list = [0]
                patience_thres = 100
            else:
                data_path = os.path.join(data_root, 'ADBench/datasets/NLP_by_BERT')
                train_option = 'IWAE_alpha1.'
                filter_net_name = 'AD_BERT512_mlp_vae_gaussian'
                ratio_pollution = 0.05
                normal_class_list = [0]
                patience_thres = 100 
            
        check_dataset_list.append(dataset)
        data_path_list.append(data_path)
        train_option_list.append(train_option)
        filter_net_name_list.append(filter_net_name)
        ratio_pollution_list.append(ratio_pollution)
        normal_class_list_list.append(normal_class_list)
        patience_thres_list.append(patience_thres)
        
    return cand_dataset_list, data_path_list, train_option_list, filter_net_name_list, ratio_pollution_list, normal_class_list_list, patience_thres_list
    