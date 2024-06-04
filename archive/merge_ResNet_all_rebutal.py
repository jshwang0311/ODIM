import pandas as pd
import os

import numpy as np


dataset_list = [
                'CIFAR10_0_all', 'CIFAR10_1_all', 'CIFAR10_2_all', 'CIFAR10_3_all', 'CIFAR10_4_all', 'CIFAR10_5_all', 'CIFAR10_6_all', 'CIFAR10_7_all', 'CIFAR10_8_all', 'CIFAR10_9_all', 'MNIST-C_brightness_all', 'MNIST-C_canny_edges_all', 'MNIST-C_dotted_line_all', 'MNIST-C_fog_all', 'MNIST-C_glass_blur_all', 'MNIST-C_identity_all', 'MNIST-C_impulse_noise_all', 'MNIST-C_motion_blur_all', 'MNIST-C_rotate_all', 'MNIST-C_scale_all', 'MNIST-C_shear_all', 'MNIST-C_shot_noise_all', 'MNIST-C_spatter_all', 'MNIST-C_stripe_all', 'MNIST-C_translate_all', 'MNIST-C_zigzag_all', 'MVTec-AD_bottle_all', 'MVTec-AD_cable_all', 'MVTec-AD_capsule_all', 'MVTec-AD_carpet_all', 'MVTec-AD_grid_all', 'MVTec-AD_hazelnut_all', 'MVTec-AD_leather_all', 'MVTec-AD_metal_nut_all', 'MVTec-AD_pill_all', 'MVTec-AD_screw_all', 'MVTec-AD_tile_all', 'MVTec-AD_toothbrush_all', 'MVTec-AD_transistor_all', 'MVTec-AD_wood_all', 'MVTec-AD_zipper_all', 'SVHN_0_all', 'SVHN_1_all', 'SVHN_2_all', 'SVHN_3_all', 'SVHN_4_all', 'SVHN_5_all', 'SVHN_6_all', 'SVHN_7_all', 'SVHN_8_all', 'SVHN_9_all'
               ]

result_dir_path = '/home/x1112480/ODIM/Results'
tr_file_name_list = [
                'ODIM_light512_AD_VResNet_mlp_vae_gaussian_train_result.csv',
                ]

col_name_list = ['ODIM']

total_tr_result_list = []
for dataset in dataset_list:
    total_tr_result = []
    for file_idx in range(len(tr_file_name_list)):
        tr_file_name = tr_file_name_list[file_idx]
        col_name = col_name_list[file_idx]
        try:
            file_name = os.path.join(result_dir_path, dataset,tr_file_name)
            result = pd.read_csv(file_name)
            result['row_names'] = result['row_names'].str.replace('Class0', dataset)
            result['row_names'] = result['row_names'].str.replace('Average', dataset+'_Average')
            result = result.set_index('row_names')
            if ('OneClassSVM' in tr_file_name) or ('LoF' in tr_file_name):
                for index in result.index:
                    if ('simulation' in index) or ('Std' in index):
                        result.loc[index] = np.nan
            if file_idx==0:
                train_template = pd.DataFrame(columns = result.columns, index = result.index)
            result.columns = col_name + '_' + result.columns
            try:
                total_tr_result = pd.concat([total_tr_result,result], axis = 1)
            except:
                total_tr_result = result
        except:
            print(f'Error Train result {dataset}')
            continue

    
    total_tr_result_list.append(total_tr_result)
        

total_tr_result_list = pd.concat(total_tr_result_list)


colname = total_tr_result_list.columns
tr_reorder_colname = []
for i in range(len(colname)):
    if'_train_auc' in colname[i]:
        tr_reorder_colname.append(colname[i])
        
for i in range(len(colname)):
    if'_train_auc' not in colname[i]:
        tr_reorder_colname.append(colname[i])
        


total_tr_result_list[tr_reorder_colname].to_csv(os.path.join(result_dir_path,'train_ResNet_all_summary.csv'))
