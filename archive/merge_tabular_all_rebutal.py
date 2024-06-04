import pandas as pd
import os

import numpy as np


dataset_list = [
                '1_ALOI_all', '3_backdoor_all', '5_campaign_all', '7_Cardiotocography_all', 
                '8_celeba_all', '9_census_all', '11_donors_all', '13_fraud_all', '19_landsat_all', 
                '22_magic.gamma_all', '27_PageBlocks_all', '33_skin_all', '35_SpamBase_all', 
                '41_Waveform_all',
    
                '12_fault_all','15_Hepatitis_all', '16_http_all', '17_InternetAds_all', 
                '21_Lymphography_all', '34_smtp_all', '37_Stamps_all', '43_WDBC_all', 
                '44_Wilt_all', '45_wine_all', '46_WPBC_all', '47_yeast_all',
    
                '2_annthyroid_all', '4_breastw_all', '6_cardio_all', '10_cover_all',
                '14_glass_all', '18_Ionosphere_all', '20_letter_all', '23_mammography_all', 
                '25_musk_all', '26_optdigits_all', '28_pendigits_all', '29_Pima_all', 
                '30_satellite_all', '31_satimage-2_all', '32_shuttle_all', '36_speech_all', 
                '38_thyroid_all', '39_vertebral_all', '40_vowels_all', '42_WBC_all'
               ]

result_dir_path = '/home/x1112480/ODIM/Results'
tr_file_name_list = [
                'ODIM_light64_%s_mlp_vae_gaussian_train_result.csv'
                ]

col_name_list = ['ODIM']

dataset_list.sort()
total_tr_result_list = []
for dataset in dataset_list:
    total_tr_result = []
    for file_idx in range(len(tr_file_name_list)):
        tr_file_name = tr_file_name_list[file_idx]
        if '%s' in tr_file_name:
            if dataset == '15_Hepatitis_all':
                tr_file_name = 'ODIM_light32_%s_mlp_vae_gaussian_train_result.csv'
                tr_file_name = tr_file_name % (dataset.replace('_all',''))
            else:
                tr_file_name = tr_file_name % (dataset.replace('_all',''))
        col_name = col_name_list[file_idx]
        try:
            file_name = os.path.join(result_dir_path, dataset,tr_file_name)
            result = pd.read_csv(file_name)
        except:
            try:
                tr_file_name = tr_file_name_list[file_idx]
                tr_file_name = tr_file_name % (dataset.split('_')[1])
                file_name = os.path.join(result_dir_path, dataset,tr_file_name)
                result = pd.read_csv(file_name)
            except:
                try:
                    tr_file_name = tr_file_name_list[file_idx]
                    tr_file_name = tr_file_name % (dataset.split('_')[1].lower())
                    file_name = os.path.join(result_dir_path, dataset,tr_file_name)
                    result = pd.read_csv(file_name)
                except:
                    print(f'Error Train result {dataset}')
                    continue
            
        result['row_names'] = result['row_names'].str.replace('Class0', dataset)
        result['row_names'] = result['row_names'].str.replace('Average', dataset+'_Average')
        result = result.set_index('row_names')
        if file_idx==0:
            train_template = pd.DataFrame(columns = result.columns, index = result.index)
        result.columns = col_name + '_' + result.columns
        if ('OneClassSVM' in tr_file_name) or ('LoF' in tr_file_name):
            for index in result.index:
                if ('simulation' in index) or ('Std' in index):
                    result.loc[index] = np.nan
        try:
            total_tr_result = pd.concat([total_tr_result,result], axis = 1)
        except:
            total_tr_result = result

    if len(total_tr_result) != 0:
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
        

total_tr_result_list[tr_reorder_colname].to_csv(os.path.join(result_dir_path,'train_tabular_All_rebutal_summary.csv'))
