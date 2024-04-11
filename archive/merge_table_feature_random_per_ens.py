import pandas as pd
import os

import numpy as np


dataset_list = [
                'annthyroid', 'breastw', 'cover', 'glass', 'ionosphere', 'letter', 'mammography', 'musk', 'optdigits',
                'pendigits', 'pima', 'speech', 'vertebral', 'vowels', 'wbc', 'arrhythmia', 'cardio', 'satellite', 
                'satimage-2', 'shuttle', 'thyroid',
                '1_ALOI', '3_backdoor', '5_campaign', '7_Cardiotocography', 
                '8_celeba', '9_census', '11_donors', '13_fraud', '19_landsat', '22_magic.gamma', 
                '27_PageBlocks', '33_skin', '35_SpamBase', '41_Waveform'
               ]

result_dir_path = '/home/x1112480/ODIM/Results'
tr_file_name_list = [
                'ODIM_light512_%s_mlp_vae_gaussian_random_per_ens_train_result.csv'
                ]
ts_file_name_list = [
                'ODIM_light512_%s_mlp_vae_gaussian_random_per_ens_test_result.csv'
                ]

col_name_list = ['ODIM_random_per_ens']

total_tr_result_list = []
total_ts_result_list = []
for dataset in dataset_list:
    total_tr_result = []
    total_ts_result = []
    for file_idx in range(len(tr_file_name_list)):
        tr_file_name = tr_file_name_list[file_idx]
        ts_file_name = ts_file_name_list[file_idx]
        if '%s' in tr_file_name:
            tr_file_name = tr_file_name % (dataset.replace('_random',''))
            ts_file_name = ts_file_name % (dataset.replace('_random',''))
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
            result = train_template.copy()
            result.columns = col_name + '_' + result.columns
            total_tr_result = pd.concat([total_tr_result,result], axis = 1)
            print(f'Error Train result {dataset}')

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
                test_template = pd.DataFrame(columns = result.columns, index = result.index)
            result.columns = col_name + '_' + result.columns
            try:
                total_ts_result = pd.concat([total_ts_result,result], axis = 1)
            except:
                total_ts_result = result
        except:
            result = test_template.copy()
            result.columns = col_name + '_' + result.columns
            total_ts_result = pd.concat([total_ts_result,result], axis = 1)
            print(f'Error test result {dataset}')
    
    total_tr_result_list.append(total_tr_result)
    total_ts_result_list.append(total_ts_result)
        

total_tr_result_list = pd.concat(total_tr_result_list)
total_ts_result_list = pd.concat(total_ts_result_list)


colname = total_tr_result_list.columns
tr_reorder_colname = []
for i in range(len(colname)):
    if'_train_auc' in colname[i]:
        tr_reorder_colname.append(colname[i])
        
for i in range(len(colname)):
    if'_train_auc' not in colname[i]:
        tr_reorder_colname.append(colname[i])
        
colname = total_ts_result_list.columns.str.replace('train','test')
total_ts_result_list.columns = colname
ts_reorder_colname = []
for i in range(len(colname)):
    if'_test_auc' in colname[i]:
        ts_reorder_colname.append(colname[i])
        
for i in range(len(colname)):
    if'_test_auc' not in colname[i]:
        ts_reorder_colname.append(colname[i])


total_tr_result_list[tr_reorder_colname].to_csv(os.path.join(result_dir_path,'train_tabular_random_per_ens_summary.csv'))
total_ts_result_list[ts_reorder_colname].to_csv(os.path.join(result_dir_path,'test_tabular_random_per_ens_summary.csv'))
