import pandas as pd
import os

import numpy as np


dataset_list = [
                'fmnist'
               ]

result_dir_path = '/home/x1112480/ODIM/Results'
tr_file_name_list = [
                'ODIM_light_mnist_mlp_vae_gaussian_pat10_train_result.csv',
                'ODIM_light_mnist_mlp_vae_gaussian_pat20_train_result.csv',
                'ODIM_light_mnist_mlp_vae_gaussian_pat50_train_result.csv',
                'ODIM_light_mnist_mlp_vae_gaussian_pat100_train_result.csv',
                'ODIM_light_mnist_mlp_vae_gaussian_pat150_train_result.csv',
                #'ODIM_light_mnist_mlp_vae_gaussian_pat200_train_result.csv'
                ]
ts_file_name_list = [
                'ODIM_light_mnist_mlp_vae_gaussian_pat10_test_result.csv',
                'ODIM_light_mnist_mlp_vae_gaussian_pat20_test_result.csv',
                'ODIM_light_mnist_mlp_vae_gaussian_pat50_test_result.csv',
                'ODIM_light_mnist_mlp_vae_gaussian_pat100_test_result.csv',
                'ODIM_light_mnist_mlp_vae_gaussian_pat150_test_result.csv',
                #'ODIM_light_mnist_mlp_vae_gaussian_pat200_test_result.csv'
                ]

col_name_list = ['ODIM_pat10','ODIM_pat20','ODIM_pat50','ODIM_pat100','ODIM_pat150']

total_tr_result_list = []
total_ts_result_list = []
for dataset in dataset_list:
    total_tr_result = []
    total_ts_result = []
    for file_idx in range(len(tr_file_name_list)):
        tr_file_name = tr_file_name_list[file_idx]
        ts_file_name = ts_file_name_list[file_idx]
        col_name = col_name_list[file_idx]
        try:
            file_name = os.path.join(result_dir_path, dataset,tr_file_name)
            result = pd.read_csv(file_name)
            result['row_names'] = result['row_names'].str.replace('Class', dataset)
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
            result['row_names'] = result['row_names'].str.replace('Class', dataset)
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

total_tr_result_list[tr_reorder_colname].to_csv(os.path.join(result_dir_path,'train_fmnist_pat_summary.csv'))
total_ts_result_list[ts_reorder_colname].to_csv(os.path.join(result_dir_path,'test_fmnist_pat_summary.csv'))
