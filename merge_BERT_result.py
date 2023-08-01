import pandas as pd
import os

import numpy as np


dataset_list = [
                '20news_0', '20news_1', '20news_2', '20news_3', '20news_4', '20news_5', 'agnews_0', 'agnews_1', 'agnews_2', 'agnews_3', 'amazon', 'imdb', 'yelp'
               ]

result_dir_path = '/home/x1112480/ODIM/Results'
tr_file_name_list = [
                'ODIM_light512_AD_BERT_mlp_vae_gaussian_train_result.csv',
                'ODIM_light512_AD_BERT256_mlp_vae_gaussian_train_result.csv',
                'ODIM_light512_AD_BERT512_mlp_vae_gaussian_train_result.csv',
                'OneClassSVM_train_result_AD_BERT256.csv',
                'LoF_train_result_AD_BERT256.csv',
                'iF_train_result_AD_BERT256.csv',
                'COPOD_train_result_AD_BERT256.csv',
                'ECOD_train_result_AD_BERT256.csv',
                'deepSVDD_AD_BERT_mlp_train_result.csv',
                'deepSVDD_AD_BERT256_mlp_train_result.csv',
                'deepSVDD_AD_BERT512_mlp_train_result.csv'
                ]
ts_file_name_list = [
                'ODIM_light512_AD_BERT_mlp_vae_gaussian_test_result.csv',
                'ODIM_light512_AD_BERT256_mlp_vae_gaussian_test_result.csv',
                'ODIM_light512_AD_BERT512_mlp_vae_gaussian_test_result.csv',
                'OneClassSVM_test_result_AD_BERT256.csv',
                'LoF_test_result_AD_BERT256.csv',
                'iF_test_result_AD_BERT256.csv',
                'COPOD_test_result_AD_BERT256.csv',
                'ECOD_test_result_AD_BERT256.csv',
                'deepSVDD_AD_BERT_mlp_test_result.csv',
                'deepSVDD_AD_BERT256_mlp_test_result.csv',
                'deepSVDD_AD_BERT512_mlp_test_result.csv'
                ]

col_name_list = ['ODIM','ODIM256','ODIM512','OneClassSVM','LoF','iF','COPOD','ECOD','deepSVDD','deepSVDD256','deepSVDD512']

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
        
total_tr_result_list[tr_reorder_colname].to_csv(os.path.join(result_dir_path,'train_BERT_summary.csv'))
total_ts_result_list[ts_reorder_colname].to_csv(os.path.join(result_dir_path,'test_BERT_summary.csv'))
