import pandas as pd
import os

import numpy as np


dataset_list = [
                '20news_0_all', '20news_1_all', '20news_2_all', '20news_3_all', '20news_4_all', '20news_5_all', 'agnews_0_all', 'agnews_1_all', 'agnews_2_all', 'agnews_3_all', 'amazon_all', 'imdb_all', 'yelp_all'
               ]

result_dir_path = '/home/x1112480/ODIM/Results'
tr_file_name_list = [
                'ODIM_light512_AD_BERT512_mlp_vae_gaussian_train_result.csv',
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
        
        
total_tr_result_list[tr_reorder_colname].to_csv(os.path.join(result_dir_path,'train_BERT_all_summary.csv'))

