import pandas as pd
import os

import numpy as np

nudge_list = ['a','b','c','d','e','f']
nudge_unknown_id_list = ['a', 'b']
nudge_known_id_list = ['c', 'd', 'e', 'f']
num_picks = 3
random_choice_threshold = 0.3
random_choice_n = np.random.binomial(num_picks,random_choice_threshold)
exploit_choice_n = num_picks - random_choice_n

if num_picks >= len(nudge_list):
    # all return
else:
    if random_choice_n > len(nudge_unknown_id_list):
        rest_n = random_choice_n - len(nudge_unknown_id_list)
        random_choice_n = len(nudge_unknown_id_list)
        exploit_choice_n += rest_n
    elif exploit_choice_n > len(nudge_known_id_list):
        rest_n = exploit_choice_n - len(nudge_known_id_list)
        exploit_choice_n = len(nudge_known_id_list)
        random_choice_n += rest_n
    
    # random_choice_n 만큼 random으로 pick!
    # exploit_choice_n 만큼  optimizer에서 pick!




dataset_list = [
                '1_ALOI', '3_backdoor', '5_campaign', '7_Cardiotocography', 
                '8_celeba', '9_census', '11_donors', '13_fraud', '19_landsat', '22_magic.gamma', 
                '27_PageBlocks', '33_skin', '35_SpamBase', '41_Waveform'
               ]

result_dir_path = '/home/x1112480/ODIM/Results'
tr_file_name = 'ODIM_light512_train_result.csv'
ts_file_name = 'ODIM_light512_test_result.csv'

total_tr_result = []
total_ts_result = []
for dataset in dataset_list:
    try:
        file_name = os.path.join(result_dir_path, dataset,tr_file_name)
        result = pd.read_csv(file_name)
        result['row_names'] = result['row_names'].str.replace('Class0', dataset)
        total_tr_result.append(result)
    except:
        print(f'Error Train result {dataset}')
    
    try:
        file_name = os.path.join(result_dir_path, dataset,tr_file_name)
        result = pd.read_csv(file_name)
        result['row_names'] = result['row_names'].str.replace('Class0', dataset)
        total_ts_result.append(result)
    except:
        print(f'Error test result {dataset}')
        

total_tr_result = pd.concat(total_tr_result, ignore_index = True)
total_ts_result = pd.concat(total_ts_result, ignore_index = True)
