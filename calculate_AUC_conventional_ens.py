import click
import torch
import logging
import random
import numpy as np
import pandas as pd

from datasets.main import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import os


from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD


import argparse

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS 
# # # # # # # # # # #

parser = argparse.ArgumentParser(description='ODIM Experiment')
# arguments for optimization
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--filter_net_name', type=str, default='')
parser.add_argument('--data_path', type=str, default='')


args = parser.parse_args()

# gpu_num = 0
# dataset_name = 'fmnist'
#gpu_num = 0
#dataset_name = '9_census'
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if __name__ == "__main__":
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # parameter setting
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    use_cuda = args.use_cuda
    gpu_num = args.gpu_num
    dataset_name = args.dataset_name
    ratio_known_normal = 0.0
    ratio_known_outlier = 0.0
    n_known_outlier_classes = 0
    data_path = '../data'
    if dataset_name == 'mnist':
        train_option = 'IWAE_alpha1._binarize'
        filter_net_name = 'mnist_mlp_vae'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4,5,6,7,8,9]
        patience_thres = 100
    elif dataset_name == 'fmnist':
        train_option = 'IWAE_alpha1._gaussian'
        filter_net_name = 'mnist_mlp_vae_gaussian'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4,5,6,7,8,9]
        patience_thres = 100
    elif dataset_name == 'wafer_scale':
        train_option = 'IWAE_alpha1._binarize'
        filter_net_name = 'mnist_mlp_vae'
        ratio_pollution = 0.1
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'annthyroid':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'annthyroid_mlp_vae_gaussian'
        ratio_pollution = 0.07
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'breastw':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'breastw_mlp_vae_gaussian'
        ratio_pollution = 0.35
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'cover':
        train_option = 'IWAE_alpha1.'
        filter_net_name =  'cover_mlp_vae_gaussian'
        ratio_pollution = 0.009
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'glass':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'glass_mlp_vae_gaussian'
        ratio_pollution = 0.042
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'ionosphere':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'ionosphere_mlp_vae_gaussian'
        ratio_pollution = 0.36
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'letter':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'letter_mlp_vae_gaussian'
        ratio_pollution = 0.0625
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'mammography':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'mammography_mlp_vae_gaussian'
        ratio_pollution = 0.0232
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'musk':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'musk_mlp_vae_gaussian'
        ratio_pollution = 0.032
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'optdigits':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'optdigits_mlp_vae_gaussian'
        ratio_pollution = 0.029
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'pendigits':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'pendigits_mlp_vae_gaussian'
        ratio_pollution = 0.0227
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'pima':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'pima_mlp_vae_gaussian'
        ratio_pollution = 0.34
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'speech':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'speech_mlp_vae_gaussian'
        ratio_pollution = 0.0165
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'vertebral':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'vertebral_mlp_vae_gaussian'
        ratio_pollution = 0.125
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'vowels':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'vowels_mlp_vae_gaussian'
        ratio_pollution = 0.034
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'wbc':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'wbc_mlp_vae_gaussian'
        ratio_pollution = 0.056
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'arrhythmia':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'arrhythmia_mlp_vae_gaussian'
        ratio_pollution = 0.14
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'cardio':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'cardio_mlp_vae_gaussian'
        ratio_pollution = 0.09
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'satellite':
        train_option = 'IWAE_alpha1.'
        filter_net_name =  'satellite_mlp_vae_gaussian'
        ratio_pollution = 0.31
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'satimage-2':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'satimage-2_mlp_vae_gaussian'
        ratio_pollution = 0.01
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'shuttle':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'shuttle_mlp_vae_gaussian'
        ratio_pollution = 0.07
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'thyroid':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'thyroid_mlp_vae_gaussian'
        ratio_pollution = 0.02
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == 'reuters':
        train_option = 'IWAE_alpha1.'
        filter_net_name = 'reuters_mlp_vae_256_128_64_gaussian'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4]
        patience_thres = 300
        
    
    
    elif dataset_name == '1_ALOI':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '1_ALOI_mlp_vae_gaussian'
        ratio_pollution = 0.0304
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '3_backdoor':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '3_backdoor_mlp_vae_gaussian'
        ratio_pollution = 0.0244
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '5_campaign':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '5_campaign_mlp_vae_gaussian'
        ratio_pollution = 0.11265
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '7_Cardiotocography':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '7_Cardiotocography_mlp_vae_gaussian'
        ratio_pollution = 0.2204
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '8_celeba':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '8_celeba_mlp_vae_gaussian'
        ratio_pollution = 0.0224
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '9_census':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '9_census_mlp_vae_gaussian'
        ratio_pollution = 0.062
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '11_donors':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '11_donors_mlp_vae_gaussian'
        ratio_pollution = 0.05925
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '13_fraud':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '13_fraud_mlp_vae_gaussian'
        ratio_pollution = 0.0017
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '19_landsat':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '19_landsat_mlp_vae_gaussian'
        ratio_pollution = 0.2071
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '22_magic.gamma':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '22_magic.gamma_mlp_vae_gaussian'
        ratio_pollution = 0.3516
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '27_PageBlocks':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '27_PageBlocks_mlp_vae_gaussian'
        ratio_pollution = 0.0946
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '33_skin':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '33_skin_mlp_vae_gaussian'
        ratio_pollution = 0.2075
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '35_SpamBase':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '35_SpamBase_mlp_vae_gaussian'
        ratio_pollution = 0.3991
        normal_class_list = [0]
        patience_thres = 100
    elif dataset_name == '41_Waveform':
        data_path = '../ADBench/datasets/Classical'
        train_option = 'IWAE_alpha1.'
        filter_net_name = '41_Waveform_mlp_vae_gaussian'
        ratio_pollution = 0.029
        normal_class_list = [0]
        patience_thres = 100
        
    elif 'CIFAR10' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100
    elif 'MNIST-C' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100
    elif 'MVTec-AD' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100
    elif 'SVHN' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100

    
    elif '20news' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100    
    elif 'agnews' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100    
    elif 'amazon' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100
    elif 'imdb' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100  
    elif 'yelp' in dataset_name:
        data_path = args.data_path
        train_option = 'IWAE_alpha1.'
        filter_net_name = args.filter_net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        patience_thres = 100  
        

    data_seed_list = [110,120,130,140,150]
    start_model_seed = 1234
    n_ens = 10
    for normal_class_idx in range(len(normal_class_list)):
        normal_class = normal_class_list[normal_class_idx]
        known_outlier_class = 0


        # Default device to 'cpu' if cuda is not available
        if not torch.cuda.is_available():
            device = 'cpu'
        else:
            torch.cuda.set_device(gpu_num)
            print('Current number of the GPU is %d'%torch.cuda.current_device())


        seed_idx = 0
        nu = 0.1
        batch_size = 100000
        num_threads = 0
        n_jobs_dataloader = 0
        
        row_name_list = []
        for seed_idx in range(len(data_seed_list)):
            row_name = f'Class{normal_class}_simulation{seed_idx+1}'
            row_name_list.append(row_name)
        row_name = f'Average'
        row_name_list.append(row_name)
        row_name = f'Std'
        row_name_list.append(row_name)
        
        oc_train_auc_list = []
        oc_train_ap_list = []
        oc_test_auc_list = []
        oc_test_ap_list = []
        
        lof_train_auc_list = []
        lof_train_ap_list = []
        lof_test_auc_list = []
        lof_test_ap_list = []
        
        if_train_auc_list = []
        if_train_ap_list = []
        if_test_auc_list = []
        if_test_ap_list = []
        
        ecod_train_auc_list = []
        ecod_train_ap_list = []
        ecod_test_auc_list = []
        ecod_test_ap_list = []
        
        copod_train_auc_list = []
        copod_train_ap_list = []
        copod_test_auc_list = []
        copod_test_ap_list = []
        for seed_idx in range(len(data_seed_list)):
            seed = data_seed_list[seed_idx]

            save_metric_dir = f'Results/{dataset_name}'
            os.makedirs(save_metric_dir, exist_ok=True)
            save_dir = os.path.join(f'Results/{dataset_name}/ODIM',f'log{seed}')
            os.makedirs(save_dir, exist_ok=True)
            save_score_dir = os.path.join(f'Results/{dataset_name}/ODIM',f'score{seed}')
            os.makedirs(save_score_dir, exist_ok=True)


            # Set seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # Load data
            dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                                   ratio_known_normal, ratio_known_outlier, ratio_pollution,
                                   random_state=np.random.RandomState(seed))


            # Train Filter model 
            train_loader, test_loader = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)


            ## extract train ys and idxs
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            
            train_set = []
            best_train_targets_list = []
            idx_list = []
            for data in train_loader:
                inputs, targets, idx = data
                inputs = inputs.view(inputs.size(0), -1)
                train_set.append(inputs.numpy())
                best_train_targets_list  =  best_train_targets_list + list(targets.numpy())
                idx_list += list(idx.numpy())

            train_set = np.concatenate(train_set)


            test_set = []
            best_test_targets_list = []
            for data in test_loader:
                inputs, targets, idx = data
                inputs = inputs.view(inputs.size(0), -1)
                test_set.append(inputs.numpy())
                best_test_targets_list  =  best_test_targets_list + list(targets.numpy())


            test_set = np.concatenate(test_set)
            
            # Set up logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log_file = save_dir + '/log_'+dataset_name+'_trainOption'+ train_option + '_normal' + str(normal_class) +'.txt'
            log_file = save_dir + '/log_normal_class'+str(normal_class).replace('[','').replace(']','').replace(', ','_')+'_ecod.txt'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info('-----------------------------------------------------------------')
            logger.info('-----------------------------------------------------------------')

            loss_column = ['y','ens_value','ens_st_value']
            n_ens = 10
            start_model_seed = 1234
            for model_iter in range(n_ens):
                model_seed = start_model_seed+(model_iter*10)
                random.seed(model_seed)
                np.random.seed(model_seed)
                
                clf = ECOD().fit(train_set)
                train_losses = clf.decision_scores_
                st_train_losses = (train_losses - train_losses.mean())/train_losses.std()

                test_losses = clf.decision_function(test_set)
                st_test_losses = (test_losses - test_losses.mean())/test_losses.std()


                if model_iter == 0:
                    ens_loss = pd.DataFrame({'y' : best_train_targets_list, 'ens_value' : train_losses, 'ens_st_value' : st_train_losses})
                    test_ens_loss = pd.DataFrame({'y' : best_test_targets_list, 'ens_value' : test_losses, 'ens_st_value' : st_test_losses})
                else:
                    merge_data = ens_loss
                    merge_data['ens_value'] = merge_data['ens_value'] + train_losses
                    merge_data['ens_st_value'] = merge_data['ens_st_value'] + st_train_losses
                    ens_loss = merge_data

                    test_merge_data = test_ens_loss
                    test_merge_data['ens_value'] = test_merge_data['ens_value'] + test_losses
                    test_merge_data['ens_st_value'] = test_merge_data['ens_st_value'] + st_test_losses
                    test_ens_loss = test_merge_data

                train_auc = roc_auc_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))
                train_ap = average_precision_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))
                test_auc = roc_auc_score(np.array(test_ens_loss['y']), np.array(test_ens_loss['ens_value']))
                test_ap = average_precision_score(np.array(test_ens_loss['y']), np.array(test_ens_loss['ens_value']))
                logger.info('train_auc: %.4f' % train_auc)
                logger.info('train_ap: %.4f' % train_ap)

                logger.info('test_auc: %.4f' % test_auc)
                logger.info('test_ap: %.4f' % test_ap)

            ecod_train_auc_list.append(train_auc)
            ecod_train_ap_list.append(train_ap)

            ecod_test_auc_list.append(test_auc)
            ecod_test_ap_list.append(test_ap)

            logger.removeHandler(file_handler)
            
            
            # Set up logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log_file = save_dir + '/log_normal_class'+str(normal_class).replace('[','').replace(']','').replace(', ','_')+'_copod.txt'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info('-----------------------------------------------------------------')
            logger.info('-----------------------------------------------------------------')

            for model_iter in range(n_ens):
                model_seed = start_model_seed+(model_iter*10)
                random.seed(model_seed)
                np.random.seed(model_seed)

                clf = COPOD().fit(train_set)
                train_losses = clf.decision_scores_
                test_losses = clf.decision_function(test_set)
                
                st_train_losses = (train_losses - train_losses.mean())/train_losses.std()
                st_test_losses = (test_losses - test_losses.mean())/test_losses.std()
                #train_losses = train_losses * (-1)

                if model_iter == 0:
                    ens_loss = pd.DataFrame({'y' : best_train_targets_list, 'ens_value' : train_losses, 'ens_st_value' : st_train_losses})
                    test_ens_loss = pd.DataFrame({'y' : best_test_targets_list, 'ens_value' : test_losses, 'ens_st_value' : st_test_losses})
                else:
                    merge_data = ens_loss
                    merge_data['ens_value'] = merge_data['ens_value'] + train_losses
                    merge_data['ens_st_value'] = merge_data['ens_st_value'] + st_train_losses
                    ens_loss = merge_data

                    test_merge_data = test_ens_loss
                    test_merge_data['ens_value'] = test_merge_data['ens_value'] + test_losses
                    test_merge_data['ens_st_value'] = test_merge_data['ens_st_value'] + st_test_losses
                    test_ens_loss = test_merge_data

                train_auc = roc_auc_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))
                train_ap = average_precision_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))
                test_auc = roc_auc_score(np.array(test_ens_loss['y']), np.array(test_ens_loss['ens_value']))
                test_ap = average_precision_score(np.array(test_ens_loss['y']), np.array(test_ens_loss['ens_value']))
                logger.info('train_auc: %.4f' % train_auc)
                logger.info('train_ap: %.4f' % train_ap)

                logger.info('test_auc: %.4f' % test_auc)
                logger.info('test_ap: %.4f' % test_ap)

            copod_train_auc_list.append(train_auc)
            copod_train_ap_list.append(train_ap)

            copod_test_auc_list.append(test_auc)
            copod_test_ap_list.append(test_ap)
            logger.removeHandler(file_handler)
        
        
        
        ecod_train_auc_list.append(np.mean(ecod_train_auc_list))
        ecod_train_auc_list.append(np.std(ecod_train_auc_list))
        ecod_train_ap_list.append(np.mean(ecod_train_ap_list))
        ecod_train_ap_list.append(np.std(ecod_train_ap_list))
        
        copod_train_auc_list.append(np.mean(copod_train_auc_list))
        copod_train_auc_list.append(np.std(copod_train_auc_list))
        copod_train_ap_list.append(np.mean(copod_train_ap_list))
        copod_train_ap_list.append(np.std(copod_train_ap_list))
        
        ecod_class_train_df = pd.DataFrame({
            'row_names' : row_name_list,
            'train_auc' : ecod_train_auc_list,
            'train_ap' : ecod_train_ap_list
        })
        ecod_class_train_df.set_index(keys = 'row_names', inplace = True)
        try:
            ecod_train_df = pd.concat([ecod_train_df, ecod_class_train_df], axis = 0)
        except:
            ecod_train_df = ecod_class_train_df
        ecod_train_df.to_csv(os.path.join(save_metric_dir,f'ECOD_ens_train_result_{filter_net_name.replace("_mlp_vae_gaussian","")}.csv'))
        
        
        copod_class_train_df = pd.DataFrame({
            'row_names' : row_name_list,
            'train_auc' : copod_train_auc_list,
            'train_ap' : copod_train_ap_list
        })
        copod_class_train_df.set_index(keys = 'row_names', inplace = True)
        try:
            copod_train_df = pd.concat([copod_train_df, copod_class_train_df], axis = 0)
        except:
            copod_train_df = copod_class_train_df
        copod_train_df.to_csv(os.path.join(save_metric_dir,f'COPOD_ens_train_result_{filter_net_name.replace("_mlp_vae_gaussian","")}.csv'))
        
        

        
        #Test result
        ecod_test_auc_list.append(np.mean(ecod_test_auc_list))
        ecod_test_auc_list.append(np.std(ecod_test_auc_list))
        ecod_test_ap_list.append(np.mean(ecod_test_ap_list))
        ecod_test_ap_list.append(np.std(ecod_test_ap_list))
        
        copod_test_auc_list.append(np.mean(copod_test_auc_list))
        copod_test_auc_list.append(np.std(copod_test_auc_list))
        copod_test_ap_list.append(np.mean(copod_test_ap_list))
        copod_test_ap_list.append(np.std(copod_test_ap_list))

        ecod_class_test_df = pd.DataFrame({
            'row_names' : row_name_list,
            'test_auc' : ecod_test_auc_list,
            'test_ap' : ecod_test_ap_list
        })
        ecod_class_test_df.set_index(keys = 'row_names', inplace = True)
        try:
            ecod_test_df = pd.concat([ecod_test_df, ecod_class_test_df], axis = 0)
        except:
            ecod_test_df = ecod_class_test_df
        ecod_test_df.to_csv(os.path.join(save_metric_dir,f'ECOD_ens_test_result_{filter_net_name.replace("_mlp_vae_gaussian","")}.csv'))
        
        
        copod_class_test_df = pd.DataFrame({
            'row_names' : row_name_list,
            'test_auc' : copod_test_auc_list,
            'test_ap' : copod_test_ap_list
        })
        copod_class_test_df.set_index(keys = 'row_names', inplace = True)
        try:
            copod_test_df = pd.concat([copod_test_df, copod_class_test_df], axis = 0)
        except:
            copod_test_df = copod_class_test_df
        copod_test_df.to_csv(os.path.join(save_metric_dir,f'COPOD_ens_test_result_{filter_net_name.replace("_mlp_vae_gaussian","")}.csv'))  
             
        



            
    
