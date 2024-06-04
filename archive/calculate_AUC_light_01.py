import click
import torch
import logging
import random
import numpy as np
import time

from datasets.main import load_dataset
from optim.prop_trainer import *
import matplotlib.pyplot as plt
from PIL import Image
import os


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score


import argparse

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS 
# # # # # # # # # # #

parser = argparse.ArgumentParser(description='ODIM Experiment')
# arguments for optimization
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--filter_net_name', type=str, default='')
parser.add_argument('--data_path', type=str, default='')


args = parser.parse_args()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if __name__ == "__main__":
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # parameter setting
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    use_cuda = args.use_cuda
    gpu_num = args.gpu_num
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    ratio_known_normal = 0.0
    ratio_known_outlier = 0.0
    n_known_outlier_classes = 0
    
    data_path = '../data'
    if dataset_name == 'mnist':
        train_option = 'IWAE_alpha1._binarize'
        filter_net_name = 'mnist_mlp_vae'
        ratio_pollution = 0.01
        normal_class_list = [0,1,2,3,4,5,6,7,8,9]
        patience_thres = 100
    elif dataset_name == 'fmnist':
        train_option = 'IWAE_alpha1._gaussian'
        filter_net_name = 'mnist_mlp_vae_gaussian'
        ratio_pollution = 0.01
        normal_class_list = [0,1,2,3,4,5,6,7,8,9]
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

        torch.cuda.set_device(gpu_num)
        print('Current number of the GPU is %d'%torch.cuda.current_device())


        seed_idx = 0
        nu = 0.1
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
        train_auc_list = []
        train_ap_list = []
        test_auc_list = []
        test_ap_list = []
        
        for seed_idx in range(len(data_seed_list)):
            seed = data_seed_list[seed_idx]

            save_metric_dir = f'Results/{dataset_name}'
            os.makedirs(save_metric_dir, exist_ok=True)
            save_dir = os.path.join(f'Results/{dataset_name}/ODIM_light{batch_size}_{filter_net_name}',f'log{seed}')
            os.makedirs(save_dir, exist_ok=True)
            save_score_dir = os.path.join(f'Results/{dataset_name}/ODIM_light{batch_size}_{filter_net_name}',f'score{seed}')
            os.makedirs(save_score_dir, exist_ok=True)
            


            # Set up logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log_file = save_dir + '/log_'+dataset_name+'_trainOption'+ train_option + '_normal' + str(normal_class) +'_pol01.txt'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info('-----------------------------------------------------------------')
            logger.info('-----------------------------------------------------------------')
            # Print paths
            logger.info('Log file is %s' % log_file)
            logger.info('Data path is %s' % data_path)
            # Print experimental setup
            logger.info('Dataset: %s' % dataset_name)
            logger.info('Normal class: %s' % normal_class)
            logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
            logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
            logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
            if n_known_outlier_classes == 1:
                logger.info('Known anomaly class: %d' % known_outlier_class)
            else:
                logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
            logger.info('Network: %s' % filter_net_name)


            # Print model configuration
            logger.info('nu-parameter: %.2f' % nu)

            # Set seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # Load data
            dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                                   ratio_known_normal, ratio_known_outlier, ratio_pollution,
                                   random_state=np.random.RandomState(seed))
            # Log random sample of known anomaly classes if more than 1 class
            if n_known_outlier_classes > 1:
                logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

            # Train Filter model 
            train_loader, test_loader = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)


            ## extract train ys and idxs
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            train_ys = []
            train_idxs = []
            for (_, outputs, idxs) in train_loader:
                train_ys.append(outputs.data.numpy())
                train_idxs.append(idxs.data.numpy())

            train_ys = np.hstack(train_ys)
            train_idxs = np.hstack(train_idxs)
            train_idxs_ys = pd.DataFrame({'idx' : train_idxs,'y' : train_ys})


            test_ys = []
            test_idxs = []
            for (_, outputs, idxs) in test_loader:
                test_ys.append(outputs.data.numpy())
                test_idxs.append(idxs.data.numpy())

            test_ys = np.hstack(test_ys)
            test_idxs = np.hstack(test_idxs)
            test_idxs_ys = pd.DataFrame({'idx' : test_idxs,'y' : test_ys})



            ens_train_me_losses = 0
            ens_st_train_me_losses = 0

            ## patience index
            train_n = train_ys.shape[0]
            check_iter = np.min(np.array([10, (train_n // batch_size)]))
            
            patience = np.ceil(patience_thres / check_iter).astype('int')
            loss_column = ['idx','ens_value','ens_st_value','y']
            for model_iter in range(n_ens):
                model_seed = start_model_seed+(model_iter*10)

                logger.info('Set model seed to %d.' % (model_seed))
                ## step 1
                train_idxs_losses, test_idxs_losses, running_time = odim_light(filter_net_name, train_loader, test_loader, check_iter, patience, model_seed,seed, logger, train_option)
                train_me_losses = (train_idxs_losses.to_numpy())[:,1]
                st_train_me_losses = (train_me_losses - train_me_losses.mean())/train_me_losses.std()
                train_idxs_losses['st_loss'] = st_train_me_losses
                add_label_idx_losses = pd.merge(train_idxs_losses, train_idxs_ys, on ='idx')
                fpr, tpr, thresholds = metrics.roc_curve(np.array(add_label_idx_losses['y']), np.array(add_label_idx_losses['loss']), pos_label=1)
                roc_auc = metrics.auc(fpr, tpr)
                logger.info('\n...Train_AUC value- VAE: %0.4f' %(roc_auc))


                test_me_losses = (test_idxs_losses.to_numpy())[:,1]
                st_test_me_losses = (test_me_losses - test_me_losses.mean())/test_me_losses.std()
                test_idxs_losses['st_loss'] = st_test_me_losses
                add_label_idx_test_losses = pd.merge(test_idxs_losses, test_idxs_ys, on ='idx')
                fpr, tpr, thresholds = metrics.roc_curve(np.array(add_label_idx_test_losses['y']), np.array(add_label_idx_test_losses['loss']), pos_label=1)
                roc_auc = metrics.auc(fpr, tpr)
                logger.info('\n...Test_AUC value- VAE: %0.4f' %(roc_auc))

                if model_iter == 0:
                    ens_loss = add_label_idx_losses
                    ens_loss.columns = loss_column

                    test_ens_loss = add_label_idx_test_losses
                    test_ens_loss.columns = loss_column
                else:
                    merge_data = pd.merge(ens_loss, train_idxs_losses, on = 'idx')
                    merge_data['ens_value'] = merge_data['ens_value'] + merge_data['loss']
                    merge_data['ens_st_value'] = merge_data['ens_st_value'] + merge_data['st_loss']
                    ens_loss = merge_data[loss_column]

                    test_merge_data = pd.merge(test_ens_loss, test_idxs_losses, on = 'idx')
                    test_merge_data['ens_value'] = test_merge_data['ens_value'] + test_merge_data['loss']
                    test_merge_data['ens_st_value'] = test_merge_data['ens_st_value'] + test_merge_data['st_loss']
                    test_ens_loss = test_merge_data[loss_column]


                train_auc = roc_auc_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))
                train_ap = average_precision_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))
                logger.info('\n ...Train_AUC value- Ens VAE: %0.4f' % train_auc)
                logger.info('\n ...Train_PRAUC value- Ens VAE: %0.4f' % train_ap)

                test_auc = roc_auc_score(np.array(test_ens_loss['y']), np.array(test_ens_loss['ens_value']))
                test_ap = average_precision_score(np.array(test_ens_loss['y']), np.array(test_ens_loss['ens_value']))
                logger.info('\n ...Test_AUC value- Ens VAE: %0.4f' %(test_auc))
                logger.info('\n ...Test_PRAUC value- Ens VAE: %0.4f' %(test_ap))

            logger.info('\n ...Final Train_AUC value of Ens VAE: %0.4f' %(train_auc))
            logger.info('\n ...Final Train_PRAUC value of Ens VAE: %0.4f' %(train_ap))
            train_auc_list.append(train_auc)
            train_ap_list.append(train_ap)

            logger.info('\n ...Final Test_AUC value of Ens VAE: %0.4f' %(test_auc))
            logger.info('\n ...Final Test_PRAUC value of Ens VAE: %0.4f' %(test_ap))
            logger.info('Running_time of InlierMem : %.4f' % (running_time))
            test_auc_list.append(test_auc)
            test_ap_list.append(test_ap)
            ens_loss.to_csv(os.path.join(save_score_dir,'score_data_pol01.csv'),index=False)
            test_ens_loss.to_csv(os.path.join(save_score_dir,'test_score_data_pol01.csv'),index=False)
            logger.removeHandler(file_handler)
        
        
        train_auc_list.append(np.mean(train_auc_list))
        train_auc_list.append(np.std(train_auc_list))
        train_ap_list.append(np.mean(train_ap_list))
        train_ap_list.append(np.std(train_ap_list))

        class_train_df = pd.DataFrame({
            'row_names' : row_name_list,
            'train_auc' : train_auc_list,
            'train_ap' : train_ap_list
        })
        class_train_df.set_index(keys = 'row_names', inplace = True)
        try:
            train_df = pd.concat([train_df, class_train_df], axis = 0)
        except:
            train_df = class_train_df

        train_df.to_csv(os.path.join(save_metric_dir,f'ODIM_light{batch_size}_{filter_net_name}_train_result_pol01.csv'))
        
        test_auc_list.append(np.mean(test_auc_list))
        test_auc_list.append(np.std(test_auc_list))
        test_ap_list.append(np.mean(test_ap_list))
        test_ap_list.append(np.std(test_ap_list))

        class_test_df = pd.DataFrame({
            'row_names' : row_name_list,
            'test_auc' : test_auc_list,
            'test_ap' : test_ap_list
        })
        class_test_df.set_index(keys = 'row_names', inplace = True)
        try:
            test_df = pd.concat([test_df, class_test_df], axis = 0)
        except:
            test_df = class_test_df

        test_df.to_csv(os.path.join(save_metric_dir,f'ODIM_light{batch_size}_{filter_net_name}_test_result_pol01.csv'))



            
    
