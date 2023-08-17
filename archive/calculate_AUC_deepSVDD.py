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
from deepSVDD import DeepSVDD

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
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--net_name', type=str, default='')
parser.add_argument('--data_path', type=str, default='')


args = parser.parse_args()

gpu_num = 0
dataset_name = '1_ALOI'
batch_size = 128
# use_cuda = True
# net_name = 'AD_NLP_mlp'
# data_path = '../ADBench/datasets/NLP_by_RoBERTa'
#data_path = '../ADBench/datasets/NLP_by_BERT'
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
    
    pretrain = True
    ae_lr = 0.001
    ae_n_epochs = 500
    ae_batch_size = 128
    ae_weight_decay = 0.5e-3
    ae_optimizer_name = 'adam'
    optimizer_name = 'adam'
    ae_lr_milestone = 250
    lr = 0.001
    n_epochs = 500
    lr_milestone = 0
    weight_decay = 0.5e-6



    cls_optimizer = 'adam'
    pre_weight_decay = 0.5e-3
    optimizer_name = 'adam'

    objective = 'one-class'
    nu = 0.1

    num_threads = 0
    n_jobs_dataloader = 0
    device = 'cuda'

    
    data_path = '../data'
    if dataset_name == 'mnist':
        
        net_name = 'mnist_mlp'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4,5,6,7,8,9]
        
    elif dataset_name == 'fmnist':
        train_option = 'IWAE_alpha1._gaussian'
        net_name = 'mnist_mlp'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4,5,6,7,8,9]
        
    elif dataset_name == 'wafer_scale':
        
        net_name = 'mnist_mlp'
        ratio_pollution = 0.1
        normal_class_list = [0]
        
    elif dataset_name == 'annthyroid':
        
        net_name = 'annthyroid_mlp'
        ratio_pollution = 0.07
        normal_class_list = [0]
        
    elif dataset_name == 'breastw':
        
        net_name = 'breastw_mlp'
        ratio_pollution = 0.35
        normal_class_list = [0]
        
    elif dataset_name == 'cover':
        
        net_name =  'cover_mlp'
        ratio_pollution = 0.009
        normal_class_list = [0]
        
    elif dataset_name == 'glass':
        
        net_name = 'glass_mlp'
        ratio_pollution = 0.042
        normal_class_list = [0]
        
    elif dataset_name == 'ionosphere':
        
        net_name = 'ionosphere_mlp'
        ratio_pollution = 0.36
        normal_class_list = [0]
        
    elif dataset_name == 'letter':
        
        net_name = 'letter_mlp'
        ratio_pollution = 0.0625
        normal_class_list = [0]
        
    elif dataset_name == 'mammography':
        
        net_name = 'mammography_mlp'
        ratio_pollution = 0.0232
        normal_class_list = [0]
        
    elif dataset_name == 'musk':
        
        net_name = 'musk_mlp'
        ratio_pollution = 0.032
        normal_class_list = [0]
        
    elif dataset_name == 'optdigits':
        
        net_name = 'optdigits_mlp'
        ratio_pollution = 0.029
        normal_class_list = [0]
        
    elif dataset_name == 'pendigits':
        
        net_name = 'pendigits_mlp'
        ratio_pollution = 0.0227
        normal_class_list = [0]
        
    elif dataset_name == 'pima':
        
        net_name = 'pima_mlp'
        ratio_pollution = 0.34
        normal_class_list = [0]
        
    elif dataset_name == 'speech':
        
        net_name = 'speech_mlp'
        ratio_pollution = 0.0165
        normal_class_list = [0]
        
    elif dataset_name == 'vertebral':
        
        net_name = 'vertebral_mlp'
        ratio_pollution = 0.125
        normal_class_list = [0]
        
    elif dataset_name == 'vowels':
        
        net_name = 'vowels_mlp'
        ratio_pollution = 0.034
        normal_class_list = [0]
        
    elif dataset_name == 'wbc':
        
        net_name = 'wbc_mlp'
        ratio_pollution = 0.056
        normal_class_list = [0]
        
    elif dataset_name == 'arrhythmia':
        
        net_name = 'arrhythmia_mlp'
        ratio_pollution = 0.14
        normal_class_list = [0]
        
    elif dataset_name == 'cardio':
        
        net_name = 'cardio_mlp'
        ratio_pollution = 0.09
        normal_class_list = [0]
        
    elif dataset_name == 'satellite':
        
        net_name =  'satellite_mlp'
        ratio_pollution = 0.31
        normal_class_list = [0]
        
    elif dataset_name == 'satimage-2':
        
        net_name = 'satimage-2_mlp'
        ratio_pollution = 0.01
        normal_class_list = [0]
        
    elif dataset_name == 'shuttle':
        
        net_name = 'shuttle_mlp'
        ratio_pollution = 0.07
        normal_class_list = [0]
        
    elif dataset_name == 'thyroid':
        
        net_name = 'thyroid_mlp'
        ratio_pollution = 0.02
        normal_class_list = [0]
        
    elif dataset_name == 'reuters':
        
        net_name = 'reuters_mlp'
        ratio_pollution = 0.1
        normal_class_list = [0,1,2,3,4]
        patience_thres = 300
    

    elif dataset_name == '1_ALOI':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '1_ALOI_mlp'
        ratio_pollution = 0.0304
        normal_class_list = [0]
        
    elif dataset_name == '3_backdoor':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '3_backdoor_mlp'
        ratio_pollution = 0.0244
        normal_class_list = [0]
        
    elif dataset_name == '5_campaign':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '5_campaign_mlp'
        ratio_pollution = 0.11265
        normal_class_list = [0]
        
    elif dataset_name == '7_Cardiotocography':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '7_Cardiotocography_mlp'
        ratio_pollution = 0.2204
        normal_class_list = [0]
        
    elif dataset_name == '8_celeba':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '8_celeba_mlp'
        ratio_pollution = 0.0224
        normal_class_list = [0]
        
    elif dataset_name == '9_census':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '9_census_mlp'
        ratio_pollution = 0.062
        normal_class_list = [0]
        
    elif dataset_name == '11_donors':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '11_donors_mlp'
        ratio_pollution = 0.05925
        normal_class_list = [0]
        
    elif dataset_name == '13_fraud':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '13_fraud_mlp'
        ratio_pollution = 0.0017
        normal_class_list = [0]
        
    elif dataset_name == '19_landsat':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '19_landsat_mlp'
        ratio_pollution = 0.2071
        normal_class_list = [0]
        
    elif dataset_name == '22_magic.gamma':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '22_magic.gamma_mlp'
        ratio_pollution = 0.3516
        normal_class_list = [0]
        
    elif dataset_name == '27_PageBlocks':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '27_PageBlocks_mlp'
        ratio_pollution = 0.0946
        normal_class_list = [0]
        
    elif dataset_name == '33_skin':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '33_skin_mlp'
        ratio_pollution = 0.2075
        normal_class_list = [0]
        
    elif dataset_name == '35_SpamBase':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '35_SpamBase_mlp'
        ratio_pollution = 0.3991
        normal_class_list = [0]
        
    elif dataset_name == '41_Waveform':
        data_path = '../ADBench/datasets/Classical'
        
        net_name = '41_Waveform_mlp'
        ratio_pollution = 0.029
        normal_class_list = [0]
        
    elif 'CIFAR10' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        
    elif 'MNIST-C' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        
    elif 'MVTec-AD' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        
    elif 'SVHN' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        

        
    elif '20news' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
            
    elif 'agnews' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
            
    elif 'amazon' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
        
    elif 'imdb' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
          
    elif 'yelp' in dataset_name:
        data_path = args.data_path
        
        net_name = args.net_name
        ratio_pollution = 0.05
        normal_class_list = [0]
          
        
    
        


    data_seed_list = [110,120,130,140,150]
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
            save_dir = os.path.join(f'Results/{dataset_name}/deepSVDD_{net_name}',f'log{seed}')
            os.makedirs(save_dir, exist_ok=True)
            save_score_dir = os.path.join(f'Results/{dataset_name}/deepSVDD_{net_name}',f'score{seed}')
            os.makedirs(save_score_dir, exist_ok=True)
            


            # Set up logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log_file = save_dir + '/log_'+dataset_name+'_deepSVDD_normal' + str(normal_class) +'.txt'
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
            logger.info('Network: %s' % net_name)


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
            
            # Define deepSVDD
            deep_SVDD = DeepSVDD(objective, nu)
            deep_SVDD.set_network(net_name)
            
            
            running_time = time.time()
            deep_SVDD.pretrain(dataset,
               optimizer_name=ae_optimizer_name,
               lr=ae_lr,
               n_epochs=ae_n_epochs,
               lr_milestones=(ae_lr_milestone,),
               batch_size=ae_batch_size,
               weight_decay=ae_weight_decay,
               device=device,
               n_jobs_dataloader=n_jobs_dataloader)
            
            
            # Train model on dataset
            deep_SVDD.train(dataset,
                            optimizer_name=optimizer_name,
                            lr=lr,
                            n_epochs=n_epochs,
                            lr_milestones=(lr_milestone,),
                            batch_size=batch_size,
                            weight_decay=weight_decay,
                            device=device,
                            n_jobs_dataloader=n_jobs_dataloader)

            running_time = time.time() - running_time
            # Test model
            deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)


            train_auc_list.append(deep_SVDD.results['train_auc'])
            train_ap_list.append(deep_SVDD.results['train_ap'])

            
            logger.info('Running_time of DeepSVDD : %.4f' % (running_time))
            test_auc_list.append(deep_SVDD.results['test_auc'])
            test_ap_list.append(deep_SVDD.results['test_ap'])
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

        train_df.to_csv(os.path.join(save_metric_dir,f'deepSVDD_{net_name}_train_result.csv'))
        
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

        test_df.to_csv(os.path.join(save_metric_dir,f'deepSVDD_{net_name}_test_result.csv'))



            
    
