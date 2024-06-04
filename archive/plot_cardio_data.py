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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats


use_cuda = True
gpu_num = 0
dataset_name = 'cardio'
batch_size = 64
ratio_known_normal = 0.0
ratio_known_outlier = 0.0
n_known_outlier_classes = 0

data_path = '../data'

train_option = 'IWAE_alpha1.'
filter_net_name = 'cardio_mlp_vae_gaussian'
ratio_pollution = 0.09
normal_class_list = [0]
patience_thres = 100
normal_class_idx = 0
        


data_seed_list = [110,120,130,140,150]
start_model_seed = 1234
n_ens = 10

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



seed = data_seed_list[seed_idx]

save_dir = os.path.join(f'Results/{dataset_name}/ODIM_light{batch_size}_{filter_net_name}',f'log{seed}')
os.makedirs(save_dir, exist_ok=True)


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
    
model_iter = 0
model_seed = start_model_seed+(model_iter*10)


if 'VAE_alpha1.' in train_option:
    num_sam = 1
    num_aggr = 1
    alpha = 1.
elif 'IWAE_alpha1.' in train_option:
    #num_sam = 100
    #num_aggr = 50
    # For CNN
    num_sam = 50
    num_aggr = 10
    alpha = 1.
elif 'VAE_alpha100.' in train_option:
    num_sam = 1
    num_aggr = 1
    alpha = 100.
elif 'IWAE_alpha100.' in train_option:
    num_sam = 100
    num_aggr = 50
    alpha = 100.
elif 'VAE_alpha0.01' in train_option:
    num_sam = 1
    num_aggr = 1
    alpha = 0.01
elif 'IWAE_alpha0.01' in train_option:
    num_sam = 100
    num_aggr = 50
    alpha = 0.01


import random
lr = 0.0001
n_epochs = 1000
lr_milestone = 50

weight_decay = 0.5e-6
pretrain = True

ae_lr = 0.001

ae_n_epochs = 350
ae_batch_size = 64

ae_weight_decay = 0.5e-6
ae_optimizer_name = 'adam'
optimizer_name = 'adam'

ae_lr_milestone = 250


cls_optimizer = 'adam'



pre_weight_decay = 0.5e-3
optimizer_name = 'adam'

objective = 'one-class'




device = 'cuda'

filter_model_lr = 0.001



tot_filter_model_n_epoch = 1000

random.seed(model_seed)
np.random.seed(model_seed)
torch.manual_seed(model_seed)
torch.cuda.manual_seed(model_seed)
torch.backends.cudnn.deterministic = True

# Initialize DeepSAD model and set neural network phi
filter_model = build_network(filter_net_name)
filter_model = filter_model.to(device)

filter_optimizer = optim.Adam(filter_model.parameters(), lr=filter_model_lr, weight_decay=weight_decay)
# Set learning rate scheduler
filter_scheduler = optim.lr_scheduler.MultiStepLR(filter_optimizer, milestones=(lr_milestone,), gamma=0.1)

# Training
start_time = time.time()
filter_model.train()

normal_epoch_loss_list = []
abn_epoch_loss_list = []


normal_epoch_loss = 0.0
abn_epoch_loss = 0.0
n_batches = 0
patience_idx = 0
best_dist = 0
mb_idx = 0
start_time = time.time()
running_time = 0.
tot_filter_model_n_epoch = 1000
save_fig_dir = os.path.join('/home/x1112480/ODIM','figs')
os.makedirs(save_fig_dir, exist_ok=True)
for epoch in range(tot_filter_model_n_epoch):
    epoch_loss = 0.0
    normal_epoch_loss = 0.0
    abn_epoch_loss = 0.0
    n_batches = 0
    epoch_start_time = time.time()
    ### Train VAE
    for data in train_loader:
        inputs, targets, idx = data
        inputs = inputs.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        if 'binarize' in train_option:
            inputs = binarize(inputs)
        # Zero the network parameter gradients
        filter_model.train()
        filter_optimizer.zero_grad()
        # Update network parameters via backpropagation: forward + backward + optimize
        if 'pixel' in filter_net_name:
            loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inputs , filter_model , num_sam , num_aggr,alpha)
        elif 'gaussian' in filter_net_name:
            loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var(inputs , filter_model , num_sam , num_aggr,alpha)
        elif 'gaussian' in train_option:
            loss,loss_vec = VAE_IWAE_loss_gaussian(inputs , filter_model , num_sam , num_aggr,alpha)
        else:
            loss,loss_vec = VAE_IWAE_loss(inputs , filter_model , num_sam , num_aggr,alpha)
        loss.backward()
        filter_optimizer.step()

    if (epoch < 5) | (epoch > 995):
        train_loss_list = []
        train_targets_list = []
        idx_list = []
        filter_model.eval()
        for data in train_loader:
            inputs, targets ,idx = data
            inputs = inputs.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            if 'binarize' in train_option:
                inputs = binarize(inputs)
            # Update network parameters via backpropagation: forward + backward + optimize
            if 'pixel' in filter_net_name:
                loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inputs , filter_model , num_sam , num_aggr,1.)
            elif 'gaussian' in filter_net_name:
                loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var(inputs , filter_model , num_sam , num_aggr,1.)
            elif 'gaussian' in train_option:
                loss,loss_vec = VAE_IWAE_loss_gaussian(inputs , filter_model , num_sam , num_aggr,1.)
            else:
                loss,loss_vec = VAE_IWAE_loss(inputs , filter_model , num_sam , num_aggr,1.)
    
            train_loss_list.append(loss_vec.data.cpu())
            train_targets_list += list(targets.numpy())
            idx_list += list(idx.numpy())
    
        train_losses = torch.cat(train_loss_list,0).numpy().reshape(-1,1)
    
        best_train_loss_list = train_loss_list
        best_idx = idx_list
        best_train_targets_list = train_targets_list
    
    
        train_losses = torch.cat(best_train_loss_list,0).numpy().reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(train_losses)
        train_losses = scaler.transform(train_losses)

        
        normal_losses = train_losses[np.where(np.array(best_train_targets_list)==0)[0]]
        abnormal_losses = train_losses[np.where(np.array(best_train_targets_list)==1)[0]]

        ks_scaler = StandardScaler()
        ks_scaler.fit(normal_losses)
        ks_normal_losses = ks_scaler.transform(normal_losses)

        ks_scaler = StandardScaler()
        ks_scaler.fit(abnormal_losses)
        ks_abnormal_losses = ks_scaler.transform(abnormal_losses)
        normal_ks = stats.kstest(ks_normal_losses, stats.norm.cdf)
        abnormal_ks = stats.kstest(ks_abnormal_losses, stats.norm.cdf)

        bins = np.linspace(train_losses.min() , train_losses.max() , 50)
        plt.figure(figsize=(4,4))
        plt.hist(normal_losses, bins, alpha=0.5, color='#070952', label='Normal' , density = True)
        plt.hist(abnormal_losses, bins, alpha=0.5, color='#DDA94B', label='Anomaly' , density = True)
        plt.legend(loc='upper right')
        
        # plt.hist(normal_losses, 30, density=True, histtype='stepfilled', alpha=0.4, label = 'normal')
        # plt.hist(abnormal_losses, 30, density=True, histtype='stepfilled', alpha=0.4, label = 'anomaly')
        # plt.legend()
        plt.savefig(os.path.join(save_fig_dir, f'epoch{epoch}_loss_hist.png'))
        plt.show()
        print(f'normal : {normal_ks}')
        print(f'abnormal : {abnormal_ks}')
