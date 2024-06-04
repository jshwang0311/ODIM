from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
#from utils.misc import binary_cross_entropy
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
from PIL import Image
from datasets.mnist import Refine_MNIST_Dataset


import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from networks.main import build_network


def binarize(inputs):
    inputs[inputs > 0.5] = 1.
    inputs[inputs <= 0.5] = 0.
    return inputs



def odim_light(filter_net_name, train_loader, test_loader, check_iter, patience, model_seed,seed, logger, train_option):

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
    for data in train_loader:
        inputs, targets, idx = data
        inputs = inputs.view(inputs.size(0), -1)
        break
    if filter_net_name == 'mnist_mlp_vae_delpix':
        filter_model = build_network(filter_net_name,x_dim=inputs.shape[1])
    elif filter_net_name == 'mnist_mlp_vae_gaussian_delpix':
        filter_model = build_network(filter_net_name,x_dim=inputs.shape[1])
    else:
        filter_model = build_network(filter_net_name)
    filter_model = filter_model.to(device)
    
    filter_optimizer = optim.Adam(filter_model.parameters(), lr=filter_model_lr, weight_decay=weight_decay)
    # Set learning rate scheduler
    filter_scheduler = optim.lr_scheduler.MultiStepLR(filter_optimizer, milestones=(lr_milestone,), gamma=0.1)

    # Training
    logger.info('Starting train filter_model...')
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

            epoch_loss += loss.item()
            normal_epoch_loss += np.array(loss_vec.data.cpu())[np.where(np.array(targets)==0.)[0]].mean()
            abn_epoch_loss += np.array(loss_vec.data.cpu())[np.where(np.array(targets)==1.)[0]].mean()
            n_batches += 1
            mb_idx += 1

            train_loss_list = []
            train_targets_list = []
            idx_list = []
            
            if check_iter != 0:
                check_logic = n_batches % check_iter
            else:
                check_logic = 0
            if check_logic == 0:
                ##### Calculate VAE Score
                filter_model.eval()
                # Calculate loss using only check iter batch.
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
                ##### minmax scaling
                scaler = MinMaxScaler()
                scaler.fit(train_losses)
                train_losses = scaler.transform(train_losses)
                ##### Second Fit GMM
                total_gm = GaussianMixture(n_components=2, random_state=0).fit(train_losses)
                total_means = (total_gm.means_).reshape(-1,)
                total_covs = (total_gm.covariances_).reshape(-1,)
                total_clean_cluster = np.argmin(total_means)
                total_posterior = (total_gm.predict_proba(train_losses))[:,total_clean_cluster]
                posterior_vec = total_posterior
                ##### Third Calculate W-dist
                dist = np.power((total_means[0]-total_means[1]),2) + \
                        np.power((total_covs[0]-total_covs[1]),2)

                if dist > best_dist:
                    patience_idx = 0
                    best_dist = dist
                    i_best_posterior_vec = posterior_vec
                    running_time += (time.time() - start_time)
                    
                    train_loss_list = []
                    train_targets_list = []
                    idx_list = []
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
                    
                    test_loss_list = []
                    test_targets_list = []
                    test_idx_list = []
                    filter_model.eval()
                    for data in test_loader:
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

                        test_loss_list.append(loss_vec.data.cpu())
                        test_targets_list += list(targets.numpy())
                        test_idx_list += list(idx.numpy())
                    start_time = time.time()
                    
                else:
                    patience_idx += 1

                if patience_idx == patience:
                    train_losses = torch.cat(best_train_loss_list,0).numpy().reshape(-1,1)
                    logger.info('update num : %d' % (mb_idx))
                    logger.info('Final Epoch : %d ' % (epoch))
                    print('break!')
                    break


        normal_epoch_loss_list.append(normal_epoch_loss/n_batches)
        abn_epoch_loss_list.append(abn_epoch_loss/n_batches)

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time

        if patience_idx == patience:
            logger.info('best_epoch: %d' % (epoch+1))  
            break

    running_time += (time.time() - start_time)
    idxs_losses = pd.DataFrame({'idx' : best_idx,'loss' : torch.cat(best_train_loss_list,0).numpy()})
    test_idxs_losses = pd.DataFrame({'idx' : test_idx_list,'loss' : torch.cat(test_loss_list,0).numpy()})
    return(idxs_losses, test_idxs_losses, running_time)


def odim_fully_fit(filter_net_name, train_loader, test_loader, check_iter, patience, model_seed,seed, logger, train_option):

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
    
    

    tot_filter_model_n_epoch = 500
    
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
    logger.info('Starting train filter_model...')
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
    for epoch in range(tot_filter_model_n_epoch):
        epoch_loss = 0.0
        normal_epoch_loss = 0.0
        abn_epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        ### Train VAE
        for data in train_loader:
            inputs, targets, _ = data
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

            epoch_loss += loss.item()
            normal_epoch_loss += np.array(loss_vec.data.cpu())[np.where(np.array(targets)==0.)[0]].mean()
            abn_epoch_loss += np.array(loss_vec.data.cpu())[np.where(np.array(targets)==1.)[0]].mean()
            n_batches += 1
            mb_idx += 1

    train_loss_list = []
    train_targets_list = []
    idx_list = []
    ##### Calculate VAE Score
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
    
    test_loss_list = []
    test_targets_list = []
    test_idx_list = []
    filter_model.eval()
    for data in test_loader:
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

        test_loss_list.append(loss_vec.data.cpu())
        test_targets_list += list(targets.numpy())
        test_idx_list += list(idx.numpy())

    idxs_losses = pd.DataFrame({'idx' : best_idx,'loss' : torch.cat(best_train_loss_list,0).numpy()})
    test_idxs_losses = pd.DataFrame({'idx' : test_idx_list,'loss' : torch.cat(test_loss_list,0).numpy()})
    return(idxs_losses, test_idxs_losses)
    


def odim_light_random(filter_net_name, train_loader, test_loader, check_iter, patience, model_seed,seed, logger, train_option):

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
    for data in train_loader:
        inputs, targets, idx = data
        break
    random_minus = np.random.choice(2,size = inputs.shape[1],p = [0.5,0.5])
    
    torch.backends.cudnn.deterministic = True
    
    # Initialize DeepSAD model and set neural network phi
    filter_model = build_network(filter_net_name)
    filter_model = filter_model.to(device)
    
    filter_optimizer = optim.Adam(filter_model.parameters(), lr=filter_model_lr, weight_decay=weight_decay)
    # Set learning rate scheduler
    filter_scheduler = optim.lr_scheduler.MultiStepLR(filter_optimizer, milestones=(lr_milestone,), gamma=0.1)

    # Training
    logger.info('Starting train filter_model...')
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
    for epoch in range(tot_filter_model_n_epoch):
        epoch_loss = 0.0
        normal_epoch_loss = 0.0
        abn_epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        ### Train VAE
        for data in train_loader:
            inputs, targets, idx = data
            inputs = inputs.view(inputs.size(0), -1)
            inputs = inputs - random_minus
            inputs = inputs.to(device)
            inputs = inputs.type(torch.float)
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

            epoch_loss += loss.item()
            normal_epoch_loss += np.array(loss_vec.data.cpu())[np.where(np.array(targets)==0.)[0]].mean()
            abn_epoch_loss += np.array(loss_vec.data.cpu())[np.where(np.array(targets)==1.)[0]].mean()
            n_batches += 1
            mb_idx += 1

            train_loss_list = []
            train_targets_list = []
            idx_list = []
            
            if check_iter != 0:
                check_logic = n_batches % check_iter
            else:
                check_logic = 0
            if check_logic == 0:
                ##### Calculate VAE Score
                filter_model.eval()
                # Calculate loss using only check iter batch.
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
                ##### minmax scaling
                scaler = MinMaxScaler()
                scaler.fit(train_losses)
                train_losses = scaler.transform(train_losses)
                ##### Second Fit GMM
                total_gm = GaussianMixture(n_components=2, random_state=0).fit(train_losses)
                total_means = (total_gm.means_).reshape(-1,)
                total_covs = (total_gm.covariances_).reshape(-1,)
                total_clean_cluster = np.argmin(total_means)
                total_posterior = (total_gm.predict_proba(train_losses))[:,total_clean_cluster]
                posterior_vec = total_posterior
                ##### Third Calculate W-dist
                dist = np.power((total_means[0]-total_means[1]),2) + \
                        np.power((total_covs[0]-total_covs[1]),2)

                if dist > best_dist:
                    patience_idx = 0
                    best_dist = dist
                    i_best_posterior_vec = posterior_vec
                    running_time += (time.time() - start_time)
                    
                    train_loss_list = []
                    train_targets_list = []
                    idx_list = []
                    for data in train_loader:
                        inputs, targets ,idx = data
                        inputs = inputs.view(inputs.size(0), -1)
                        inputs = inputs - random_minus
                        inputs = inputs.to(device)
                        inputs = inputs.type(torch.float)
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
                    
                    test_loss_list = []
                    test_targets_list = []
                    test_idx_list = []
                    filter_model.eval()
                    for data in test_loader:
                        inputs, targets ,idx = data
                        inputs = inputs.view(inputs.size(0), -1)
                        inputs = inputs - random_minus
                        inputs = inputs.to(device)
                        inputs = inputs.type(torch.float)
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

                        test_loss_list.append(loss_vec.data.cpu())
                        test_targets_list += list(targets.numpy())
                        test_idx_list += list(idx.numpy())
                    start_time = time.time()
                    
                else:
                    patience_idx += 1

                if patience_idx == patience:
                    train_losses = torch.cat(best_train_loss_list,0).numpy().reshape(-1,1)
                    logger.info('update num : %d' % (mb_idx))
                    logger.info('Final Epoch : %d ' % (epoch))
                    print('break!')
                    break


        normal_epoch_loss_list.append(normal_epoch_loss/n_batches)
        abn_epoch_loss_list.append(abn_epoch_loss/n_batches)

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time

        if patience_idx == patience:
            logger.info('best_epoch: %d' % (epoch+1))  
            break

    running_time += (time.time() - start_time)
    idxs_losses = pd.DataFrame({'idx' : best_idx,'loss' : torch.cat(best_train_loss_list,0).numpy()})
    test_idxs_losses = pd.DataFrame({'idx' : test_idx_list,'loss' : torch.cat(test_loss_list,0).numpy()})
    return(idxs_losses, test_idxs_losses, running_time)



def logp_z_normal(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Per Sample Normalization method
################################################################################################
def AvgPool_square(inputs,kernel_size,stride):
    padding_size = int(kernel_size/2)
    m = nn.AvgPool2d(kernel_size,stride = stride,padding = padding_size)
    output = m(inputs)
    return output



################################################################################################
## Per Sample Normalization method
################################################################################################
def PerSampleNorm(inputs):
    device = inputs.device
    n_channel = inputs.shape[1]
    for i in range(n_channel):
        channel_data = inputs[:,i,:,:]
        channel_data = channel_data.view(channel_data.size(0), -1)
        #p95 = torch.quantile(channel_data, 0.95, dim=1, keepdim=True)
        #p05 = torch.quantile(channel_data, 0.05, dim=1, keepdim=True)
        p95 = torch.from_numpy(np.quantile(channel_data.cpu().data.numpy(), 0.95, axis = 1,keepdims = True)).float().to(device)
        p05 = torch.from_numpy(np.quantile(channel_data.cpu().data.numpy(), 0.05, axis = 1,keepdims = True)).float().to(device)
        channel_data = (channel_data - p05)/(p95 - p05)
        channel_data = torch.clamp(channel_data, min=0., max = 1.).reshape(inputs[:,i,:,:].shape)
        inputs[:,i,:,:] = channel_data
    return inputs


################################################################################################
## Per Sample Normalization method
################################################################################################
def PerSampleNorm2(inputs):
    device = inputs.device
    n_channel = inputs.shape[1]
    for i in range(n_channel):
        channel_data = inputs[:,i,:,:]
        channel_data = channel_data.view(channel_data.size(0), -1)
        mean_data = torch.mean(channel_data,dim = 1, keepdim = True).float().to(device)
        std_data = torch.std(channel_data,dim = 1,keepdim = True).float().to(device)
        channel_data = (channel_data - mean_data)/std_data + 0.5        
        channel_data = torch.clamp(channel_data, min=0., max = 1.).reshape(inputs[:,i,:,:].shape)
        inputs[:,i,:,:] = channel_data
    return inputs

################################################################################################
## Global Normalization method
################################################################################################
def GlobalNorm(inputs,global_mean,global_std):
    device = inputs.device
    n_channel = inputs.shape[1]
    for i in range(n_channel):
        channel_data = inputs[:,i,:,:]
        channel_data = channel_data.view(channel_data.size(0), -1)
        
        std_data = torch.std(channel_data,dim = 1,keepdim = True).float().to(device)
        mean_data = torch.mean(channel_data,dim = 1, keepdim = True).float().to(device)
        channel_data = (channel_data-mean_data)/std_data
        
        
        #channel_data = (channel_data - global_mean[i])/global_std[i] + 0.5
        channel_data = channel_data*global_std[i] + global_mean[i]
        
        channel_data = torch.clamp(channel_data, min=0., max = 1.).reshape(inputs[:,i,:,:].shape)
        inputs[:,i,:,:] = channel_data
    return inputs


################################################################################################
## Generate background image
################################################################################################
def gen_background(inputs,perturb_rate):
    perturb_pixel = torch.bernoulli(torch.tensor(perturb_rate).repeat(inputs.shape)).to(inputs.device)
    min_input = inputs.min()
    max_input = inputs.max()
    noising = (max_input-min_input)*torch.rand(inputs.shape).to(inputs.device) + min_input
    perturb = perturb_pixel*noising
    inputs = inputs*(1-perturb_pixel)+perturb
    return inputs


################################################################################################
## LogLikelihood Ratio
################################################################################################
def LLR(inputs, back_inputs , vae, back_vae , num_sam):    
    z_mu, z_logvar = vae.encoder(inputs)    
    log_loss_vec = []
    for iter in range(num_sam):
        sample_z , _ = vae.encoder.Sample_Z(z_mu , z_logvar)

        ######################################
        ## calculate weight
        log_p = logp_z_std_mvn(sample_z)
        log_q = logp_z_normal(sample_z , z_mu, z_logvar)

        ######################################
        ## calculate RE
        x_mu = vae.decoder(sample_z)
        x_mu = x_mu.view(x_mu.size(0), -1)
        min_epsilon = 1e-5
        max_epsilon = 1.-1e-5

        x_mu = torch.clamp(x_mu , min=min_epsilon , max=max_epsilon)
        log_recon = (inputs * torch.log(x_mu) + (1.-inputs) * torch.log( 1.-x_mu )).sum(1)

        log_loss = (log_recon + log_p - log_q)
        log_loss_vec.append(log_loss.data.cpu().numpy())
            
    log_loss_vec = np.vstack(log_loss_vec).astype('float64')
    min_loss = np.expand_dims(log_loss_vec.min(axis = 0),0)
    log_loss_vec = log_loss_vec - min_loss
    
    min_loss = min_loss.reshape(-1)
    raw_app_ll = np.log(np.sum(np.exp(log_loss_vec),0)/num_sam) + min_loss
    
    
    z_mu, z_logvar = back_vae.encoder(back_inputs)    
    log_loss_vec = []
    for iter in range(num_sam):
        sample_z , _ = back_vae.encoder.Sample_Z(z_mu , z_logvar)

        ######################################
        ## calculate weight
        log_p = logp_z_std_mvn(sample_z)
        log_q = logp_z_normal(sample_z , z_mu, z_logvar)

        ######################################
        ## calculate RE
        x_mu = back_vae.decoder(sample_z)
        x_mu = x_mu.view(x_mu.size(0), -1)
        min_epsilon = 1e-5
        max_epsilon = 1.-1e-5

        x_mu = torch.clamp(x_mu , min=min_epsilon , max=max_epsilon)
        log_recon = (back_inputs * torch.log(x_mu) + (1.-back_inputs) * torch.log( 1.-x_mu )).sum(1)

        log_loss = (log_recon + log_p - log_q)
        log_loss_vec.append(log_loss.data.cpu().numpy())
            
    log_loss_vec = np.vstack(log_loss_vec).astype('float64')
    min_loss = np.expand_dims(log_loss_vec.min(axis = 0),0)
    log_loss_vec = log_loss_vec - min_loss
    
    min_loss = min_loss.reshape(-1)
    back_app_ll = np.log(np.sum(np.exp(log_loss_vec),0)/num_sam) + min_loss

    return raw_app_ll - back_app_ll, raw_app_ll, back_app_ll

################################################################################################
## Calculate ELBO Ratio
################################################################################################
def Elbo_ratio(inputs, back_inputs , vae, back_vae, alpha):
    z_mu , z_log_var = vae.encoder(inputs)
    sam_z , _ = vae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = vae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z(inputs, x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)
    
    kl_loss = log_p - log_q
    elbo = log_recon + alpha*kl_loss
    
    
    z_mu , z_log_var = back_vae.encoder(back_inputs)
    sam_z , _ = back_vae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = back_vae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z(back_inputs, x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)
    
    kl_loss = log_p - log_q
    back_elbo = log_recon + alpha*kl_loss

        
    return -(elbo-back_elbo), elbo, back_elbo

################################################################################################
## Calculate log p(z)
################################################################################################
def logp_z(z , z_mu_ps , z_log_var_ps , p_cluster):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : n_pseudo * z_dim
    z_ten = z.unsqueeze(1)                                                              ## mini_batch * 1 * z_dim
    z_mu_ps_ten , z_log_var_ps_ten = z_mu_ps.unsqueeze(0) , z_log_var_ps.unsqueeze(0)   ## 1 * n_pseudo * z_dim
    
    logp = (-z_log_var_ps_ten/2-torch.pow((z_ten-z_mu_ps_ten) , 2)/(2*torch.exp(z_log_var_ps_ten))).sum(2) + \
            (torch.log(p_cluster).unsqueeze(0)) ## logp : mini_batch * n_pseudo
    
    max_logp = torch.max(logp , 1)[0]
    #logp = torch.log((torch.exp(logp)).sum(1))
    
    logp = max_logp + torch.log(torch.sum(torch.exp(logp - max_logp.unsqueeze(1)), 1))
    
    return logp

################################################################################################
## Calculate log p(z|x)
################################################################################################

def logp_z_given_x(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector


################################################################################################
## Calculate log p(x|z)
################################################################################################
   
def logp_x_given_z(x , x_mu):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    #logp = -(1/2)*((x-x_mu)**2).sum(1)
    
    return logp

################################################################################################
## Calculate log p(x|z)
################################################################################################
   
def logp_x_given_z_gaussian(x , x_mu):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    #logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    logp = -(1/2)*((x-x_mu)**2).sum(1)
    
    return logp


################################################################################################
## Calculate log p(x|z)
################################################################################################
   
def logp_x_given_z_gaussian_mean_var(x , x_mu, x_logvar):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    #logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    #logp = -(1/2)*((x-x_mu)**2).sum(1)
    logp = (-x_logvar-(1/(2*torch.exp(x_logvar)))*((x-x_mu)**2)).sum(1)
    
    return logp

################################################################################################
## Calculate recon loss for rsrae
################################################################################################

def recon_loss_rsrae_l21(x , x_mu):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    #recon_loss = torch.sum(torch.pow(torch.norm(x - x_mu, p=2), 1))
    recon_loss = torch.sum(torch.norm(x - x_mu, p=2, dim = 1))
    
    return recon_loss

################################################################################################
## Calculate log pdf of normal distribution
## z ~ N(0,1)
################################################################################################

def logp_z_std_mvn(z):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -torch.pow(z , 2)/2
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate log pdf of normal distribution
################################################################################################

def logq_z(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate calculate_center function
################################################################################################
def calculate_center(x , gmvae):   

    z_mu , z_log_var = gmvae.encoder(x)
        
    return z_mu


################################################################################################
## Calculate calculate_center function
################################################################################################
def SVDD_loss(x , gmvae,c):   

    z_mu , z_log_var = gmvae.encoder(x)
    dist = torch.sum((z_mu - c) ** 2, dim=1)
    loss = torch.mean(dist)
    
    return loss,dist


################################################################################################
## Calculate VAE loss function
################################################################################################
def VAE_loss(x , gmvae, alpha):   

    z_mu , z_log_var = gmvae.encoder(x)
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = gmvae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z(x , x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)

    elbo = log_recon + alpha*(log_p - log_q)
        
    return -elbo.mean(),-elbo

################################################################################################
## Calculate VAE two hyper loss function
################################################################################################
def VAE_two_hyper_loss(x , gmvae, alpha,alpha_tilde):   

    z_mu , z_log_var = gmvae.encoder(x)
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = gmvae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z(x , x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)

    elbo = alpha_tilde*log_recon + alpha*(log_p - log_q)
        
    return -elbo.mean(),-elbo

################################################################################################
## Calculate VAE two hyper loss function
################################################################################################
def VAE_two_hyper_loss_gaussian(x , gmvae, alpha,alpha_tilde):   

    z_mu , z_log_var = gmvae.encoder(x)
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = gmvae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)

    elbo = alpha_tilde*log_recon + alpha*(log_p - log_q)
        
    return -elbo.mean(),-elbo

################################################################################################
## Calculate GMVAE loss function
################################################################################################
def GMVAE_two_hyper_loss(x , gmvae , alpha,alpha_tilde):   

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(sam_z)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    
    log_loss = alpha_tilde*log_recon + alpha*(log_p - log_q)

    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate GMVAE loss function + priorReg
################################################################################################
def GMVAE_AddpriorRegMM_two_hyper_loss(x , gmvae , alpha,alpha_tilde,beta):   

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(sam_z)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    
    log_loss = alpha_tilde*log_recon + alpha*(log_p - log_q)
    
    
    dist = torch.zeros(z_mu_ps.shape[0])
    for i in range(z_mu_ps.shape[0]):
        dist[i] = (torch.sum((torch.sum((z_mu_ps[i,:] - z_mu_ps)**2,dim=1) + torch.sum((torch.exp(z_log_var_ps[i,:]/2) - torch.exp(z_log_var_ps/2))**2,dim=1))*p_cluster))
    
    priorReg = dist.max()-dist.min()

    return -log_loss.mean() - beta*priorReg,-log_loss


################################################################################################
## Calculate GMVAE loss function + priorReg
################################################################################################
def GMVAE_AddpriorRegTT_two_hyper_loss(x , gmvae , alpha,alpha_tilde,beta, Top):   

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(sam_z)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    
    log_loss = alpha_tilde*log_recon + alpha*(log_p - log_q)
    
    
    dist = torch.zeros(z_mu_ps.shape[0])
    for i in range(z_mu_ps.shape[0]):
        dist[i] = (torch.sum((torch.sum((z_mu_ps[i,:] - z_mu_ps)**2,dim=1) + torch.sum((torch.exp(z_log_var_ps[i,:]/2) - torch.exp(z_log_var_ps/2))**2,dim=1))*p_cluster))
    
    dist_num = dist.shape[0]
    priorReg = dist.sort()[0][dist_num-Top:dist_num].mean() -  dist.sort()[0][:Top].mean()

    return -log_loss.mean() - beta*priorReg,-log_loss



################################################################################################
## Calculate GMVAE loss function
################################################################################################
def GMVAE_two_hyper_loss_option(x , gmvae , alpha,alpha_tilde,cluster_param_option):   

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    
    if cluster_param_option:
        p_cluster = gmvae.P_cluster()
    else:
        p_cluster = gmvae.P_cluster().data
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(sam_z)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    
    log_loss = alpha_tilde*log_recon + alpha*(log_p - log_q)

    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate GMVAE loss function
################################################################################################
def GMVAE_two_hyper_loss_option_novar(x , gmvae , alpha,alpha_tilde,cluster_param_option):   

    z_mu_ps = gmvae.gen_z_information()
    
    if cluster_param_option:
        p_cluster = gmvae.P_cluster()
    else:
        p_cluster = gmvae.P_cluster().data
    
    z_mu = gmvae.encoder(x)    

    ######################################
    ## calculate weight
    log_p = logp_z(z_mu , z_mu_ps , torch.zeros(z_mu_ps.shape,device = z_mu_ps.device) , p_cluster)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(z_mu)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    
    log_loss = alpha_tilde*log_recon + alpha*log_p

    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def GMVAE_IWAE_two_hyper_loss(x , gmvae , num_sam , num_aggr,alpha,alpha_tilde):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z(x , x_mu_sam_z)
        elbo = alpha_tilde*log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate IWAE loss function
################################################################################################
def GMVAE_IWAE_two_hyper_gaussian_loss(x , gmvae , num_sam , num_aggr,alpha,alpha_tilde):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)
        elbo = alpha_tilde*log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss + prior reg function
################################################################################################
def GMVAE_IWAE_AddpriorRegMM_two_hyper_gaussian_loss(x , gmvae , num_sam , num_aggr,alpha,alpha_tilde, beta):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)
        elbo = alpha_tilde*log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
    

    dist = torch.zeros(z_mu_ps.shape[0])
    for i in range(z_mu_ps.shape[0]):
        dist[i] = (torch.sum((torch.sum((z_mu_ps[i,:] - z_mu_ps)**2,dim=1) + torch.sum((torch.exp(z_log_var_ps[i,:]/2) - torch.exp(z_log_var_ps/2))**2,dim=1))*p_cluster))
    
    priorReg = dist.max()-dist.min()


    return -log_loss.mean() - beta*priorReg ,-log_loss

################################################################################################
## Calculate IWAE loss + prior reg function
################################################################################################
def GMVAE_IWAE_AddpriorRegTT_two_hyper_gaussian_loss(x , gmvae , num_sam , num_aggr,alpha,alpha_tilde, beta, Top):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)
        elbo = alpha_tilde*log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
    
    dist = torch.zeros(z_mu_ps.shape[0])
    for i in range(z_mu_ps.shape[0]):
        dist[i] = (torch.sum((torch.sum((z_mu_ps[i,:] - z_mu_ps)**2,dim=1) + torch.sum((torch.exp(z_log_var_ps[i,:]/2) - torch.exp(z_log_var_ps/2))**2,dim=1))*p_cluster))
    
    dist_num = dist.shape[0]
    priorReg = dist.sort()[0][dist_num-Top:dist_num].mean() -  dist.sort()[0][:Top].mean()


    return -log_loss.mean() - beta*priorReg ,-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def GMVAE_IWAE_two_hyper_gaussian_mean_var_loss(x , gmvae , num_sam , num_aggr,alpha,alpha_tilde):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        #log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)
        elbo = alpha_tilde*log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate IWAE loss function + priorReg
################################################################################################
def GMVAE_IWAE_AddpriorRegMM_two_hyper_gaussian_mean_var_loss(x , gmvae , num_sam , num_aggr,alpha,alpha_tilde, beta):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        #log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)
        elbo = alpha_tilde*log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
    
    

    dist = torch.zeros(z_mu_ps.shape[0])
    for i in range(z_mu_ps.shape[0]):
        dist[i] = (torch.sum((torch.sum((z_mu_ps[i,:] - z_mu_ps)**2,dim=1) + torch.sum((torch.exp(z_log_var_ps[i,:]/2) - torch.exp(z_log_var_ps/2))**2,dim=1))*p_cluster))
    
    priorReg = dist.max()-dist.min()

    return -log_loss.mean() - beta*priorReg,-log_loss


################################################################################################
## Calculate IWAE loss function + priorReg
################################################################################################
def GMVAE_IWAE_AddpriorRegTT_two_hyper_gaussian_mean_var_loss(x , gmvae , num_sam , num_aggr,alpha,alpha_tilde, beta,Top):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        #log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)
        elbo = alpha_tilde*log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
    
    
    dist = torch.zeros(z_mu_ps.shape[0])
    for i in range(z_mu_ps.shape[0]):
        dist[i] = (torch.sum((torch.sum((z_mu_ps[i,:] - z_mu_ps)**2,dim=1) + torch.sum((torch.exp(z_log_var_ps[i,:]/2) - torch.exp(z_log_var_ps/2))**2,dim=1))*p_cluster))
    
    dist_num = dist.shape[0]
    priorReg = dist.sort()[0][dist_num-Top:dist_num].mean() -  dist.sort()[0][:Top].mean()

    return -log_loss.mean() - beta*priorReg,-log_loss


################################################################################################
## Calculate VAMP loss function
################################################################################################
def VAMP_two_hyper_loss(x , gmvae , alpha,alpha_tilde):   

    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(sam_z)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    
    log_loss = alpha_tilde*log_recon + alpha*(log_p - log_q)

    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate VAMP loss function
################################################################################################
def VAMP_IWAE_two_hyper_loss(x , gmvae , num_sam , num_aggr ,  alpha,alpha_tilde):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []
    
    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    
    x=x.repeat(num_aggr,1)
    z_mu, z_log_var = gmvae.encoder(x)
    
    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z(x , x_mu_sam_z)
        elbo = alpha_tilde*log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate AE loss + rsr function
################################################################################################
def RSRAE_loss(x , rsrae, lambda1, lambda2):   

    z_mu = rsrae.encoder(x)
    z = rsrae.rsr(z_mu)
    
    x_mu_sam_z = rsrae.decoder(z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate recon
    recon = recon_loss_rsrae_l21(x , x_mu_sam_z)
    
    
    #######################################
    ## calculate rsrae
    z_hat = rsrae.rsr.A @ z_mu.view(z_mu.size(0), rsrae.rsr.h, 1)
    AtAz = (rsrae.rsr.A.T @ z_hat).squeeze(2)
    term1 = torch.sum(torch.norm(z_mu - AtAz, p=2, dim = 1))

    term2 = torch.sum(torch.square(rsrae.rsr.A @ rsrae.rsr.A.T - torch.eye(rsrae.rsr.latent_dim, device = rsrae.rsr.A.device)))

    rsrae = lambda1 * term1 + lambda2 * term2
        
    return recon + rsrae


################################################################################################
## Calculate VAE two hyper loss function and return detail results 
################################################################################################
def VAE_two_hyper_loss_detail(x , gmvae, alpha,alpha_tilde):   

    z_mu , z_log_var = gmvae.encoder(x)
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = gmvae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z(x , x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)
    
    kl_loss = log_p - log_q

    elbo = alpha_tilde*log_recon + alpha*kl_loss
        
    return -elbo.mean(),-elbo, -log_recon, -kl_loss, z_mu.data.cpu(), z_log_var.data.cpu()

################################################################################################
## Calculate VAE total loss function
################################################################################################
def VAE_total_loss(x , gmvae, alpha):   

    z_mu , z_log_var = gmvae.encoder(x)
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = gmvae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z(x , x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)
    
    kl_loss = log_p - log_q
    elbo = log_recon + alpha*kl_loss
        
    return -elbo.mean(),-elbo,-log_recon,-kl_loss

################################################################################################
## Calculate VAE total loss function
################################################################################################
def VAE_total_loss_gaussian(x , gmvae, alpha):   

    z_mu , z_log_var = gmvae.encoder(x)
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = gmvae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)
    
    kl_loss = log_p - log_q
    elbo = log_recon + alpha*kl_loss
        
    return -elbo.mean(),-elbo,-log_recon,-kl_loss

################################################################################################
## Calculate VAE total loss function and return detail results
################################################################################################
def VAE_total_loss_detail(x , gmvae, alpha):   

    z_mu , z_log_var = gmvae.encoder(x)
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
    x_mu_sam_z = gmvae.decoder(sam_z)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    log_p = logp_z_std_mvn(sam_z)

    log_recon = logp_x_given_z(x , x_mu_sam_z)

    log_q = logq_z(sam_z , z_mu , z_log_var)
    
    kl_loss = log_p - log_q
    elbo = log_recon + alpha*kl_loss
        
    return -elbo.mean(),-elbo,-log_recon,-kl_loss, z_mu.data.cpu(), z_log_var.data.cpu()

################################################################################################
## Calculate AE loss function
################################################################################################
def AE_loss(x , gmvae):   

    z_mu , z_log_var = gmvae.encoder(x)
    x_mu_sam_z = gmvae.decoder(z_mu)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    elbo = logp_x_given_z(x , x_mu_sam_z)
    
        
    return -elbo.mean(),-elbo

################################################################################################
## Calculate AE loss function
################################################################################################
def AE_loss_gaussian(x , gmvae):   

    z_mu , z_log_var = gmvae.encoder(x)
    x_mu_sam_z = gmvae.decoder(z_mu)
    x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0),-1)

    ######################################
    ## calculate elbo
    #elbo = logp_x_given_z(x , x_mu_sam_z)
    elbo = logp_x_given_z_gaussian(x , x_mu_sam_z)
        
    return -elbo.mean(),-elbo

################################################################################################
## Calculate VAMP loss function
################################################################################################
def VAMP_loss(x , gmvae , alpha):   

    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(sam_z)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    
    log_loss = log_recon + alpha*(log_p - log_q)

    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate VAMP loss function
################################################################################################
def VAMP_loss_detail(x , gmvae , alpha):   

    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(sam_z)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    kl_loss = log_p - log_q
    log_loss = log_recon + alpha*kl_loss

    return -log_loss.mean(),-log_loss,-log_recon,-kl_loss

################################################################################################
## Calculate GMVAE loss function
################################################################################################
def GMVAE_loss_detail(x , gmvae , alpha):   

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    ######################################
    ## calculate RE
    x_mu = gmvae.decoder(sam_z)
    x_mu = x_mu.view(x_mu.size(0),-1)
    log_recon = logp_x_given_z(x , x_mu)
    kl_loss = log_p - log_q
    log_loss = log_recon + alpha*kl_loss

    return -log_loss.mean(),-log_loss,-log_recon,-kl_loss

################################################################################################
## Calculate VAMP loss function
################################################################################################
def VAMP_IWAE_loss(x , gmvae , num_sam , num_aggr , alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []
    
    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    
    x=x.repeat(num_aggr,1)
    z_mu, z_log_var = gmvae.encoder(x)
    
    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z(x , x_mu_sam_z)
        elbo = log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate VAMP loss function
################################################################################################
def VAMP_IWAE_pixelcnn_loss(x , gmvae , num_sam , num_aggr , alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []
    
    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    
    x=x.repeat(num_aggr,1)
    z_mu, z_log_var = gmvae.encoder(x)
    
    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(x,sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z(x , x_mu_sam_z)
        elbo = log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def IWAE_loss(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z(x , x_mu_sam_z)
        elbo = log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def IWAE_pixelcnn_loss(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(x,sam_z)
        ##for cifar10
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        log_recon = logp_x_given_z(x , x_mu_sam_z)
        elbo = log_recon + (log_p - log_q)*alpha
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss
################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z(x , x_mu_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian_mean_var_pixelcnn(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(x,sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian_mean_var(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss



################################################################################################
## Calculate VAMP KL loss function
################################################################################################
def VAMP_KL_loss(x , gmvae):   

    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
    log_q = logp_z_given_x(sam_z , z_mu , z_logvar)

    
    log_loss = (log_p - log_q)

    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate VAMP KL loss function
################################################################################################
def VAMP_Ploss(x , gmvae,option):   

    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    if option:
        z_mu_ps,z_log_var_ps,p_cluster = z_mu_ps.data,z_log_var_ps.data,p_cluster.data

    
    z_mu, z_logvar = gmvae.encoder(x)    
    sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

    ######################################
    ## calculate weight
    log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)

    
    log_loss = log_p

    return -log_loss

################################################################################################
## Calculate VAMP KL loss function
################################################################################################
def VAMP_Ploss_ms(x , gmvae, num_sam , num_aggr,option):   
    
    num_iter = int(num_sam / num_aggr)
    ## calculate pseudo input
    pseudo_x = gmvae.Pseudo_input()
    ## calculate mu and log var of pseudo input on embedding space
    z_mu_ps , z_log_var_ps = gmvae.encoder(pseudo_x)
    ## calculate the clustering probability vector
    p_cluster = gmvae.P_cluster()
    if option:
        z_mu_ps,z_log_var_ps,p_cluster = z_mu_ps.data,z_log_var_ps.data,p_cluster.data

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)
    log_loss_list = []
    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_loss_list.append(log_p.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)    

    return -log_loss


################################################################################################
## Calculate KL loss function
################################################################################################
def KL_loss(x , gmvae , num_sam , num_aggr):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_q = logp_z_given_x(sam_z , z_mu , z_log_var)

        kl_div = (log_p-log_q)
        #kl_div = (log_q-log_p)
        log_loss_list.append(kl_div.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate GMVAE loss function
################################################################################################
def GMVAE_loss_old(x , gmvae , num_sam , num_aggr,option):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()
    
    if option:
        z_mu_ps,z_log_var_ps,p_cluster = z_mu_ps.data,z_log_var_ps.data,p_cluster.data

    x=x.repeat(num_aggr,1)

    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        #x_mu_sam_z = gmvae.decoder(sam_z)
        ##for cifar10
        #x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z(sam_z , z_mu_ps , z_log_var_ps , p_cluster)
        log_loss_list.append(log_p.reshape(num_aggr , -1).t())

    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)

    return -log_loss


################################################################################################
## Delete Noise data
################################################################################################
def refine_dataset_method(gmvae,dataset,train_loader,device,threshold):

    tot_data = train_loader.dataset.dataset.data
    subset_data = tot_data[train_loader.dataset.indices,:,:]
    
    tot_label = train_loader.dataset.dataset.targets
    subset_label = tot_label[train_loader.dataset.indices]
    print(len(np.where(np.array(subset_label)>0)[0]))
    print(len(subset_label))
    
    batch_size = 128
    tot_iternum = int(subset_data.shape[0]/batch_size)
    transform = transforms.ToTensor()
    gmvae.eval()
    z_mu_ps , z_log_var_ps = gmvae.gen_z_information()
    p_cluster = gmvae.P_cluster()
    z_mu_ps = z_mu_ps.unsqueeze(0)
    subset_ind = []
    life_label_list = []
    del_label_list = []
    for i in range(tot_iternum):
        start_ind = i*batch_size
        if i==tot_iternum-1:
            end_ind = subset_data.shape[0]
        else:
            end_ind = (i+1)*batch_size
        batch_data = subset_data[range(start_ind,end_ind)]
        batch_data_size = batch_data.size(0)
        batch_data = batch_data.view(batch_data_size, -1)
        batch_data = Image.fromarray(batch_data.numpy(), mode='L')
        batch_data = transform(batch_data)            
        batch_data = batch_data.to(device)
        batch_data = batch_data.view(batch_data_size, -1)
        z_mu , _ = gmvae.encoder(batch_data)
        z_mu = z_mu.unsqueeze(1)
        dist = (((z_mu_ps - z_mu)**2).sum(2))**(1/2)
        train_cluster = torch.argmin(dist,dim=1)
        p_train_cluster = p_cluster[train_cluster]
        
        batch_label = subset_label[range(start_ind,end_ind)]
        bool_ind = (np.asarray((p_train_cluster>threshold).cpu(),dtype=bool))
        
        life_batch_label = batch_label[np.array(range(len(batch_label)))[bool_ind]]
        del_batch_label = batch_label[np.array(range(len(batch_label)))[~bool_ind]]
        if len(life_label_list)==0:
            life_label_list = list(np.array(life_batch_label))
            del_label_list = list(np.array(del_batch_label))
        else:
            life_label_list = life_label_list + list(np.array(life_batch_label))
            del_label_list = del_label_list + list(np.array(del_batch_label))
        
        if len(subset_ind)==0:
            subset_ind = list(np.array(range(start_ind,end_ind))[(np.asarray((p_train_cluster>threshold).cpu(),dtype=bool))])
        else:
            subset_ind = subset_ind + list(np.array(range(start_ind,end_ind))[(np.asarray((p_train_cluster>threshold).cpu(),dtype=bool))])
            
    df_label = pd.DataFrame({'label' : subset_label})
    df_life_label = pd.DataFrame({'label' : life_label_list})
    df_del_label = pd.DataFrame({'label' : del_label_list})
    print(df_label['label'].value_counts())
    print(df_life_label['label'].value_counts())
    print(df_del_label['label'].value_counts())
    print(len(np.where(np.array(subset_label)>0)[0])/len(subset_label))
    print(len(np.where(np.array(life_label_list)>0)[0])/len(life_label_list))
    print(len(np.where(np.array(del_label_list)>0)[0])/len(del_label_list))
    
    
    refine_dataset = Refine_MNIST_Dataset(dataset,subset_ind)
    
    return refine_dataset

################################################################################################
## Approximate p(x;\theta,\phi)
################################################################################################
def Approximate_logp_x(x , z_mu_ps , z_log_var_ps , gmvae , num_sam):    
    '''
    x = inputs ; z_mu_ps =  pseudo_mu ; z_log_var_ps = pseudo_logvar ; gmvae = gmiwae.net 
    '''
    #z_mu, z_logvar = deepkde.encoder(x)    
    z_mu, z_logvar = gmvae.encoder(x)
    z_mu_ps_ten = z_mu_ps.unsqueeze(0)
    z_log_var_ps_ten = z_log_var_ps.unsqueeze(0)
    
    n_pseudo = z_mu_ps.shape[0]

    log_loss_vec = []
    for iter in range(num_sam):
        sample_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

        ######################################
        ## calculate weight
        sample_z_ten = sample_z.unsqueeze(1)

        sample_log_p = (-0.5*(z_log_var_ps_ten+\
                        torch.pow(sample_z_ten-z_mu_ps_ten,2)/torch.exp(z_log_var_ps_ten))).sum(2)-np.log(n_pseudo)

        log_p_max , _ = torch.max(sample_log_p , 1)

        ## calculate log-sum-exp
        log_p = log_p_max + torch.log(torch.sum(torch.exp(sample_log_p - log_p_max.unsqueeze(1)), 1))
        log_q = (-0.5*(z_logvar+torch.pow(sample_z-z_mu,2)/torch.exp(z_logvar))).sum(1)

        ######################################
        ## calculate RE
        x_mu = gmvae.decoder(sample_z)
        x_mu = x_mu.view(x_mu.size(0), -1)
        min_epsilon = 1e-5
        max_epsilon = 1.-1e-5

        x_mu = torch.clamp(x_mu , min=min_epsilon , max=max_epsilon)
        log_recon = (x * torch.log(x_mu) + (1.-x) * torch.log( 1.-x_mu )).sum(1)

        log_loss = log_recon + log_p - log_q
        
        log_loss_vec.append(log_loss.data.cpu().numpy())
            
    log_loss_vec = np.vstack(log_loss_vec).astype('float64')
    
    '''
    min_loss = np.expand_dims(log_loss_vec.min(axis = 0),0)
    mean_loss = log_loss_vec.mean(axis=0)
    log_loss_vec = log_loss_vec - min_loss

    min_loss = min_loss.reshape(-1)
    app_ll = np.log(np.sum(np.exp(log_loss_vec),0)/num_sam) + min_loss
    '''
    max_loss = np.expand_dims(log_loss_vec.max(axis = 0),0)
    mean_loss = log_loss_vec.mean(axis=0)
    log_loss_vec = log_loss_vec - max_loss

    max_loss = max_loss.reshape(-1)
    app_ll = np.log(np.sum(np.exp(log_loss_vec),0)/num_sam) + max_loss

    return app_ll , np.mean(app_ll - mean_loss)

################################################################################################
## Approximate p(x;\theta,\phi)
################################################################################################

def Approximate_logp_x_vae(x , gmvae , num_sam , num_aggr):    
  
    #x = x.repeat(num_aggr,1)
    #num_iter = int(num_sam/num_aggr)
    z_mu, z_logvar = gmvae.encoder(x)    

    #log2 = torch.log(torch.ones_like(z_logvar)*add_var)
    #z_logvar = z_logvar + log2

    log_loss_vec = []
    for iter in range(num_sam):
        sample_z , _ = gmvae.encoder.Sample_Z(z_mu , z_logvar)

        ######################################
        ## calculate weight
        log_p = logp_z_std_mvn(sample_z)
        log_q = logp_z_normal(sample_z , z_mu, z_logvar)

        ######################################
        ## calculate RE
        x_mu = gmvae.decoder(sample_z)
        x_mu = x_mu.view(x_mu.size(0), -1)
        min_epsilon = 1e-5
        max_epsilon = 1.-1e-5

        x_mu = torch.clamp(x_mu , min=min_epsilon , max=max_epsilon)
        log_recon = (x * torch.log(x_mu) + (1.-x) * torch.log( 1.-x_mu )).sum(1)

        log_loss = (log_recon + log_p - log_q)
        
        
        log_loss_vec.append(log_loss.data.cpu().numpy())
            
    log_loss_vec = np.vstack(log_loss_vec).astype('float64')
    min_loss = np.expand_dims(log_loss_vec.min(axis = 0),0)
    mean_loss = log_loss_vec.mean(axis=0)
    log_loss_vec = log_loss_vec - min_loss
    
    min_loss = min_loss.reshape(-1)
    app_ll = np.log(np.sum(np.exp(log_loss_vec),0)/num_sam) + min_loss

    return app_ll , np.mean(app_ll - mean_loss)

