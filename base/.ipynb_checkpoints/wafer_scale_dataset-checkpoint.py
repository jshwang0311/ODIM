from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np
import pandas as pd


class WaferDataset(Dataset):
    """
    ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        
        
        self.file_name = 'LSWMD.pkl'
        self.data_file = self.root / self.file_name


        df = pd.read_pickle(self.data_file)
        df['y']=df.failureType
        mapping_type={'Center':1,'Donut':1,'Edge-Loc':1,'Edge-Ring':1,'Loc':1,'Random':1,'Scratch':1,'Near-full':1,'none':0}
        df=df.replace({'y':mapping_type})
        df_withlabel = df[(df['y']>=0) & (df['y']<=1)]
        df_withlabel =df_withlabel.reset_index(drop=True)
        X_list = []
        for i in range(df_withlabel.shape[0]):
            img = df_withlabel.waferMap[i]
            img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            #X_list.append(np.expand_dims(img,0))
            X_list.append(np.expand_dims(img.reshape(-1),0))
        X_raw = np.concatenate(X_list,axis = 0)
        y = np.array(df_withlabel['y'], dtype = 'int64')
        
        minmax_scaler = MinMaxScaler().fit(X_raw)
        X = minmax_scaler.transform(X_raw)

        
        
        '''
        def find_dim(x):
            dim0=np.size(x,axis=0)
            dim1=np.size(x,axis=1)
            return dim0,dim1
        df_withlabel['waferMapDim']=df_withlabel.waferMap.apply(find_dim)
        print(df_withlabel['waferMapDim'].value_counts()[0:20])
        waferDim_unique = pd.unique(df_withlabel.waferMapDim)
        fig, ax = plt.subplots(nrows = 18, ncols = 18, figsize=(30, 30))
        ax = ax.ravel(order='C')
        for i in range(324):
            img = df_withlabel[df_withlabel.waferMapDim==waferDim_unique[i]].reset_index(drop=True).waferMap[0]
            ax[i].imshow(img)
            ax[i].set_title(waferDim_unique[i], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.tight_layout()
        plt.show()         
        #!pip install opencv-python
        fig, ax = plt.subplots(nrows = 18, ncols = 18, figsize=(30, 30))
        ax = ax.ravel(order='C')
        for i in range(324):
            img = df_withlabel[df_withlabel.waferMapDim==waferDim_unique[i]].reset_index(drop=True).waferMap[0]
            resize_image = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            ax[i].imshow(resize_image)
            ax[i].set_title(waferDim_unique[i], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.tight_layout()
        plt.show() 
        '''
        
        #mat = pd.read_pickle(self.data_file)
        #X = mat['X']
        #y = mat['y'].ravel()
        idx_total = np.array(range(X.shape[0]))
        idx_norm = y == 0
        idx_out = y == 1

        # 60% data for training and 40% for testing; keep outlier ratio
        X_train_norm, X_test_norm, y_train_norm, y_test_norm,idx_train_norm,idx_test_norm = train_test_split(X[idx_norm], y[idx_norm],idx_total[idx_norm],
                                                                                test_size=0.4,
                                                                                random_state=random_state)
        X_train_out, X_test_out, y_train_out, y_test_out,idx_train_out,idx_test_out = train_test_split(X[idx_out], y[idx_out],idx_total[idx_out],
                                                                            test_size=0.4,
                                                                            random_state=random_state)
        X_train = np.concatenate((X_train_norm, X_train_out))
        X_test = np.concatenate((X_test_norm, X_test_out))
        y_train = np.concatenate((y_train_norm, y_train_out))
        y_test = np.concatenate((y_test_norm, y_test_out))
        idx_train = np.concatenate((idx_train_norm, idx_train_out))
        idx_test = np.concatenate((idx_test_norm, idx_test_out))

        if self.train:
            self.data = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
            self.indices = torch.tensor(idx_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)
            self.indices = torch.tensor(idx_test, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target,indices = self.data[index], int(self.targets[index]), int(self.semi_targets[index]),int(self.indices[index])

        #return sample, target, semi_target, index
        #return sample, target, index,indices
        return sample, target, indices  ####deepSVDD때문에 2번째것에서 변경

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)


