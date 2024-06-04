from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np


class AdBench_Random_Dataset(Dataset):
    """
    ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    urls = {
        'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=1',
        'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1',
        'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=1',
        'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=1',
        'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=1',
        'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1'
    }

    def __init__(self, root: str, dataset_name: str, train=True, feature_range = (0,1), random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        # if isinstance(root, torch._six.string_classes):
        #     root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        self.file_name = self.dataset_name + '.npz'
        self.data_file = self.root / self.file_name


        mat = np.load(self.data_file, allow_pickle=True)
        
        X = mat['X']
        y = mat['y'].ravel()
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


        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler(feature_range = feature_range).fit(X_train)
        X_train_scaled = minmax_scaler.transform(X_train)
        random_minus = np.random.choice(2,size = X_train_scaled.shape[1],p = [0.5,0.5])
        X_train_scaled = X_train_scaled - random_minus
        X_test_scaled = minmax_scaler.transform(X_test)
        X_test_scaled = X_test_scaled - random_minus
        
        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
            self.indices = torch.tensor(idx_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
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

    def download(self):
        """Download the ODDS dataset if it doesn't exist in root already."""

        if self._check_exists():
            return

        # download file
        download_url(self.urls[self.dataset_name], self.root, self.file_name)

        print('Done!')
