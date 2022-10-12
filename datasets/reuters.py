from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.reuters_dataset import ReutersDataset
from .preprocessing import create_semisupervised_setting

import torch


class ReutersADDataset(BaseADDataset):

    def __init__(self, root: str, dataset_name: str, n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None, normal_class: int = 0):
        super().__init__(root)

        # Define normal and outlier classes
        self.root = root
        self.n_classes = 2  # 0: normal, 1: outlier
        
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 5))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)
        
        
        self.known_outlier_classes = ()

        # Get train set
        train_set = ReutersDataset(root=self.root, dataset_name=dataset_name, train=True, random_state=random_state, outlier_classes = list(self.outlier_classes))

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), (0,),
                                                             (1,), self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        self.ori_train_set = train_set
        self.ori_train_indices = idx

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = ReutersDataset(root=self.root, dataset_name=dataset_name, train=False, random_state=random_state, outlier_classes = list(self.outlier_classes))
    
    def refine_method(self,subset_indices = None):
        if subset_indices != None:
            self.train_set = Subset(self.ori_train_set, subset_indices)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,drop_last = False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=drop_last)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
