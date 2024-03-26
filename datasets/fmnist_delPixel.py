from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random


class FashionMNIST_delPixel_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # FashionMNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyFashionMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform,
                                   download=True)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        self.ori_train_set = train_set
        self.ori_train_indices = idx
        
        data_std = torch.std(train_set.data[idx,:].to(torch.float32),dim=0)
        nonzero_std_idx = torch.where(data_std!=0.)
        filter_data = []
        for i in range(train_set.data.shape[0]):
            filter_data.append(train_set.data[i,:][nonzero_std_idx].unsqueeze(0))
        filter_data = torch.concat(filter_data,0)
        train_set.data = filter_data

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        test_set = MyFashionMNIST(root=self.root, train=False, transform=transform,
                                       target_transform=target_transform, download=True)
        filter_data = []
        for i in range(test_set.data.shape[0]):
            filter_data.append(test_set.data[i,:][nonzero_std_idx].unsqueeze(0))
        filter_data = torch.concat(filter_data,0)
        test_set.data = filter_data
        self.test_set = test_set


class Refine_FashionMNIST_Dataset(TorchvisionDataset):

    def __init__(self, mnist,subset_indices = None):
        super().__init__(mnist)

        # Define normal and outlier classes
        self.n_classes = mnist.n_classes
        self.normal_classes = mnist.normal_classes
        self.outlier_classes = mnist.outlier_classes
        self.known_outlier_classes = mnist.known_outlier_classes

        self.test_set = mnist.test_set
        if subset_indices == None:
            self.train_set = mnist.train_set
        else:
            self.train_set = Subset(mnist.ori_train_set, subset_indices)

class MyFashionMNIST(FashionMNIST):
    """
    Torchvision FashionMNIST class with additional targets for the semi-supervised setting and patch of __getitem__
    method to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyFashionMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MyFashionMNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #return img, target, semi_target, index
        return img, target, index
