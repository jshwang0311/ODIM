from .mnist import MNIST_Dataset
from .fmnist import FashionMNIST_Dataset
from .odds import ODDSADDataset
from .wafer import WaferADDataset
from .wafer_scale import WaferScaleADDataset
from .wafer_scale_deepsvdd import WaferScaleADDataset_deepsvdd
from .svhn import SVHN_Dataset
from .reuters import ReutersADDataset
from .adbench import AdBenchDataset
import os

def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, ZCA_option : bool = False,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fmnist', "reuters", 
                            'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid', 'svhn',
                           'annthyroid', 'breastw', 'cover', 'ecoli', 'glass', 'ionosphere', 'letter', 'lympho', 'mammography', 'musk', 'optdigits', 'pendigits', 'pima', 'speech', 'vertebral', 'vowels', 'wbc', 'wine', 'wafer', 'wafer_scale', 'wafer_scale_deepsvdd',
                           'arrhythmia_notest', 'cardio_notest', 'satellite_notest', 'satimage-2_notest', 'shuttle_notest', 'thyroid_notest',
                       'annthyroid_notest', 'breastw_notest', 'cover_notest', 'ecoli_notest', 'glass_notest', 'ionosphere_notest', 'letter_notest', 'lympho_notest', 'mammography_notest', 'musk_notest', 'optdigits_notest', 'pendigits_notest', 'pima_notest', 'speech_notest', 'vertebral_notest', 'vowels_notest', 'wbc_notest', 'wine_notest',
                            '1_ALOI', '3_backdoor', '5_campaign', '7_Cardiotocography', '8_celeba', '9_census', '11_donors', '13_fraud', '19_landsat', '22_magic.gamma', 
                            '27_PageBlocks', '33_skin', '35_SpamBase', '41_Waveform'
)
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name in ('1_ALOI', '3_backdoor', '5_campaign', '7_Cardiotocography', '8_celeba', '9_census', '11_donors', '13_fraud', '19_landsat', '22_magic.gamma', 
                            '27_PageBlocks', '33_skin', '35_SpamBase', '41_Waveform'):
        dataset = AdBenchDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    
    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path,
                                normal_class=normal_class,
                                known_outlier_class=known_outlier_class,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution)
        
    if dataset_name == 'reuters':
        dataset = ReutersADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state,
                                  normal_class=normal_class)

    if dataset_name == 'fmnist':
        dataset = FashionMNIST_Dataset(root=data_path,
                                       normal_class=normal_class,
                                       known_outlier_class=known_outlier_class,
                                       n_known_outlier_classes=n_known_outlier_classes,
                                       ratio_known_normal=ratio_known_normal,
                                       ratio_known_outlier=ratio_known_outlier,
                                       ratio_pollution=ratio_pollution)
    if dataset_name == 'svhn':
        dataset = SVHN_Dataset(root=os.path.join(data_path,'SVHN'),
                                       normal_class=normal_class,
                                       known_outlier_class=known_outlier_class,
                                       n_known_outlier_classes=n_known_outlier_classes,
                                       ratio_known_normal=ratio_known_normal,
                                       ratio_known_outlier=ratio_known_outlier,
                                       ratio_pollution=ratio_pollution)


    if dataset_name == 'wafer':
        dataset =  WaferADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)
        
        
    if dataset_name == 'wafer_scale':
        dataset =  WaferScaleADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)
    if dataset_name == 'wafer_scale_deepsvdd':
        dataset =  WaferScaleADDataset_deepsvdd(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)


    if dataset_name in ('arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid',
                       'annthyroid', 'breastw', 'cover', 'ecoli', 'glass', 'ionosphere', 'letter', 'lympho', 'mammography', 'musk', 'optdigits', 'pendigits', 'pima', 'speech', 'vertebral', 'vowels', 'wbc', 'wine'):
        dataset = ODDSADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    return dataset
