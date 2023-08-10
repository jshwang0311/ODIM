from .mnist import MNIST_Dataset
from .fmnist import FashionMNIST_Dataset
from .odds import ODDSADDataset
from .wafer import WaferADDataset
from .wafer_scale import WaferScaleADDataset
from .wafer_scale_deepsvdd import WaferScaleADDataset_deepsvdd
from .svhn import SVHN_Dataset
from .reuters import ReutersADDataset
from .adbench import AdBenchDataset

from .odds_random import ODDSADRandomDataset
from .adbench_random import AdBenchRandomDataset
import os

def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, ZCA_option : bool = False, feature_range = (0,1),
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fmnist', "reuters", 
                            'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid', 'svhn',
                           'annthyroid', 'breastw', 'cover', 'ecoli', 'glass', 'ionosphere', 'letter', 'lympho', 'mammography', 'musk', 'optdigits', 'pendigits', 'pima', 'speech', 'vertebral', 'vowels', 'wbc', 'wine', 'wafer', 'wafer_scale', 'wafer_scale_deepsvdd',
                           'arrhythmia_notest', 'cardio_notest', 'satellite_notest', 'satimage-2_notest', 'shuttle_notest', 'thyroid_notest',
                       'annthyroid_notest', 'breastw_notest', 'cover_notest', 'ecoli_notest', 'glass_notest', 'ionosphere_notest', 'letter_notest', 'lympho_notest', 'mammography_notest', 'musk_notest', 'optdigits_notest', 'pendigits_notest', 'pima_notest', 'speech_notest', 'vertebral_notest', 'vowels_notest', 'wbc_notest', 'wine_notest',
                            '1_ALOI', '3_backdoor', '5_campaign', '7_Cardiotocography', '8_celeba', '9_census', '11_donors', '13_fraud', '19_landsat', '22_magic.gamma', 
                            '27_PageBlocks', '33_skin', '35_SpamBase', '41_Waveform',
                            
                            'arrhythmia_random', 'cardio_random', 'satellite_random', 'satimage-2_random', 'shuttle_random', 'thyroid_random',
                           'annthyroid_random', 'breastw_random', 'cover_random', 'glass_random', 'ionosphere_random', 'letter_random', 'lympho_random', 'mammography_random', 'musk_random', 'optdigits_random', 'pendigits_random', 'pima_random', 'speech_random', 'vertebral_random', 'vowels_random', 'wbc_random',
                            '1_ALOI_random', '3_backdoor_random', '5_campaign_random', '7_Cardiotocography_random', '8_celeba_random', '9_census_random', '11_donors_random', '13_fraud_random', '19_landsat_random', '22_magic.gamma_random', 
                            '27_PageBlocks_random', '33_skin_random', '35_SpamBase_random', '41_Waveform_random',
                            
                            'CIFAR10_0', 'CIFAR10_1', 'CIFAR10_2', 'CIFAR10_3', 'CIFAR10_4', 'CIFAR10_5', 'CIFAR10_6', 'CIFAR10_7', 'CIFAR10_8', 'CIFAR10_9', 'MNIST-C_brightness', 'MNIST-C_canny_edges', 'MNIST-C_dotted_line', 'MNIST-C_fog', 'MNIST-C_glass_blur', 'MNIST-C_identity', 'MNIST-C_impulse_noise', 'MNIST-C_motion_blur', 'MNIST-C_rotate', 'MNIST-C_scale', 'MNIST-C_shear', 'MNIST-C_shot_noise', 'MNIST-C_spatter', 'MNIST-C_stripe', 'MNIST-C_translate', 'MNIST-C_zigzag', 'MVTec-AD_bottle', 'MVTec-AD_cable', 'MVTec-AD_capsule', 'MVTec-AD_carpet', 'MVTec-AD_grid', 'MVTec-AD_hazelnut', 'MVTec-AD_leather', 'MVTec-AD_metal_nut', 'MVTec-AD_pill', 'MVTec-AD_screw', 'MVTec-AD_tile', 'MVTec-AD_toothbrush', 'MVTec-AD_transistor', 'MVTec-AD_wood', 'MVTec-AD_zipper', 'SVHN_0', 'SVHN_1', 'SVHN_2', 'SVHN_3', 'SVHN_4', 'SVHN_5', 'SVHN_6', 'SVHN_7', 'SVHN_8', 'SVHN_9',
                            '20news_0', '20news_1', '20news_2', '20news_3', '20news_4', '20news_5', 'agnews_0', 'agnews_1', 'agnews_2', 'agnews_3', 'amazon', 'imdb', 'yelp'
                            
                            
)
    assert dataset_name in implemented_datasets

    dataset = None
    
    
    if dataset_name in ('1_ALOI_random', '3_backdoor_random', '5_campaign_random', '7_Cardiotocography_random', '8_celeba_random', '9_census_random', '11_donors_random', '13_fraud_random', '19_landsat_random', '22_magic.gamma_random', 
                            '27_PageBlocks_random', '33_skin_random', '35_SpamBase_random', '41_Waveform_random'):
        dataset_name = dataset_name.replace('_random','')
        dataset = AdBenchRandomDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
                                random_state=random_state)

    
    
    if dataset_name in ('arrhythmia_random', 'cardio_random', 'satellite_random', 'satimage-2_random', 'shuttle_random', 'thyroid_random',
                           'annthyroid_random', 'breastw_random', 'cover_random', 'glass_random', 'ionosphere_random', 'letter_random', 'lympho_random', 'mammography_random', 'musk_random', 'optdigits_random', 'pendigits_random', 'pima_random', 'speech_random', 'vertebral_random', 'vowels_random', 'wbc_random'):
        dataset_name = dataset_name.replace('_random','')
        dataset = ODDSADRandomDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
                                random_state=random_state)
        
    
    
    if dataset_name in ('1_ALOI', '3_backdoor', '5_campaign', '7_Cardiotocography', '8_celeba', '9_census', '11_donors', '13_fraud', '19_landsat', '22_magic.gamma', 
                            '27_PageBlocks', '33_skin', '35_SpamBase', '41_Waveform',
                       'CIFAR10_0', 'CIFAR10_1', 'CIFAR10_2', 'CIFAR10_3', 'CIFAR10_4', 'CIFAR10_5', 'CIFAR10_6', 'CIFAR10_7', 'CIFAR10_8', 'CIFAR10_9', 'MNIST-C_brightness', 'MNIST-C_canny_edges', 'MNIST-C_dotted_line', 'MNIST-C_fog', 'MNIST-C_glass_blur', 'MNIST-C_identity', 'MNIST-C_impulse_noise', 'MNIST-C_motion_blur', 'MNIST-C_rotate', 'MNIST-C_scale', 'MNIST-C_shear', 'MNIST-C_shot_noise', 'MNIST-C_spatter', 'MNIST-C_stripe', 'MNIST-C_translate', 'MNIST-C_zigzag', 'MVTec-AD_bottle', 'MVTec-AD_cable', 'MVTec-AD_capsule', 'MVTec-AD_carpet', 'MVTec-AD_grid', 'MVTec-AD_hazelnut', 'MVTec-AD_leather', 'MVTec-AD_metal_nut', 'MVTec-AD_pill', 'MVTec-AD_screw', 'MVTec-AD_tile', 'MVTec-AD_toothbrush', 'MVTec-AD_transistor', 'MVTec-AD_wood', 'MVTec-AD_zipper', 'SVHN_0', 'SVHN_1', 'SVHN_2', 'SVHN_3', 'SVHN_4', 'SVHN_5', 'SVHN_6', 'SVHN_7', 'SVHN_8', 'SVHN_9', 
                        '20news_0', '20news_1', '20news_2', '20news_3', '20news_4', '20news_5', 'agnews_0', 'agnews_1', 'agnews_2', 'agnews_3', 'amazon', 'imdb', 'yelp'):
        dataset = AdBenchDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
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
                                feature_range = feature_range,
                                random_state=random_state)

    return dataset
