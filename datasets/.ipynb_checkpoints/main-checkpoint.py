from .mnist import MNIST_Dataset
from .mnist_delPixel import MNIST_delPixel_Dataset
from .fmnist import FashionMNIST_Dataset
from .fmnist_delPixel import FashionMNIST_delPixel_Dataset
from .odds import ODDSADDataset
from .wafer import WaferADDataset
from .wafer_scale import WaferScaleADDataset
from .wafer_scale_deepsvdd import WaferScaleADDataset_deepsvdd
from .svhn import SVHN_Dataset
from .reuters import ReutersADDataset
from .adbench import AdBenchDataset
from .adbench_all import AdBench_All_Dataset

from .odds_random import ODDSADRandomDataset
from .odds_std import ODDSADStdDataset
from .adbench_random import AdBenchRandomDataset
from .adbench_std import AdBenchStdDataset
import os

def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, ZCA_option : bool = False, feature_range = (0,1),
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist_delpix','mnist', 'fmnist', 'fmnist_delpix', "reuters", 
                            'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid', 'svhn',
                           'annthyroid', 'breastw', 'cover', 'ecoli', 'glass', 'ionosphere', 'letter', 'lympho', 
                            'mammography', 'musk', 'optdigits', 'pendigits', 'pima', 'speech', 'vertebral', 'vowels', 
                            'wbc', 'wine', 'wafer', 'wafer_scale', 'wafer_scale_deepsvdd',
                           'arrhythmia_notest', 'cardio_notest', 'satellite_notest', 'satimage-2_notest', 'shuttle_notest', 'thyroid_notest',
                           'annthyroid_notest', 'breastw_notest', 'cover_notest', 'ecoli_notest', 'glass_notest', 'ionosphere_notest',
                            'letter_notest', 'lympho_notest', 'mammography_notest', 'musk_notest', 'optdigits_notest', 'pendigits_notest',
                            'pima_notest', 'speech_notest', 'vertebral_notest', 'vowels_notest', 'wbc_notest', 'wine_notest',
                            '1_ALOI', '3_backdoor', '5_campaign', '7_Cardiotocography', '8_celeba', '9_census', '11_donors', '13_fraud', '19_landsat', '22_magic.gamma', 
                            '27_PageBlocks', '33_skin', '35_SpamBase', '41_Waveform',
                            
                            '12_fault','15_Hepatitis', '16_http', '17_InternetAds', '21_Lymphography', '34_smtp', '37_Stamps', '43_WDBC', '44_Wilt', '45_wine', '46_WPBC', '47_yeast',
                            
                            
                            
                            'arrhythmia_random', 'cardio_random', 'satellite_random', 'satimage-2_random', 'shuttle_random',
                            'thyroid_random',
                           'annthyroid_random', 'breastw_random', 'cover_random', 'glass_random', 'ionosphere_random',
                            'letter_random', 'lympho_random', 'mammography_random', 'musk_random', 'optdigits_random', 
                            'pendigits_random', 'pima_random', 'speech_random', 'vertebral_random', 'vowels_random', 'wbc_random',
                            '1_ALOI_random', '3_backdoor_random', '5_campaign_random', '7_Cardiotocography_random', '8_celeba_random',
                            '9_census_random', '11_donors_random', '13_fraud_random', '19_landsat_random', '22_magic.gamma_random', 
                            '27_PageBlocks_random', '33_skin_random', '35_SpamBase_random', '41_Waveform_random',
                            
                            
                            '1_ALOI_all', '3_backdoor_all', '5_campaign_all', '7_Cardiotocography_all', '8_celeba_all', '9_census_all', '11_donors_all', '13_fraud_all', '19_landsat_all', 
                          '22_magic.gamma_all', '27_PageBlocks_all', '33_skin_all', '35_SpamBase_all', '41_Waveform_all',
                         'CIFAR10_0_all', 'CIFAR10_1_all', 'CIFAR10_2_all', 'CIFAR10_3_all', 'CIFAR10_4_all', 'CIFAR10_5_all', 'CIFAR10_6_all', 'CIFAR10_7_all', 
                          'CIFAR10_8_all', 'CIFAR10_9_all', 'MNIST-C_brightness_all', 'MNIST-C_canny_edges_all', 'MNIST-C_dotted_line_all', 'MNIST-C_fog_all', 
                          'MNIST-C_glass_blur_all', 'MNIST-C_identity_all', 'MNIST-C_impulse_noise_all', 'MNIST-C_motion_blur_all', 'MNIST-C_rotate_all', 
                          'MNIST-C_scale_all', 'MNIST-C_shear_all', 'MNIST-C_shot_noise_all', 'MNIST-C_spatter_all', 'MNIST-C_stripe_all', 
                          'MNIST-C_translate_all', 'MNIST-C_zigzag_all', 'MVTec-AD_bottle_all', 'MVTec-AD_cable_all', 'MVTec-AD_capsule_all', 
                          'MVTec-AD_carpet_all', 'MVTec-AD_grid_all', 'MVTec-AD_hazelnut_all', 'MVTec-AD_leather_all', 'MVTec-AD_metal_nut_all',
                          'MVTec-AD_pill_all', 'MVTec-AD_screw_all', 'MVTec-AD_tile_all', 'MVTec-AD_toothbrush_all', 'MVTec-AD_transistor_all', 
                          'MVTec-AD_wood_all', 'MVTec-AD_zipper_all', 'SVHN_0_all', 'SVHN_1_all', 'SVHN_2_all', 'SVHN_3_all', 'SVHN_4_all', 'SVHN_5_all', 
                          'SVHN_6_all', 'SVHN_7_all', 'SVHN_8_all', 'SVHN_9_all', 
                          '20news_0_all', '20news_1_all', '20news_2_all', '20news_3_all', '20news_4_all', '20news_5_all', 
                          'agnews_0_all', 'agnews_1_all', 'agnews_2_all', 'agnews_3_all', 'amazon_all', 'imdb_all', 'yelp_all',
                         
                         '12_fault_all','15_Hepatitis_all', '16_http_all', '17_InternetAds_all', '21_Lymphography_all', '34_smtp_all', '37_Stamps_all', '43_WDBC_all', '44_Wilt_all', '45_wine_all', '46_WPBC_all', '47_yeast_all',
                            
                            '2_annthyroid_all', '4_breastw_all', '6_cardio_all', '10_cover_all', '14_glass_all', '18_Ionosphere_all', '20_letter_all', '23_mammography_all', '25_musk_all', '26_optdigits_all', '28_pendigits_all', '29_Pima_all', '30_satellite_all', '31_satimage-2_all', '32_shuttle_all', '36_speech_all', '38_thyroid_all', '39_vertebral_all', '40_vowels_all', '42_WBC_all',
                            
                            'CIFAR10_0', 'CIFAR10_1', 'CIFAR10_2', 'CIFAR10_3', 'CIFAR10_4', 'CIFAR10_5', 
                            'CIFAR10_6', 'CIFAR10_7', 'CIFAR10_8', 'CIFAR10_9', 
                            'MNIST-C_brightness', 'MNIST-C_canny_edges', 'MNIST-C_dotted_line', 'MNIST-C_fog', 'MNIST-C_glass_blur', 
                            'MNIST-C_identity', 'MNIST-C_impulse_noise', 'MNIST-C_motion_blur', 'MNIST-C_rotate', 'MNIST-C_scale',
                            'MNIST-C_shear', 'MNIST-C_shot_noise', 'MNIST-C_spatter', 'MNIST-C_stripe', 'MNIST-C_translate', 
                            'MNIST-C_zigzag', 'MVTec-AD_bottle', 'MVTec-AD_cable', 'MVTec-AD_capsule', 'MVTec-AD_carpet', 
                            'MVTec-AD_grid', 'MVTec-AD_hazelnut', 'MVTec-AD_leather', 'MVTec-AD_metal_nut', 'MVTec-AD_pill',
                            'MVTec-AD_screw', 'MVTec-AD_tile', 'MVTec-AD_toothbrush', 'MVTec-AD_transistor', 'MVTec-AD_wood', 'MVTec-AD_zipper',
                            'SVHN_0', 'SVHN_1', 'SVHN_2', 'SVHN_3', 'SVHN_4', 'SVHN_5', 'SVHN_6', 'SVHN_7', 'SVHN_8', 'SVHN_9',
                            '20news_0', '20news_1', '20news_2', '20news_3', '20news_4', '20news_5', 
                            'agnews_0', 'agnews_1', 'agnews_2', 'agnews_3', 'amazon', 'imdb', 'yelp',
                            
                            'arrhythmia_std', 'cardio_std', 'satellite_std', 'satimage-2_std', 'shuttle_std', 'thyroid_std',
                           'annthyroid_std', 'breastw_std', 'cover_std', 'glass_std', 'ionosphere_std', 'letter_std', 
                            'lympho_std', 'mammography_std', 'musk_std', 'optdigits_std', 'pendigits_std', 'pima_std', 
                            'speech_std', 'vertebral_std', 'vowels_std', 'wbc_std',
                            '1_ALOI_std', '3_backdoor_std', '5_campaign_std', '7_Cardiotocography_std', '8_celeba_std',
                            '9_census_std', '11_donors_std', '13_fraud_std', '19_landsat_std', '22_magic.gamma_std', 
                            '27_PageBlocks_std', '33_skin_std', '35_SpamBase_std', '41_Waveform_std'
                            
                            )
    assert dataset_name in implemented_datasets

    dataset = None
    
    
    if dataset_name in ('1_ALOI_random', '3_backdoor_random', '5_campaign_random', '7_Cardiotocography_random',
                        '8_celeba_random', '9_census_random', '11_donors_random', '13_fraud_random', '19_landsat_random', '22_magic.gamma_random', 
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
        
    elif dataset_name in ('1_ALOI_std', '3_backdoor_std', '5_campaign_std', '7_Cardiotocography_std',
                          '8_celeba_std', '9_census_std', '11_donors_std', '13_fraud_std', '19_landsat_std', '22_magic.gamma_std', 
                        '27_PageBlocks_std', '33_skin_std', '35_SpamBase_std', '41_Waveform_std'):
        dataset_name = dataset_name.replace('_std','')
        dataset = AdBenchStdDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
                                random_state=random_state)
        
    elif dataset_name in ('arrhythmia_std', 'cardio_std', 'satellite_std', 'satimage-2_std', 'shuttle_std', 'thyroid_std',
                           'annthyroid_std', 'breastw_std', 'cover_std', 'glass_std', 'ionosphere_std', 'letter_std', 
                          'lympho_std', 'mammography_std', 'musk_std', 'optdigits_std', 'pendigits_std', 'pima_std',
                          'speech_std', 'vertebral_std', 'vowels_std', 'wbc_std'):
        dataset_name = dataset_name.replace('_std','')
        dataset = ODDSADStdDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
                                random_state=random_state)

    
    
    elif dataset_name in ('arrhythmia_random', 'cardio_random', 'satellite_random', 'satimage-2_random', 'shuttle_random', 'thyroid_random',
                           'annthyroid_random', 'breastw_random', 'cover_random', 'glass_random', 'ionosphere_random',
                          'letter_random', 'lympho_random', 'mammography_random', 'musk_random', 'optdigits_random', 
                          'pendigits_random', 'pima_random', 'speech_random', 'vertebral_random', 'vowels_random', 'wbc_random'):
        dataset_name = dataset_name.replace('_random','')
        dataset = ODDSADRandomDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
                                random_state=random_state)
        
    
    
    elif (dataset_name in ('1_ALOI_all', '3_backdoor_all', '5_campaign_all', '7_Cardiotocography_all', '8_celeba_all', '9_census_all', '11_donors_all', '13_fraud_all', '19_landsat_all', 
                          '22_magic.gamma_all', '27_PageBlocks_all', '33_skin_all', '35_SpamBase_all', '41_Waveform_all',
                         'CIFAR10_0_all', 'CIFAR10_1_all', 'CIFAR10_2_all', 'CIFAR10_3_all', 'CIFAR10_4_all', 'CIFAR10_5_all', 'CIFAR10_6_all', 'CIFAR10_7_all', 
                          'CIFAR10_8_all', 'CIFAR10_9_all', 'MNIST-C_brightness_all', 'MNIST-C_canny_edges_all', 'MNIST-C_dotted_line_all', 'MNIST-C_fog_all', 
                          'MNIST-C_glass_blur_all', 'MNIST-C_identity_all', 'MNIST-C_impulse_noise_all', 'MNIST-C_motion_blur_all', 'MNIST-C_rotate_all', 
                          'MNIST-C_scale_all', 'MNIST-C_shear_all', 'MNIST-C_shot_noise_all', 'MNIST-C_spatter_all', 'MNIST-C_stripe_all', 
                          'MNIST-C_translate_all', 'MNIST-C_zigzag_all', 'MVTec-AD_bottle_all', 'MVTec-AD_cable_all', 'MVTec-AD_capsule_all', 
                          'MVTec-AD_carpet_all', 'MVTec-AD_grid_all', 'MVTec-AD_hazelnut_all', 'MVTec-AD_leather_all', 'MVTec-AD_metal_nut_all',
                          'MVTec-AD_pill_all', 'MVTec-AD_screw_all', 'MVTec-AD_tile_all', 'MVTec-AD_toothbrush_all', 'MVTec-AD_transistor_all', 
                          'MVTec-AD_wood_all', 'MVTec-AD_zipper_all', 'SVHN_0_all', 'SVHN_1_all', 'SVHN_2_all', 'SVHN_3_all', 'SVHN_4_all', 'SVHN_5_all', 
                          'SVHN_6_all', 'SVHN_7_all', 'SVHN_8_all', 'SVHN_9_all', 
                          '20news_0_all', '20news_1_all', '20news_2_all', '20news_3_all', '20news_4_all', '20news_5_all', 
                          'agnews_0_all', 'agnews_1_all', 'agnews_2_all', 'agnews_3_all', 'amazon_all', 'imdb_all', 'yelp_all',
                         
                         '12_fault_all','15_Hepatitis_all', '16_http_all', '17_InternetAds_all', '21_Lymphography_all', '34_smtp_all', '37_Stamps_all', '43_WDBC_all', '44_Wilt_all', '45_wine_all', '46_WPBC_all', '47_yeast_all',
                           '2_annthyroid_all', '4_breastw_all', '6_cardio_all', '10_cover_all', '14_glass_all', '18_Ionosphere_all', '20_letter_all', '23_mammography_all', '25_musk_all', '26_optdigits_all', '28_pendigits_all', '29_Pima_all', '30_satellite_all', '31_satimage-2_all', '32_shuttle_all', '36_speech_all', '38_thyroid_all', '39_vertebral_all', '40_vowels_all', '42_WBC_all'
)):
        dataset_name = dataset_name.replace('_all','')
        dataset = AdBench_All_Dataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
                                random_state=random_state)
    
    elif dataset_name in ('1_ALOI', '3_backdoor', '5_campaign', '7_Cardiotocography', '8_celeba', '9_census', '11_donors', '13_fraud', '19_landsat', 
                          '22_magic.gamma', '27_PageBlocks', '33_skin', '35_SpamBase', '41_Waveform',
                         'CIFAR10_0', 'CIFAR10_1', 'CIFAR10_2', 'CIFAR10_3', 'CIFAR10_4', 'CIFAR10_5', 'CIFAR10_6', 'CIFAR10_7', 
                          'CIFAR10_8', 'CIFAR10_9', 'MNIST-C_brightness', 'MNIST-C_canny_edges', 'MNIST-C_dotted_line', 'MNIST-C_fog', 
                          'MNIST-C_glass_blur', 'MNIST-C_identity', 'MNIST-C_impulse_noise', 'MNIST-C_motion_blur', 'MNIST-C_rotate', 
                          'MNIST-C_scale', 'MNIST-C_shear', 'MNIST-C_shot_noise', 'MNIST-C_spatter', 'MNIST-C_stripe', 
                          'MNIST-C_translate', 'MNIST-C_zigzag', 'MVTec-AD_bottle', 'MVTec-AD_cable', 'MVTec-AD_capsule', 
                          'MVTec-AD_carpet', 'MVTec-AD_grid', 'MVTec-AD_hazelnut', 'MVTec-AD_leather', 'MVTec-AD_metal_nut',
                          'MVTec-AD_pill', 'MVTec-AD_screw', 'MVTec-AD_tile', 'MVTec-AD_toothbrush', 'MVTec-AD_transistor', 
                          'MVTec-AD_wood', 'MVTec-AD_zipper', 'SVHN_0', 'SVHN_1', 'SVHN_2', 'SVHN_3', 'SVHN_4', 'SVHN_5', 
                          'SVHN_6', 'SVHN_7', 'SVHN_8', 'SVHN_9', 
                          '20news_0', '20news_1', '20news_2', '20news_3', '20news_4', '20news_5', 
                          'agnews_0', 'agnews_1', 'agnews_2', 'agnews_3', 'amazon', 'imdb', 'yelp',
                         
                         '12_fault','15_Hepatitis', '16_http', '17_InternetAds', '21_Lymphography', '34_smtp', '37_Stamps', '43_WDBC', '44_Wilt', '45_wine', '46_WPBC', '47_yeast'
                         
                         ):
        dataset = AdBenchDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
                                random_state=random_state)

    
    elif dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path,
                                normal_class=normal_class,
                                known_outlier_class=known_outlier_class,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution)
    elif dataset_name == 'mnist_delpix':
        dataset = MNIST_delPixel_Dataset(root=data_path,
                                normal_class=normal_class,
                                known_outlier_class=known_outlier_class,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution)
        
    elif dataset_name == 'reuters':
        dataset = ReutersADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state,
                                  normal_class=normal_class)

    elif dataset_name == 'fmnist':
        dataset = FashionMNIST_Dataset(root=data_path,
                                       normal_class=normal_class,
                                       known_outlier_class=known_outlier_class,
                                       n_known_outlier_classes=n_known_outlier_classes,
                                       ratio_known_normal=ratio_known_normal,
                                       ratio_known_outlier=ratio_known_outlier,
                                       ratio_pollution=ratio_pollution)
    elif dataset_name == 'fmnist_delpix':
        dataset = FashionMNIST_delPixel_Dataset(root=data_path,
                                       normal_class=normal_class,
                                       known_outlier_class=known_outlier_class,
                                       n_known_outlier_classes=n_known_outlier_classes,
                                       ratio_known_normal=ratio_known_normal,
                                       ratio_known_outlier=ratio_known_outlier,
                                       ratio_pollution=ratio_pollution)
    elif dataset_name == 'svhn':
        dataset = SVHN_Dataset(root=os.path.join(data_path,'SVHN'),
                                       normal_class=normal_class,
                                       known_outlier_class=known_outlier_class,
                                       n_known_outlier_classes=n_known_outlier_classes,
                                       ratio_known_normal=ratio_known_normal,
                                       ratio_known_outlier=ratio_known_outlier,
                                       ratio_pollution=ratio_pollution)


    elif dataset_name == 'wafer':
        dataset =  WaferADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)
        
        
    elif dataset_name == 'wafer_scale':
        dataset =  WaferScaleADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)
    elif dataset_name == 'wafer_scale_deepsvdd':
        dataset =  WaferScaleADDataset_deepsvdd(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)


    elif dataset_name in ('arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid',
                           'annthyroid', 'breastw', 'cover', 'ecoli', 'glass', 'ionosphere', 
                          'letter', 'lympho', 'mammography', 'musk', 'optdigits', 'pendigits',
                          'pima', 'speech', 'vertebral', 'vowels', 'wbc', 'wine'):
        dataset = ODDSADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                feature_range = feature_range,
                                random_state=random_state)

    return dataset
