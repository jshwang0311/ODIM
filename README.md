# ODIM
## Abstract
This study aims to solve the unsupervised outlier detection problem where training data contains some outliers, and any label information about inliers and outliers is not given. 
We propose a powerful and efficient learning framework to identify inliers in a training data set using deep neural networks. 
We start with a new observation called the inlier-memorization (IM) effect. 
When we train a deep generative model with data contaminated with outliers, the model first memorizes inliers before outliers. 
Exploiting this finding, we develop a new method called the outlier detection via the IM effect (ODIM). 
The ODIM only requires a few updates; thus, it is time-efficient, tens of times faster than other deep-learning-based algorithms. 
Also, the ODIM filters out inliers successfully, regardless of the types of data such as tabular and image. 
For detail, the following paper is described:
* Dongha Kim, Jaesung Hwang, Jongjin Lee, Kungwoong Kim and Yongdai Kim, ODIM: a fast method to identify inliers via inlier-memorization effect of deep generative models.

## Run the Experiments
In this experiments, you can calculate TrainAUC, TrainAP(AveragePrecision), TestAUC and TestAP of datasets using ODIM.
```bash
python calculate_AUC_light_all.py --dataset_name_option "adbench_all"  --gpu_num 0 --batch_size 64
```
And you can change `dataset_name_option` to one of None, "adbench", "adbench_all", "all".
If you set `dataset_name_option` to None, you must specify one of the dataset names "mnist", "fmnist", "wafer_scale", or adbench dataset names (e.g., "1_ALOI") in `dataset_name`.
If you set `dataset_name_option` to "adench_all", you can get the trainAUC and trainAP for the entire AdBench dataset.
If you set `dataset_name_option` to "adench", you can get the trainAUC, trainAP, testAUC, and testAP after randomly splitting the AdBench dataset into train and test data.
