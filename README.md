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
* Dongha Kim, Jaesung Hwang, Kungwoong Kim and Yongdai Kim, ODIM: a fast method to identify inliers via inlier-memorization effect of deep generative models.

## Run the Experiments
In this experiments, you can calculate TrainAUC, TrainAP(AveragePrecision), TestAUC and TestAP of datasets using ODIM.
```bash
python calculate_AUC.py --dataset_name 'mnist' --gpu_num 0
```
You can change `dataset_name` to one of mnist, fmnist, wafer_scale, reuters and tables (annthyroid, brestw, cover, ...).
