# AffectNet_cluster
- Cluster the images in each category of expressions from AffectNet dataset

# Getting Started

## Preprocessing
- The tool provides two clustering methods
  - Based on AU detection result: Use the JAANet_AU_detection to perform AU detection on the images in each expression category and save the results to a csv file
  - Based on the features extracted from SIFT

## Clustering
- AU_cluster
```
python AU_cluster.py --cluster_num 8 --exp_label 1 --AU_root /mnt/data-1/zhiwei.huang/AffectNet/au_detection_info
```
- SIFT_cluster
```
python SIFT_cluster.py --cluster_num 8 --exp_label 1 --dataset_root /mnt/data-1/zhiwei.huang/AffectNet --imgs_folder Manually_Annotated/Manually_Annotated_Images/ --csv_name affectNet_formal_training.csv
