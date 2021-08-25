import os
import cv2
import numpy as np
from sklearn.cluster import KMeans  # 在sklearn.cluster库中有KMeans类可以直接使用。
import os.path
from dataloader_cluster import DataLoader
import torch
import logging
import argparse
import pandas as pd
from ast import literal_eval
import math
import joblib


# image_cluster 使用SIFT和KMEANS算法进行图像聚类，得到图像标签
def image_clusterByKMeans(cluster_num, path_filenames, img_size):
    features = []

    sift = cv2.xfeatures2d.SIFT_create()  # 使用SIFT算法提取图像特征
    # count  = 0
    files = path_filenames  # 特征检测
    bad_img = []
    for i, file in enumerate(files):
        img = cv2.imread(file)  # 读入文件
        img = cv2.resize(img, (img_size, img_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化处理，加快计算速度

        kp, des = sift.detectAndCompute(gray, None)  # 调用SIFT算法检测并计算描述符

        # print(des)
        # print(des.shape)
        # 有些图片对应的feature为None，选择跳过这些图片
        if des is None:
            # path_filenames.remove(file)
            print("the feature of img{} is nonetype".format(i))
            bad_img.append(i)
            continue
        reshape_feature = des.reshape(-1, 1)
        # print(reshape_feature.shape)

        # features.append(reshape_feature[0].tolist()) #第一种取第一个元素
        reshape_feature_Mean = np.array([np.mean(reshape_feature)])  # 第二种取平均值
        features.append(reshape_feature_Mean.tolist())

    input_x = np.array(features)  # 计算关键点 因为KMeans.fix(X[,y) X是需要2D 而不是1D
    kmeans = KMeans(n_clusters=cluster_num, max_iter=200).fit(input_x)  # 关键点聚类
    return kmeans.labels_, kmeans.cluster_centers_, bad_img  # 返回标签以及聚类中心


def main(config):

    label_mapping = {'1': 'happy', '2': 'sad', '3': 'surprise', '4': 'fear', '5': 'disgust', '6': 'angry', '7': 'contempt'}

    loader = None

    img_root = os.path.join(config.dataset_root, config.imgs_folder)

    train_exp_csv_file = os.path.join(config.dataset_root, config.csv_name)

    train_dataset = DataLoader(
        img_size=config.img_size, exp_class=config.exp_label, is_transform=False
    )
    train_dataset.load_data(train_exp_csv_file, img_root)
    loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
    )

    all_path = []
    for i, batch in enumerate(loader):
        if i % 5 == 0:
            print("batch {} / {}".format(i, len(loader)))

        imgs_path = batch[1]
        all_path = all_path + list(imgs_path)

    # with open("/home/users/zhiwei.huang/project_run/FER/JAAE_AU/data/list/AffectNet_{}_path.txt".format(label_mapping[str(config.exp_label)]), "w") as f:
    #     for path in all_path:
    #         f.write(path+'\n')             #记录affectNet训练集中该表情类别下所有图片的路径

    imgnum = len(all_path)  # affectNet训练集中该表情对应的图片数量
    print("img_num is", imgnum)
    # ②调用image_cluster函数实现SIFT提取特征并使用KNN聚类方法，得到图像标签
    labels, cluster_centers, bad_img = image_clusterByKMeans(config.cluster_num, all_path, config.img_size)
    print(labels.tolist())

    ###########统计结果并把图片复制到其所属聚类中心文件夹下
    res = labels.tolist()
    # print("bad_img_id is", bad_img)
    print("label_num is (filter bad_img)", len(res))


    result = [[] for i in range(config.cluster_num)]
    img_id = 0
    for i in range(len(res)):
        if img_id in bad_img:
            img_id += 1
            continue
        img_label = res[i]
        result[img_label].append(img_id)
        img_id += 1


    for i in range(config.cluster_num):
        print('第{}个聚类中心含样本{}个'.format(i + 1, res.count(i)))
        # print(len(result[i]))


    exp_type = label_mapping[str(config.exp_label)]
    output_dir = os.path.join(config.dataset_root, 'sift_exp_cluster_{}'.format(config.cluster_num), exp_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(config.cluster_num):
        save_dir = os.path.join(output_dir, 'centroid_{}'.format(i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        size = len(result[i])
        for j in range(size):
            # plt.subplot(1, size, j + 1)
            file_path = all_path[result[i][j]]
            os.system('cp {} {}'.format(file_path, save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument(
        "--exp_label", type=int, default=1, help="the exp_label you want to cluster"
    )
    parser.add_argument("--img_size", type=int, default=128, help="image resolution")

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/data-1/zhiwei.huang/AffectNet",
        help="dataset_root",
    )
    parser.add_argument(
        "--imgs_folder", type=str, default="Manually_Annotated/Manually_Annotated_Images/"
    )
    parser.add_argument(
        "--csv_name", type=str, default="affectNet_formal_training.csv"
    )

    # Cluster configuration.
    parser.add_argument("--batch_size", type=int, default=64, help="mini-batch size")
    parser.add_argument('--cluster_num', type=int, default=12)


    config = parser.parse_args()
    print(config)
    main(config)



