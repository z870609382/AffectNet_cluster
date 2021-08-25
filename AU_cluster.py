import os
import cv2
import numpy as np
from sklearn.cluster import KMeans  # 在sklearn.cluster库中有KMeans类可以直接使用
from sklearn.metrics import silhouette_score
import os.path
from dataloader_cluster import DataLoader
import torch
import logging
import argparse
import pandas as pd
from ast import literal_eval
import math
import joblib
import matplotlib.pyplot as plt

def evaluation_function(au_array, max_cluster):
    assert max_cluster > 1, "max_cluster needs to be greater than 2"
    clusters = np.arange(2, max_cluster + 1)
    models = [KMeans(n_clusters=cluster, max_iter=1000).fit(au_array) for cluster in clusters]
    sc_score_list = [silhouette_score(au_array, model.labels_) for model in models]      #score∈[-1, 1],越大越好，且当值为负时，表明样本被分配到错误的簇中，聚类结果不可接受；对于接近0的结果，则表明聚类结果有重叠的情况。
    plt.figure()
    plt.plot(clusters, sc_score_list, '*-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient Score')
    plt.show()
    return clusters[np.argmax(sc_score_list)] #返回sc_score最大对应的聚类中心个数


def main(config):

    label_mapping = {'1': 'happy', '2': 'sad', '3': 'surprise', '4': 'fear', '5': 'disgust', '6': 'angry', '7': 'contempt'}

    AU_result = os.path.join(config.AU_root, 'DISFA_pred_affectNet_{}_AU.csv'.format(label_mapping[str(config.exp_label)]))
    loan_data = pd.read_csv(AU_result)     #读取记录AU检测结果的csv文件
    au = []
    for result in loan_data['AU_classify']:
        if type(result) != str:
            break
        result_trans = literal_eval(result)
        au.append(result_trans)
    au = np.array(au)
    # cluster_num = evaluation_function(au, 12)       #可以根据silhouette_score评估该选几个聚类中心
    # print(cluster_num)


    print('exp_label{} cluster begin'.format(config.exp_label))
    kmeans = KMeans(n_clusters=config.cluster_num, max_iter=1000).fit(au)  # 关键点聚类
    labels, cluster_centers = kmeans.labels_, kmeans.cluster_centers_



    ###########统计结果并把图片复制到其所属聚类中心文件夹下
    res = labels.tolist()
    result = [[] for i in range(config.cluster_num)]
    img_id = 0
    for i in range(len(res)):
        img_label = res[i]
        result[img_label].append(img_id)
        img_id += 1


    for i in range(config.cluster_num):
        print('第{}个聚类中心含样本{}个'.format(i + 1, res.count(i)))


    exp_type = label_mapping[str(config.exp_label)]
    output_dir = os.path.join(config.save_root, 'au_intensity_exp_cluster_{}'.format(config.cluster_num), exp_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joblib.dump(kmeans, os.path.join(output_dir, '{}_kmeans.model'.format(exp_type)))  #保存kmeans模型参数，之后可以用joblib.load加载并predict其他数据

    for i in range(config.cluster_num):

        save_dir = os.path.join(output_dir, 'centroid_{}'.format(i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        size = len(result[i])
        for j in range(size):
            file_path = loan_data['img_path'][result[i][j]]
            os.system('cp {} {}'.format(file_path, save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument(
        "--exp_label", type=int, default=1, help="the exp_label you want to cluster"
    )

    #Directories
    parser.add_argument(
        "--AU_root", type=str, default="/mnt/data-1/zhiwei.huang/AffectNet/au_detection_info"    #存放AU检测结果的csv文件所在文件夹
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/mnt/data-1/zhiwei.huang/AffectNet",
        help="model_save_root",
    )
    # Cluster configuration.
    parser.add_argument('--cluster_num', type=int, default=8)

    config = parser.parse_args()
    print(config)
    main(config)
    print('exp_label{} cluster finished'.format(config.exp_label))



