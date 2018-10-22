#! -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import os
import sys
import codecs

"""This example shows you an example case of flexible-clustering on image data.
In this example, it uses sub data from cifar-10 image collection.
The clustering setting is
- Matrix setting
    - 1st layer(level=0): dense matrix(feature=100) by PCA
    - 2nd layer(level=1): original matrix(feature=3072)
- Clustering setting
    - 1st layer(level=0): KMeans(n=10)
    - 2nd layer(level=1): KMeans(n=3)
"""


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


ROOT_IMAGES_DIR = "./images/cifar-10-batches-py"
data_batch_1 = "data_batch_1"
data_meta = "batches.meta"

image_file = unpickle(os.path.join(ROOT_IMAGES_DIR, data_batch_1))
meta_file = unpickle(os.path.join(ROOT_IMAGES_DIR, data_meta))

import sys
sys.path.append("..")
from flexible_clustering_tree.interface import FlexibleClustering
from flexible_clustering_tree.models import FeatureMatrixObject, MultiFeatureMatrixObject, ClusteringOperator, MultiClusteringOperator

label_index2label = {i: label for i, label in enumerate(meta_file[b'label_names'])}
matrix_index2label = {i: str(label_index2label[label_index]) for i, label_index in enumerate(image_file[b'labels'])}
original_feature_matrix = image_file[b'data']

limit_of_sample = 1000
sampled_original_feature_matrix = original_feature_matrix[:limit_of_sample]
sampled_matrix_index2label = {i: str(label_index2label[label_index])
                              for i, label_index in enumerate(image_file[b'labels']) if i < limit_of_sample}
# feature decomposition with PCA. We set this matrix as 1st layer(level=0)
from sklearn.decomposition.pca import PCA
dense_sampled_original_feature_matrix = PCA(n_components=100).fit_transform(sampled_original_feature_matrix)
f_obj_1st = FeatureMatrixObject(0,  dense_sampled_original_feature_matrix)

# set matrix object
f_obj_2nd = FeatureMatrixObject(1,  sampled_original_feature_matrix)
multi_f_obj = MultiFeatureMatrixObject([f_obj_1st, f_obj_2nd], sampled_matrix_index2label)

# set clustering algorithm
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
c_obj_1st = ClusteringOperator(level=0, n_cluster=10, instance_clustering=KMeans(n_clusters=10))
c_obj_2nd = ClusteringOperator(level=1, n_cluster=3, instance_clustering=KMeans(n_clusters=3))
multi_c_obj = MultiClusteringOperator([c_obj_1st, c_obj_2nd])


# run flexible clustering with max depth = 5
flexible_clustering_runner = FlexibleClustering(max_depth=3)
index2cluster_id = flexible_clustering_runner.fit_transform(x=multi_f_obj, multi_clustering_operator=multi_c_obj)

# generate html page with collapsible tree
with codecs.open("animal_example.html", "w") as f:
    f.write(flexible_clustering_runner.clustering_tree.to_html())
