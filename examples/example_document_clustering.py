#! -*- coding: utf-8 -*-
import logging
import re
import numpy
import gensim
import codecs
from collections import Counter
from nltk import stem
from typing import List
from flexible_clustering_tree.logger import logger
# make download 20news group file
from sklearn.datasets import fetch_20newsgroups
# load module
import sys
sys.path.append("..")
from flexible_clustering_tree.interface import FlexibleClustering
from flexible_clustering_tree.models import FeatureMatrixObject, MultiFeatureMatrixObject, ClusteringOperator, MultiClusteringOperator

"""This example shows flexible-clustering on text data(20-news).
Sometimes, news subject explains enough information to understand. 
Therefore, you don't need to all text to run clustering roughly.
However, you might wanna use text to run clustering after rough-clustering.
Here, the idea is implemented with following setting,
Feature Matrix setting
- 1st layer(level=0): dense vector of subject text(used word2vec mean)
- 2nd layer(level=1): dense vector by feature-decomposition SVD
- 2nd layer(level=1): sparse vector which is weighted with TF-IDF

Clustering setting
- 1st layer(level=0): HDBSCAN
- 2nd layer(level=1): Kmeans

Note: It code needs word2vec model. Please download some models beforehand.
If you don't have any models, you could use google pre-trained model. You could download it from

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
"""
DATA_LIMIT = 1000

newsgroups_train = fetch_20newsgroups(subset='train')
logger.setLevel(logging.DEBUG)
lemmatizer = stem.WordNetLemmatizer()
# load word2vec model with gensim
logger.debug("Loading word2vec model...")
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

category_names = newsgroups_train.target_names
logger.debug("20-news has {} categories".format(len(category_names)))


def extract_subject_name(document: str)->str:
    """It gets only Subject name from news document
    """
    return re.search(r'Subject:\s.+?\n', document)\
        .group(0)\
        .replace("\nSubject: ", "").strip()


def run_nltk_lemma(subject_name: str)->List[str]:
    return [lemmatizer.lemmatize(t).strip(':?!><') for t in subject_name.lower().split()]

# takes only subject name of news text
logger.debug("Now pre-processing on subject text...")
news_subjects = [extract_subject_name(d) for d in newsgroups_train.data[:DATA_LIMIT]]
news_subjects_lemma = [run_nltk_lemma(d) for d in news_subjects]
logger.debug("Now making 1st layer matrix...")
feature_matrix_1st = numpy.array(
    [numpy.mean(numpy.array([w2v_model[t] for t in d if t in w2v_model]), axis=0)
     for d in news_subjects_lemma])

# use dense vector as 2nd layer feature-matrix
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
logger.debug("Now making 2nd layer matrix...")
svd_transformer = TruncatedSVD(n_components=100)
news_text = [dict(Counter(run_nltk_lemma(d))) for d in newsgroups_train.data[:DATA_LIMIT]]
feature_matrix_2nd = svd_transformer.fit_transform(v.fit_transform(news_text))

# use TF-IDF vector as 3nd layer feature-matrix
logger.debug("Now making 3rd layer matrix...")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tf_idf_vector = vectorizer.fit_transform(newsgroups_train.data[:DATA_LIMIT])
feature_matrix_3rd = tf_idf_vector

# set feature matrix
f_obj_1st = FeatureMatrixObject(0, feature_matrix_1st)
f_obj_2nd = FeatureMatrixObject(1, feature_matrix_2nd)
f_obj_3rd = FeatureMatrixObject(2, feature_matrix_3rd)
index2category = {i: t for i, t in enumerate(newsgroups_train.target_names[:DATA_LIMIT])}
dict_index2label = {i: index2category[t_no] for i, t_no in enumerate(newsgroups_train.target[:DATA_LIMIT])}
multi_feautre_matrix = MultiFeatureMatrixObject(
    [f_obj_1st, f_obj_2nd, f_obj_3rd],
    dict_index2label=dict_index2label
)

# set clustering operation
from sklearn.cluster.k_means_ import KMeans
from hdbscan.hdbscan_ import HDBSCAN
c_operation_1st = ClusteringOperator(0, -1, HDBSCAN())
c_operation_2nd = ClusteringOperator(1, 5, KMeans(5))
multi_clustering = MultiClusteringOperator([c_operation_1st, c_operation_2nd])

# run flexible clustering
clustering_runner = FlexibleClustering(max_depth=3)
index2cluster_no = clustering_runner.fit_transform(multi_feautre_matrix, multi_clustering)
html = clustering_runner.clustering_tree.to_html()

with codecs.open("20news_example.html", "w", "utf-8") as f:
    f.write(html)

# generate objects for table
table_objects = clustering_runner.clustering_tree.to_objects()
import pandas
print(pandas.DataFrame(table_objects['cluster_information']))
print(pandas.DataFrame(table_objects['leaf_information']))
