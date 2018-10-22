#! -*- coding: utf-8 -*-
from flexible_clustering_tree.models import ClusteringOperator, MultiClusteringOperator, MultiFeatureMatrixObject, \
    FeatureMatrixObject, ClusterTreeObject
import unittest
# clustering algorithm
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
# else
from typing import List
import numpy


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def generate_random_matrix(self, n_data: int, n_feature: int):
        return numpy.random.rand(n_data, n_feature)

    def test_exception_MultiFeatureMatrixObject_un_match(self):
        """raise exception if index size of the matrix is not same as index:label dict."""
        feature_matrix_obj = [FeatureMatrixObject(0, self.generate_random_matrix(100, 200))]

        with self.assertRaises(Exception):
            MultiFeatureMatrixObject(feature_matrix_obj,
                                     {i: "test-{}".format(i) for i in range(10)},
                                     {i: {} for i in range(10)})

    def test_non_zero(self):
        feature_matrix_obj = [FeatureMatrixObject(1, self.generate_random_matrix(100, 200))]
        with self.assertRaises(Exception):
            MultiFeatureMatrixObject(feature_matrix_obj,
                                     {i: "test-{}".format(i) for i in range(100)},
                                     {i: {} for i in range(10)})

        clustering_algorithm = [ClusteringOperator(1, n_cluster=-1, instance_clustering=HDBSCAN())]
        with self.assertRaises(Exception):
            MultiClusteringOperator(clustering_algorithm)

if __name__ == '__main__':
    unittest.main()
