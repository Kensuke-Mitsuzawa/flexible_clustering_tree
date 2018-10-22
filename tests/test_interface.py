#! -*- coding: utf-8 -*-
import unittest
from flexible_clustering_tree.interface import FlexibleClustering


class TestInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_fit_transform(self):
        import numpy
        input_feature_matrix_1st = numpy.random.rand(100, 200)
        input_feature_matrix_2nd = numpy.random.rand(100, 200)
        from flexible_clustering_tree.models import FeatureMatrixObject, MultiFeatureMatrixObject, ClusteringOperator, MultiClusteringOperator
        feature_matrix_objects = [FeatureMatrixObject(level=0, matrix_object=input_feature_matrix_1st), FeatureMatrixObject(level=1, matrix_object=input_feature_matrix_2nd)]
        index2label = {i: "test-{}".format(i) for i in range(0, 100)}
        multi_feature_input = MultiFeatureMatrixObject(feature_matrix_objects, index2label)
        from sklearn.cluster import KMeans
        multi_clustering_operator = MultiClusteringOperator([ClusteringOperator(level=0, n_cluster=3, instance_clustering=KMeans(n_clusters=3))])
        f_clustering = FlexibleClustering(max_depth=10)
        data2cluster = f_clustering.fit_transform(multi_feature_input, multi_clustering_operator)
        self.assertTrue(data2cluster, dict)


if __name__ == '__main__':
    unittest.main()
