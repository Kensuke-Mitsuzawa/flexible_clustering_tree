#! -*- coding: utf-8 -*-
from flexible_clustering_tree.base_clustering import RecursiveClustering
from flexible_clustering_tree.models import ClusteringOperator, MultiClusteringOperator, MultiFeatureMatrixObject, FeatureMatrixObject, ClusterTreeObject
import unittest
# clustering algorithm
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
# else
from typing import List
import numpy


class TestBaseClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def generate_random_matrix(self, n_data: int, n_feature: int):
        return numpy.random.rand(n_data, n_feature)

    def __extract_tag_id(self, children_nodes, seq_extracted_ids: List[str]):
        """method to extract all data-id in a tree"""
        for node_object in children_nodes:
            if "_children" in node_object:
                self.__extract_tag_id(node_object["_children"], seq_extracted_ids)
            elif "children" in node_object:
                self.__extract_tag_id(node_object["children"], seq_extracted_ids)
            elif "name" in node_object:
                seq_extracted_ids.append(int(node_object["name"]))
        else:
            return seq_extracted_ids

    def test_single_matrix_clustering(self):
        """a case to test run clustering on single matrix and single clustering-algorithm"""
        # set feature matrix input
        input_feature_matrix = self.generate_random_matrix(100, 200)
        multi_feature_input = MultiFeatureMatrixObject([FeatureMatrixObject(level=0, matrix_object=input_feature_matrix)],
                                                       dict_index2label={i: "test-{}".format(i) for i in range(0, 100)})
        # set clustering operator
        multi_clustering_operator = MultiClusteringOperator(
            [ClusteringOperator(level=0, n_cluster=3, instance_clustering=KMeans(n_clusters=3))]
        )

        clustering_runner = RecursiveClustering()
        tree_object = clustering_runner.run_recursive_clustering(multi_feature_matrix_object=multi_feature_input,
                                                                 multi_clustering_operator=multi_clustering_operator,
                                                                 max_depth=10,
                                                                 is_auto_switch=True)
        self.assertTrue(isinstance(tree_object, ClusterTreeObject))
        dict_collapsible_tree = tree_object.to_dict()
        self.assertTrue(dict_collapsible_tree, dict)
        data_ids = self.__extract_tag_id(dict_collapsible_tree["children"], [])
        for i in range(0, 100):
            self.assertTrue(i in data_ids)

    def test_multi_clustering(self):
        # set feature matrix input
        input_feature_matrix = self.generate_random_matrix(300, 200)
        multi_feature_input = MultiFeatureMatrixObject([FeatureMatrixObject(level=0, matrix_object=input_feature_matrix)],
                                                       dict_index2label={i: "test-{}".format(i) for i in range(0, 300)})
        # set clustering operator
        multi_clustering_operator = MultiClusteringOperator(
            [
                ClusteringOperator(level=0, n_cluster=-1, instance_clustering=HDBSCAN(min_cluster_size=2)),
                ClusteringOperator(level=1, n_cluster=3, instance_clustering=KMeans(n_clusters=3))
            ]
        )

        clustering_runner = RecursiveClustering()
        tree_object = clustering_runner.run_recursive_clustering(multi_feature_matrix_object=multi_feature_input,
                                                                 multi_clustering_operator=multi_clustering_operator,
                                                                 max_depth=10,
                                                                 is_auto_switch=True)
        self.assertTrue(isinstance(tree_object, ClusterTreeObject))
        dict_collapsible_tree = tree_object.to_dict()
        self.assertTrue(dict_collapsible_tree, dict)
        data_ids = self.__extract_tag_id(dict_collapsible_tree["children"], [])
        for i in range(0, 100):
            self.assertTrue(i in data_ids)
        # test generating html file
        html_file = tree_object.to_html()
        self.assertTrue(isinstance(html_file, str))
        converted_obj = tree_object.to_objects()
        self.assertTrue(isinstance(converted_obj, dict))
        self.assertTrue('cluster_information' in converted_obj)
        self.assertTrue('leaf_information' in converted_obj)


if __name__ == '__main__':
    unittest.main()
