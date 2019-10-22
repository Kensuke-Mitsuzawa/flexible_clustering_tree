#! -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from flexible_clustering_tree.models import MultiFeatureMatrixObject, MultiClusteringOperator, ClusterTreeObject
from flexible_clustering_tree.base_clustering import RecursiveClustering
from typing import Dict


class FlexibleClustering(BaseEstimator):
    def __init__(self,
                 max_depth: int,
                 is_auto_switch: bool=True,
                 threshold_ratio_auto_switch: float=10.0,
                 threshold_minimum_unique_vector: int=10,
                 multi_clustering_operator: MultiClusteringOperator=None):
        """Flexible clustering

        :param max_depth: max depth of a tree. It keeps clustering until this parameter as much as possible.
        :param is_auto_switch:
        :param threshold_ratio_auto_switch:
        :param threshold_minimum_unique_vector:
        :param multi_clustering_operator:

        Attributes
        ----------
        clustering_tree: `ClusterTreeObject`
            An object of clustering result. It keeps all information of a result.
            You could get a nested tree object with `to_dict()`

        Examples
        ----------
        >>> import numpy
        >>> input_feature_matrix_1st = numpy.random.rand(100, 200)
        >>> input_feature_matrix_2nd = numpy.random.rand(100, 200)
        >>> from flexible_clustering_tree.models import FeatureMatrixObject, MultiFeatureMatrixObject, ClusteringOperator, MultiClusteringOperator
        >>> feature_matrix_objects = [FeatureMatrixObject(level=0, matrix_object=input_feature_matrix_1st), FeatureMatrixObject(level=1, matrix_object=input_feature_matrix_2nd)]
        >>> index2label = {i: "test-{}".format(i) for i in range(0, 100)}
        >>> multi_feature_input = MultiFeatureMatrixObject(feature_matrix_objects, index2label)
        >>> from sklearn.cluster import KMeans
        >>> multi_clustering_operator = MultiClusteringOperator([ClusteringOperator(level=0, n_cluster=3, instance_clustering=KMeans(n_clusters=3))])
        >>> f_clustering = FlexibleClustering(max_depth=10)
        >>> f_clustering.fit_transform(multi_feature_input, multi_clustering_operator)

        Notes
        ------
        """
        self.multi_clustering_operator = multi_clustering_operator
        self.max_depth = max_depth
        self.is_auto_switch = is_auto_switch
        self.threshold_ratio_auto_switch = threshold_ratio_auto_switch
        self.threshold_minimum_unique_vector = threshold_minimum_unique_vector
        self.clustering_tree = None  # type: ClusterTreeObject

    def fit(self,
            x: MultiFeatureMatrixObject,
            y=None,
            multi_clustering_operator: MultiClusteringOperator=None):
        """
        :param x: `MultiFeatureMatrixObject` which has feature matrix for each level
        :param y: not in use
        :param multi_clustering_operator:
        :return: itself
        """
        if self.multi_clustering_operator is None and multi_clustering_operator is None:
            raise Exception("multi_clustering_operator must be defined.")
        elif multi_clustering_operator is None:
            multi_clustering_operator = self.multi_clustering_operator

        clustering_runner = RecursiveClustering()
        clustering_tree = clustering_runner.run_recursive_clustering(
            multi_clustering_operator=multi_clustering_operator,
            multi_feature_matrix_object=x,
            max_depth=self.max_depth,
            is_auto_switch=self.is_auto_switch,
            threshold_ratio_auto_switch=self.threshold_ratio_auto_switch,
            threshold_minimum_unique_vector=self.threshold_minimum_unique_vector)
        self.clustering_tree = clustering_tree

        return self

    def transform(self, x)->Dict[int, int]:
        """Get clustering result. The result is shown with a dict object.
         You're supposed to run method 'fit' first.

        :param x: not in use.
        :return: dict object of {data-id: cluster-id}
        """
        if self.clustering_tree is None:
            raise Exception("run method 'fit' first.")
        return self.clustering_tree.get_labels()

    def fit_transform(self,
                      x: MultiFeatureMatrixObject,
                      multi_clustering_operator: MultiClusteringOperator=None)->Dict[int, int]:
        """This method directly get result.

        :param x:
        :param multi_clustering_operator:
        :return:
        """
        self.fit(x, multi_clustering_operator=multi_clustering_operator)
        return self.transform(x)

    def predict(self, x):
        raise NotImplementedError("method predict is not defined.")

    def score(self, x, y):
        raise NotImplementedError("method score is not defined.")

    def get_params(self, deep=True):
        return {
            "multi_clustering_operator": self.multi_clustering_operator.to_dict(),
            "max_depth": self.max_depth,
            "is_auto_switch": self.is_auto_switch,
            "threshold_ratio_auto_switch": self.threshold_ratio_auto_switch,
            "threshold_minimum_unique_vector": self.threshold_minimum_unique_vector
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
