#! -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, ClusterMixin
from flexible_clustering_tree.models import FeatureMatrixObject
from itertools import groupby
import numpy


class StringAggregation(BaseEstimator, ClusterMixin):
    """Class to aggregate information by string"""
    def __init__(self):
        pass

    def fit(self, X: FeatureMatrixObject):
        """run aggregation by string"""
        __t_d_id2str = [(d_id, __str) for d_id, __str in enumerate(X.feature_strings)]
        labels_ = []
        c_id = 0
        for __str, g_obj in groupby(sorted(__t_d_id2str, key=lambda t: t[1]), key=lambda t: t[1]):
            labels_ += [c_id for t in list(g_obj)]
            c_id += 1
        else:
            self.labels_ = labels_

        return self

    def fit_predict(self, X: FeatureMatrixObject, y=None):
        self.fit(X)
        return self.labels_

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass


def test():
    matrix_obj_input = numpy.random.rand(100, 128)
    string_inputs = ['d'] * 10 + ['e'] * 10 + ['c'] * 10 + ['a'] * 10 + ['b'] * 10 + ['f'] * 50
    matrix_obj_1st = FeatureMatrixObject(level=0, matrix_object=matrix_obj_input, feature_strings=string_inputs)
    aggregation_obj = StringAggregation()
    aggregation_obj.fit(matrix_obj_1st)
    labels = aggregation_obj.fit_predict(matrix_obj_1st)
    assert len(labels) == len(string_inputs)


if __name__ == '__main__':
    test()
