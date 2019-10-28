#! -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, ClusterMixin
from itertools import groupby
import numpy
from typing import List


class StringAggregation(BaseEstimator, ClusterMixin):
    """Class to aggregate information by string"""
    def __init__(self):
        pass

    def fit(self, X: List[str]):
        """run aggregation by string"""
        if not isinstance(X, list):
            raise Exception(
                'StringAggregation has not list object. This class expects str of list object, but {} is given.'
                    .format(type(X)))

        __t_d_id2str = [(d_id, __str) for d_id, __str in enumerate(X)]
        labels_ = []
        c_id = 0
        for __str, g_obj in groupby(sorted(__t_d_id2str, key=lambda t: t[1]), key=lambda t: t[1]):
            labels_ += [c_id for t in list(g_obj)]
            c_id += 1
        else:
            self.labels_ = labels_

        return self

    def fit_predict(self, X: List[str], y=None):
        self.fit(X)
        return self.labels_

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass

    def __str__(self):
        return 'StringAggregation'


def test():
    matrix_obj_input = numpy.random.rand(100, 128)
    string_inputs = ['d'] * 10 + ['e'] * 10 + ['c'] * 10 + ['a'] * 10 + ['b'] * 10 + ['f'] * 50
    aggregation_obj = StringAggregation()
    aggregation_obj.fit(string_inputs)
    labels = aggregation_obj.fit_predict(string_inputs)
    assert len(labels) == len(string_inputs)


if __name__ == '__main__':
    test()
