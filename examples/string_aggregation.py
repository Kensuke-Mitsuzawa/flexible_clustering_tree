import numpy
from flexible_clustering_tree import FeatureMatrixObject, MultiFeatureMatrixObject, ClusteringOperator, \
    MultiClusteringOperator, FlexibleClustering

"""This script shows how you run the combination of string-aggregation and clustering.
At the first layer, it runs string-aggregation,
however, in the second layer, Kmeans runs.
"""

input_feature_matrix_1st = ['d'] * 10 + ['e'] * 10 + ['c'] * 10 + ['a'] * 10 + ['b'] * 10 + ['f'] * 50
input_feature_matrix_2nd = numpy.random.rand(100, 200)


feature_matrix_objects = [FeatureMatrixObject(level=0, feature_strings=input_feature_matrix_1st),
                          FeatureMatrixObject(level=1, matrix_object=input_feature_matrix_2nd)]
index2label = {i: "test-{}".format(i) for i in range(0, 100)}
multi_feature_input = MultiFeatureMatrixObject(feature_matrix_objects, index2label)
from flexible_clustering_tree import StringAggregation
from sklearn.cluster import KMeans

multi_clustering_operator = MultiClusteringOperator([
    ClusteringOperator(level=0, n_cluster=-1, instance_clustering=StringAggregation()),
    ClusteringOperator(level=1, n_cluster=3, instance_clustering=KMeans(n_clusters=3))
])
f_clustering = FlexibleClustering(max_depth=3)
data2cluster = f_clustering.fit_transform(multi_feature_input, multi_clustering_operator)
html_doc = f_clustering.clustering_tree.to_html()

# Visualized output into html file with D3.js
with open('string_aggregation_example.html', 'w') as f:
    f.write(html_doc)

# get object, which can be used as an input of Pandas
import pandas
table_objects = f_clustering.clustering_tree.to_objects()
print(pandas.DataFrame(table_objects['cluster_information']))
print(pandas.DataFrame(table_objects['leaf_information']))



