# flexible-clustering-tree

- - -

# What's this?

In the context of __clustering__ task, `flexible-clustering-tree` provides you __easy__ and __controllable__ clustering framework.

![](https://user-images.githubusercontent.com/1772712/47308081-9980cd00-d66b-11e8-98c0-a275db021cd7.gif) 

# Background

Let's suppose, you have huge data. You'd like to observe data as easy as possible.

Hierarchical clustering is a way to see clustering tree.
However, hierarchical clustering tends to fall into local optimization.

So, you need other clustering method.
But at the same time, you wanna observe your data with tree structure style.

here, `flexible-clustering-tree` could give you simple way from data into tree viewer(d3 based)

You could set any kinds of clustering algorithm such as Kmeans, DBSCAN, Spectral-Clustering.

## Multi feature and Multi clustering

During making a tree, you might want use various kind of clustering algorithm.
For example, you use Kmeans for the 1st later of a tree, and DBSCAN for the 2nd layer of a tree.

And you might also use various kind of feature type depending on a layer of a tree. 
For example, in the context of document clustering, "title" of news for the 1st layer, and "whole text" for the 2nd layer.

The example below, this is a clustering tree of 20-news data set.

- 1st layer(red highlight) is done with HDBSCAN clustering, and feature is dense vector of `Subject` text, which is converted by word2vec model.
- 2nd layer(blue highlight) is done with Kmeans, and feature is sparse vector of whole text(BOW). 

You could design your clustering tree as you want! 

![](https://user-images.githubusercontent.com/1772712/47308468-abaf3b00-d66c-11e8-9a08-26facc39e80e.png)

Both are possible `flexible-clustering-tree`!

# Contribution

- Easy interface(scikit-learn way) from data(feature matrix) into a tree viewer
- Possible to make various clustering algorithms ensemble
- Possible to set various feature types

# How to use?

```python
from flexible_clustering_tree import FeatureMatrixObject, MultiFeatureMatrixObject
from flexible_clustering_tree import ClusteringOperator, MultiClusteringOperator
from flexible_clustering_tree import FlexibleClustering
import numpy
import codecs

# set feature matrix
# an input of 1st layer is list of string
input_string = ['d'] * 10 + ['e'] * 10 + ['c'] * 10 + ['a'] * 10 + ['b'] * 10 + ['f'] * 50
f_obj_1st = FeatureMatrixObject(0, feature_strings=input_string)
# an input of 2nd layer is the dense matrix (100, 300)
f_obj_2nd = FeatureMatrixObject(1, matrix_object=numpy.random.rand(100, 300))
# an input of 3rd layer is the dense matrix (100, 50)
f_obj_3rd = FeatureMatrixObject(2, matrix_object=numpy.random.rand(100, 50))
dict_index2label = {i: "label-{}".format(i) for i in range(0, 100)}
multi_feature_matrix = MultiFeatureMatrixObject(
    [f_obj_1st, f_obj_2nd, f_obj_3rd],
    dict_index2label=dict_index2label
)

# set clustering operation
from sklearn.cluster.k_means_ import KMeans
from hdbscan.hdbscan_ import HDBSCAN
from flexible_clustering_tree import StringAggregation
c_operation_1st = ClusteringOperator(0, -1, StringAggregation())
c_operation_2nd = ClusteringOperator(1, 10, KMeans(10))
c_operation_3rd = ClusteringOperator(2, -1, HDBSCAN())
multi_clustering = MultiClusteringOperator([c_operation_1st, c_operation_2nd])

# run flexible clustering
clustering_runner = FlexibleClustering(max_depth=5)
index2cluster_no = clustering_runner.fit_transform(multi_feature_matrix, multi_clustering)
html = clustering_runner.clustering_tree.to_html()

# output to html
with codecs.open("out.html", "w", "utf-8") as f:
    f.write(html)
    
# you're also able to generate tables via Pandas.
import pandas
table_objects = clustering_runner.clustering_tree.to_objects()
print(pandas.DataFrame(table_objects['cluster_information']))
print(pandas.DataFrame(table_objects['leaf_information']))
```

The output of pandas table is below.

The relation-table of clusters is in `cluster_information`.

```
    cluster_id  parent_id  depth_level  clustering_method
0            0         -1            1  StringAggregation
1            1         -1            1  StringAggregation
2            2         -1            1  StringAggregation
3            3         -1            1  StringAggregation
4            4         -1            1  StringAggregation
5            5         -1            1  StringAggregation
6            6          5            2             KMeans
7            7          5            2             KMeans
8            8          5            2             KMeans
9            9          5            2             KMeans
10          10          5            2             KMeans
11          11          5            2             KMeans
12          12          5            2             KMeans
13          13          5            2             KMeans
14          14          5            2             KMeans
15          15          5            2             KMeans
```

The relation-table of leaf nodes is in `leaf_information`.

```
    leaf_id  cluster_id     label  args
0         0           0   label-0  None
1         1           0   label-1  None
2         2           0   label-2  None
3         3           0   label-3  None
4         4           0   label-4  None
..      ...         ...       ...   ...
95       95          14  label-95  None
96       96           8  label-96  None
97       97          13  label-97  None
98       98          10  label-98  None
99       99          12  label-99  None
[100 rows x 4 columns]
```


You could see examples at `/examples`.


# setup

```bash
pip install flexible_clustering_tree
```

or close this repository 

```bash
python setup.py install
```

# For Developers

## Environment

- Python >= 3.x
    
## Dev/Test environment by Docker

You build dev/test environment with Docker container.
Here is simple procedure,

1. build docker image
2. start docker container
3. run test in the container

```bash
$ cd tests
$ docker-compose build
$ docker-compose up
$ docker run --name test-container -v `pwd`:/codes/flexible-clustering-tree/ -dt tests_dev_env
$ docker exec -it test-container python /codes/flexible-clustering-tree/setup.py test
```

If you're using pycharm professional edition, you could call a docker-compose file as Python interpreter.
 



