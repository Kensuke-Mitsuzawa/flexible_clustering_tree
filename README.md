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

# set feature matrix
f_obj_1st = FeatureMatrixObject(0, numpy.random.rand(500, 600))
f_obj_2nd = FeatureMatrixObject(1, numpy.random.rand(500, 300))
f_obj_3rd = FeatureMatrixObject(2, numpy.random.rand(500, 50))
dict_index2label = {i: "label-{}".format(i) for i in range(0, 500)}
multi_feature_matrix = MultiFeatureMatrixObject(
    [f_obj_1st, f_obj_2nd, f_obj_3rd],
    dict_index2label=dict_index2label
)

# set clustering operation
from sklearn.cluster.k_means_ import KMeans
from hdbscan.hdbscan_ import HDBSCAN
c_operation_1st = ClusteringOperator(0, 10, KMeans(10))
c_operation_2nd = ClusteringOperator(1, 5, KMeans(5))
multi_clustering = MultiClusteringOperator([c_operation_1st, c_operation_2nd])

# run flexible clustering
clustering_runner = FlexibleClustering(max_depth=3)
index2cluster_no = clustering_runner.fit_transform(multi_feature_matrix, multi_clustering)
html = clustering_runner.clustering_tree.to_html()

# output to html
with codecs.open("out.html", "w", "utf-8") as f:
    f.write(html)
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
 



