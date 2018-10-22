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

Both are possible `flexible-clustering-tree`!

# Contribution

- Easy interface(scikit-learn way) from data(feature matrix) into a tree viewer
- Possible to make various clustering algorithms ensemble
- Possible to set various feature types


# setup

```
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
 



