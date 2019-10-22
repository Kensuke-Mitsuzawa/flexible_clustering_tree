#! -*- coding: utf-8 -*-
# matrix object
from numpy import ndarray, median, vstack
from scipy.sparse.csr import csr_matrix
import numpy
# typing
from typing import List, Tuple, Dict, Union
# data object
from flexible_clustering_tree.models import \
    ClusterObject, ClusteringOperator, \
    MultiClusteringOperator, \
    MultiFeatureMatrixObject, ClusterTreeObject
# else
from sqlitedict import SqliteDict
from itertools import groupby
import os
import tempfile
import uuid
import traceback
from itertools import chain
from collections import Counter
# logger
from flexible_clustering_tree.logger import logger


class TempNodeInfo(object):
    __slots__ = ("data_ids", "center_data_id", "clustering_matrix_level")

    def __init__(self,
                 data_ids: List[int],
                 clustering_matrix_level: int,
                 center_data_id: int = None):
        self.data_ids = data_ids
        self.center_data_id = center_data_id
        self.clustering_matrix_level = clustering_matrix_level


class RecursiveClustering(object):
    def __init__(self,
                 is_use_cache: bool=True,
                 path_cache_working_dir: str=tempfile.mkdtemp(),
                 path_cache_file: str=str(uuid.uuid4())):
        # depth parameter ##
        self.depth = 0
        # initial value for cluster number
        self.cluster_id_i = 0
        # initial value for cluster number
        self.threshold_ratio_auto_switch = 10.0
        # minimum number of data point in a cluster
        self.threshold_minimum_unique_vector = 10
        self.is_auto_switch = True
        # dict object for child-node: parent-node #
        self.dict_child_id2parent_id = {}  # type: Dict[int,int]
        # dict object to save clustering result at depth level = i
        self.dict_depth2clustering_result = {}  # type: Dict[int,Dict[int, ClusterObject]]

        if is_use_cache:
            path_cache_file = os.path.join(path_cache_working_dir, path_cache_file)
            self.cache_dict_obj = SqliteDict(path_cache_file)
        else:
            self.cache_dict_obj = {}

    ###########################################################################################

    @staticmethod
    def detect_invalid_vector(input_matrix: Union[ndarray, csr_matrix],
                              dict_index2label: Dict[int, str],
                              limit_diff_max_min: int = 1000) -> List[int]:
        """It detects invalid vector. Here "invalid vector" means
        1. there is huge value gap between max value and min value.
        2. there is nan in a vector
        3. there is inf in a vector

        :param input_matrix: input matrix
        :param dict_index2label: a dict object of index number and its label
        :param limit_diff_max_min: threshold of value gap between min and max
        """
        if isinstance(input_matrix, csr_matrix):
            matrix_obj = input_matrix.toarray()
        elif isinstance(input_matrix, ndarray):
            matrix_obj = input_matrix
        else:
            raise Exception(
                "The input matrix object is {}. It expects ndarray or csr_matrix".format(type(input_matrix)))

        seq_invalid_index = []  # type: List[int]
        for vector_index, vector in enumerate(matrix_obj):
            # nan check
            if numpy.any(numpy.isnan(vector)):
                logger.warning('Skip vector-id={}, tag={} because it has nan value'.format(
                    vector_index, dict_index2label[vector_index]))
                seq_invalid_index.append(vector_index)
                continue
            # inf check
            elif not numpy.all(numpy.isfinite(vector)):
                logger.warning('Skip vector-id={}, tag={} because it has infinite value'.format(
                    vector_index, dict_index2label[vector_index]))
                seq_invalid_index.append(vector_index)
                continue
            else:
                # value gap check
                diff_max_min_matrix = abs(vector.min()) + abs(vector.max())
                if diff_max_min_matrix > limit_diff_max_min:
                    logger.warning('Skip text-id={}\
                    because it has large gap in min value and max value.\
                    gap={}, gap_limit={}'.format(vector_index, diff_max_min_matrix, limit_diff_max_min))
                    continue
                else:
                    continue
        else:
            pass

        return seq_invalid_index

    @staticmethod
    def filter_out_invalid_vector_matrix(seq_invalid_vector_index: List[int],
                                         input_matrix: Union[ndarray, csr_matrix]
                                         ) -> Union[ndarray, csr_matrix]:
        """It filters out invalid vector and re-set index number of a matrix.
        """
        if isinstance(input_matrix, csr_matrix):
            matrix_obj = input_matrix.toarray()
        elif isinstance(input_matrix, ndarray):
            matrix_obj = input_matrix
        else:
            raise Exception(
                "The input matrix object is {}. It expects ndarray or csr_matrix".format(type(input_matrix)))

        updated_matrix_obj = numpy.delete(matrix_obj, seq_invalid_vector_index, 0)

        if isinstance(input_matrix, csr_matrix):
            return csr_matrix(updated_matrix_obj)
        elif isinstance(input_matrix, ndarray):
            return updated_matrix_obj
        else:
            raise Exception(
                "The input matrix object is {}. It expects ndarray or csr_matrix".format(type(input_matrix)))

    @staticmethod
    def filter_out_invalid_vector_dict(seq_invalid_vector_index: List[int],
                                       dict_index2label: Dict[int, str]
                                       ) -> Dict[int, str]:
        """It filters out invalid vector and re-set index number of a matrix.
        """
        i = 0
        updaed_dict_index2tag = {}
        for vector_index, tag_t in dict_index2label.items():
            if vector_index in seq_invalid_vector_index:
                continue
            else:
                updaed_dict_index2tag[i] = tag_t
                i += 1
        else:
            pass

        return updaed_dict_index2tag

    def construct_matrix_and_dict_index(self,
                                        seq_main_vector: List[ndarray],
                                        dict_index2label: Dict[int, str]
                                        ) -> Tuple[Union[csr_matrix, ndarray], List[int]]:
        """It generates matrix(ndarray) from list[ndarray]. At the same time, it checks and removes invalid vector.
        """
        main_matrix_obj = numpy.array(seq_main_vector, dtype=numpy.double)
        seq_invalid_index_main_matrix = self.detect_invalid_vector(main_matrix_obj, dict_index2label)
        set_invalid_vector_index = seq_invalid_index_main_matrix

        if len(set_invalid_vector_index) > 0:
            main_matrix_obj = self.filter_out_invalid_vector_matrix(set_invalid_vector_index, main_matrix_obj)
        else:
            pass

        return main_matrix_obj, set_invalid_vector_index

    ###########################################################################################

    @staticmethod
    def generate_subset_matrix(feature_matrix: Union[csr_matrix, ndarray],
                               seq_index_subset_matrix: List[int])->Tuple[Union[ndarray], Dict[int, int]]:
        """It generates sub-matrix from the original given matrix.
        """
        seq_stack_vector = [None] * len(seq_index_subset_matrix)
        matrix_type = None
        for i, sub_matrix_index in enumerate(seq_index_subset_matrix):
            if isinstance(feature_matrix, csr_matrix):
                seq_stack_vector[i] = feature_matrix.getrow(sub_matrix_index).toarray()
                matrix_type = 'csr_matrix'
            elif isinstance(feature_matrix, ndarray):
                seq_stack_vector[i] = feature_matrix[sub_matrix_index]
                matrix_type = 'ndarray'
            else:
                raise NotImplementedError()

        if matrix_type == 'csr_matrix':
            subset_adjacency_matrix = numpy.concatenate(seq_stack_vector)
        elif matrix_type == 'ndarray':
            subset_adjacency_matrix = numpy.array(seq_stack_vector)
        else:
            raise Exception()

        dict_sub_matrix_index2original_matrix_index = {
            i: index_original_matrix for i, index_original_matrix in enumerate(seq_index_subset_matrix)
        }
        return subset_adjacency_matrix, dict_sub_matrix_index2original_matrix_index

    @staticmethod
    def get_median_of_clusters(seq_local_clustering_result: Dict[int, TempNodeInfo]) -> float:
        """computes median of #data in each cluster.
        """
        seq_cluster_size = [len(_.data_ids) for _ in seq_local_clustering_result.values()]
        median_cluster_size = median(seq_cluster_size)  # type: float
        return median_cluster_size

    @staticmethod
    def count_unique_vectors(target_matrix: Union[csr_matrix, ndarray])->int:
        if isinstance(target_matrix, csr_matrix):
            target_matrix = target_matrix.toarray()
        try:
            matrix_size = vstack({tuple(row) for row in target_matrix}).shape
        except Exception as e:
            logger.error(e, type(target_matrix), target_matrix)
            raise Exception(traceback.extract_stack())
        return matrix_size[0]

    @staticmethod
    def __check_result_distinct(this_level: List[Tuple[int, ClusterObject]])->bool:
        """method for debug. It checks duplication of data-id"""
        seq_ids = Counter(list(chain.from_iterable([t_obj[1].data_ids for t_obj in this_level])))
        for instance_id, freq in dict(seq_ids).items():
            if not freq == 1:
                raise Exception()
        return True

    ###########################################################################################

    @staticmethod
    def get_average_vector(matrix_object: ndarray)->ndarray:
        """it computes average for the given matrix.
        """
        return numpy.mean(numpy.array(matrix_object), axis=0)

    @staticmethod
    def func_key_tuple_dataid_clusterid(t: Tuple[int, int])->int:
        return t[1]

    def generate_sub_clusters(self,
                              clustering_operator: ClusteringOperator,
                              this_level: List[Tuple[int, ClusterObject]]
                              ) -> Dict[int, TempNodeInfo]:
        """It starts recursive-clustering.

        :param clustering_operator
        :param this_level: A sequence of () node object in a layer during BFS processing.
        :return: {cluster-id: TempNodeInfo}
        """
        # key is parent-cluster-id, value is tuple of (list of data-id, most center data-id)
        dict_local_clustering_result = {}  # type: Dict[int, TempNodeInfo]

        for parent_cluster_id, cluster_info_obj in this_level:
            assert isinstance(cluster_info_obj, ClusterObject)
            if cluster_info_obj.feature_matrix.shape == (1, 1):
                logger.debug(msg='Impossible to run clustering anymore. Dpeth={}'.format(self.depth))
                continue
            if cluster_info_obj.feature_matrix.shape[0] <= clustering_operator.n_cluster:
                logger.debug(msg='Impossible to run clustering anymore. Dpeth={}'.format(self.depth))
                continue

            core_obj = clustering_operator.instance_clustering
            try:
                core_obj.fit(X=cluster_info_obj.feature_matrix)
            except Exception as e:
                error = traceback.format_exc()
                raise AttributeError(
                    "Exception={}. \
                    clustering generator does NOT have fit method. \
                    Check your function. Traceback={}".format(str(e), error))

            # if #label == 1, no clustering process
            if len(set(core_obj.labels_)) == 1:
                continue
            # dict object to convert local-cluster-id: global-cluster-id
            dict_local_clusterid2common_cluster_id = {}
            for cluster_id in set(core_obj.labels_):
                dict_local_clusterid2common_cluster_id[cluster_id] = self.cluster_id_i
                self.cluster_id_i += 1

            # child-parent relationship between clusters
            for local_cluster_id in set(core_obj.labels_):
                self.dict_child_id2parent_id[
                    dict_local_clusterid2common_cluster_id[local_cluster_id]] = parent_cluster_id

            # saves clustering result at this depth level
            # run groupby to make a structure of Dict[cluster-id, List[tag-node-id]]
            seq_t_local_cluster_id = [
                (cluster_info_obj.dict_submatrix_index2original_matrix_index[int(local_instance_id)], cluster_id)
                for local_instance_id, cluster_id in enumerate(core_obj.labels_)]

            for local_cluster_id, g_obj in groupby(
                    sorted(seq_t_local_cluster_id, key=self.func_key_tuple_dataid_clusterid),
                    key=self.func_key_tuple_dataid_clusterid):
                temp_info = TempNodeInfo([t[0] for t in g_obj], cluster_info_obj.matrix_depth_level, None)
                # here key id of dict is global cluster-id in a tree
                dict_local_clustering_result[dict_local_clusterid2common_cluster_id[local_cluster_id]] = temp_info

        return dict_local_clustering_result

    def __get_feature_matrix_in_next_level(self, multi_feature_object: MultiFeatureMatrixObject):
        """It gets feature-matrix in the next node level."""
        if self.depth + 1 in multi_feature_object.dict_level2matrix_obj:
            target_matrix = multi_feature_object.dict_level2matrix_obj[self.depth + 1]
            matrix_depth_level = self.depth + 1
        else:
            _max = max(multi_feature_object.dict_level2matrix_obj.keys())
            target_matrix = multi_feature_object.dict_level2matrix_obj[_max]
            matrix_depth_level = _max
        return target_matrix, matrix_depth_level

    def get_clustering_class_name(self, multi_clustering_operator: MultiClusteringOperator)->str:
        try:
            if self.depth in multi_clustering_operator.dict_level2operator:
                clustering_label = multi_clustering_operator.dict_level2operator[self.depth]\
                    .instance_clustering.__class__.__name__
            else:
                clustering_label = multi_clustering_operator.get_default_clustering_algorithm()\
                    .instance_clustering.__class__.__name__
        except Exception as e:
            logger.warning(e)
            clustering_label = "undefined"

        return clustering_label

    def post_process_clustering(self,
                                dict_local_clustering_result: Dict[int, TempNodeInfo],
                                multi_feature_object: MultiFeatureMatrixObject,
                                multi_clustering_operator: MultiClusteringOperator) -> List[Tuple[int, ClusterObject]]:
        """runs post-process after clustering. This method is called in each layer of a tree.

        :param dict_local_clustering_result: output of self.generate_sub_clusters()
        :param multi_feature_object:
        :param multi_clustering_operator:
        :return: node object to pass into next clustering level. (cluster-id, cluster-node-object)
        """
        seq_stack_next_level = [None] * len(list(dict_local_clustering_result.keys()))
        list_i = 0
        # computes median & variance
        median_cluster_size = self.get_median_of_clusters(dict_local_clustering_result)

        for cluster_id, t_matrix_index_cluster_element in dict_local_clustering_result.items():
            diff_ratio_against_median = len(t_matrix_index_cluster_element.data_ids) / median_cluster_size
            if self.is_auto_switch:
                if diff_ratio_against_median <= self.threshold_ratio_auto_switch:
                    # if auto-switch is on && diff from median is lower than threshold; then use a matrix in next level
                    target_matrix, matrix_depth_level = self.__get_feature_matrix_in_next_level(multi_feature_object)
                else:
                    # if auto-switch is on && diff from median is bigger than threshold; then use first matrix.
                    target_matrix = multi_feature_object.dict_level2matrix_obj[0]
                    matrix_depth_level = 0
            else:
                # use a matrix in next level
                target_matrix, matrix_depth_level = self.__get_feature_matrix_in_next_level(multi_feature_object)

            subset_matrix, dict_submatrix_ind2original_matrix_ind = self.generate_subset_matrix(
                feature_matrix=target_matrix,
                seq_index_subset_matrix=t_matrix_index_cluster_element.data_ids)

            clustering_label = self.get_clustering_class_name(multi_clustering_operator)
            cluster_info_object = ClusterObject(
                cluster_id=cluster_id,
                parent_cluster_id=self.dict_child_id2parent_id[cluster_id],
                data_ids=t_matrix_index_cluster_element.data_ids,
                average_vector=self.get_average_vector(subset_matrix),
                matrix_depth_level=matrix_depth_level,
                feature_matrix=subset_matrix,
                dict_submatrix_index2original_matrix_index=dict_submatrix_ind2original_matrix_ind,
                clustering_label=clustering_label
            )
            # saves nodes at this level
            self.dict_depth2clustering_result[self.depth][cluster_id] = cluster_info_object
            # judge whether it should run clustering in the next level
            if self.count_unique_vectors(subset_matrix) > self.threshold_minimum_unique_vector:
                # there is enough data point than threshold in a node -> run clustering in the next level
                seq_stack_next_level[list_i] = (cluster_id, cluster_info_object)
            else:
                # else; ends here and skip clustering in the next level
                logger.debug(
                    msg='it does NOT run clustering on Cluster_id={}. Skip.'.format(cluster_info_object.cluster_id))
            list_i += 1

        # get clustering setting in the next level
        if self.depth+1 in multi_clustering_operator.dict_level2operator:
            cluster_operator_obj = multi_clustering_operator.dict_level2operator[self.depth+1]
        else:
            cluster_operator_obj = multi_clustering_operator.dict_level2operator[0]

        this_level = [t for t in seq_stack_next_level
                      if t is not None
                      and t[1].feature_matrix.shape[0] > cluster_operator_obj.n_cluster]
        self.__check_result_distinct(this_level)

        return this_level

    def __generate_first_layer(self,
                               feature_matrix_object: Union[csr_matrix, ndarray],
                               root_node_id: int=-1)->List[Tuple[int, ClusterObject]]:
        """it generates the first layer of a tree. At the first layer, no need to run clustering.
        Just it's okay to put feature matrix into a root node.

        :return: cluster node object in the first layer. (-1, cluster-node-object)
        """
        first_layer_matrix_obj = feature_matrix_object
        if isinstance(first_layer_matrix_obj, csr_matrix):
            dict_submatrix_index2original_matrix_index_level1 = {i: i for i in
                                                                 range(0, first_layer_matrix_obj.shape[0])}
        else:
            dict_submatrix_index2original_matrix_index_level1 = {i: i for i in range(0, len(first_layer_matrix_obj))}

        initial_clusterinformation_obj = ClusterObject(cluster_id=-1,
                                                       parent_cluster_id=-1,
                                                       data_ids=list(range(0, feature_matrix_object.shape[0])),
                                                       feature_matrix=feature_matrix_object,
                                                       dict_submatrix_index2original_matrix_index=dict_submatrix_index2original_matrix_index_level1,
                                                       average_vector=self.get_average_vector(feature_matrix_object),
                                                       clustering_label=None)
        this_level = [(root_node_id, initial_clusterinformation_obj)]  # type: List[Tuple[int,ClusterObject]]
        # child-parent relationship
        self.dict_child_id2parent_id = {}
        self.dict_depth2clustering_result[self.depth] = {}
        return this_level

    def __generate_first_layer_given(self,
                                     seq_cluster_object_first_layer)->List[Tuple[int,ClusterObject]]:
        """It generates first layer of a tree.
        """
        this_level = seq_cluster_object_first_layer
        self.dict_child_id2parent_id = {
            t_parentid_cluster_obj[1].cluster_id: -1
            for t_parentid_cluster_obj in seq_cluster_object_first_layer}
        self.dict_depth2clustering_result = {
            self.depth: {t_parentid_cluster_obj[1].cluster_id: t_parentid_cluster_obj[1]
                         for t_parentid_cluster_obj in seq_cluster_object_first_layer}
        }
        self.depth += 1
        self.dict_depth2clustering_result[self.depth] = {}
        self.cluster_id_i = max([t_parentid_cluster_obj[1].cluster_id
                                 for t_parentid_cluster_obj in seq_cluster_object_first_layer]) + 1
        return this_level

    def run_recursive_clustering(self,
                                 multi_clustering_operator: MultiClusteringOperator,
                                 multi_feature_matrix_object: MultiFeatureMatrixObject,
                                 max_depth: int,
                                 is_auto_switch: bool = True,
                                 threshold_ratio_auto_switch: float = 10.0,
                                 threshold_minimum_unique_vector: int = 10,
                                 initial_cluster_id: int = 0) -> ClusterTreeObject:
        """It runs clustering recursively. To process each node in a tree, it runs with BFS(Breadth First Search) way

        :param multi_clustering_operator:
        :param multi_feature_matrix_object:
        :param max_depth: maximum depth of keeping clustering
        :param is_auto_switch: boolean flag of outlier detection.
        To detect outlier cluster, it uses median size of all cluster in a level.
        If a cluster size is much bigger than median, it is regarded as outlier.
        :param threshold_ratio_auto_switch: threshold value of is_auto_switch
        :param threshold_minimum_unique_vector: minimum value to run clustering.
        If #data-id in a cluster is smaller than this value, it avoid clustering.
        :param initial_cluster_id: usually 0.
        :return: a tree which has cluster-nodes
        """
        # depth #
        self.depth = 0
        # node-id of the root node #
        root_node_id = -1
        # initial id to start counting cluster-node
        self.cluster_id_i = initial_cluster_id
        self.threshold_ratio_auto_switch = threshold_ratio_auto_switch
        self.threshold_minimum_unique_vector = threshold_minimum_unique_vector
        self.is_auto_switch = is_auto_switch

        # generate a node in 1st layer(without clustering)
        this_level = self.__generate_first_layer(multi_feature_matrix_object.dict_level2matrix_obj[0],
                                                 root_node_id)

        while this_level:
            logger.info(msg='Processing depth level = {}'.format(self.depth))
            logger.info(msg='This level has {} objects to be processed.'.format(len(this_level)))
            # ======================================================================================================
            instance_clustering = multi_clustering_operator.get_clustering_instance(level=self.depth)
            dict_local_clustering_result = self.generate_sub_clusters(instance_clustering, this_level)
            # ======================================================================================================
            if self.depth == 0 and len(dict_local_clustering_result) == 0:
                raise Exception('un-expected error. No cluster at level=0.')
            this_level = self.post_process_clustering(dict_local_clustering_result,
                                                      multi_feature_matrix_object,
                                                      multi_clustering_operator)
            self.depth += 1
            if self.depth == max_depth:
                break
            else:
                self.dict_depth2clustering_result[self.depth] = {}

        tree_object = ClusterTreeObject(
            dict_child_id2parent_id=self.dict_child_id2parent_id,
            dict_depth2clustering_result=self.dict_depth2clustering_result,
            multi_matrix_object=multi_feature_matrix_object,
            multi_clustering_object=multi_clustering_operator)
        return tree_object
