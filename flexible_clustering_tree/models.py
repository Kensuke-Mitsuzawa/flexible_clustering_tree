#! -*- coding: utf-8 -*-
from numpy import ndarray
from scipy.sparse.csr import csr_matrix
# typing
from typing import List, Dict, Optional, Tuple, Union, Any
# jinja2
from jinja2 import Environment, BaseLoader
# else
from itertools import groupby
from collections import Counter
from flexible_clustering_tree.logger import logger
from copy import deepcopy
import pkgutil
import json


class FeatureMatrixObject(object):
    __slots__ = ('level', 'matrix_object', 'feature_strings', 'feature_object')

    def __init__(self,
                 level: int,
                 matrix_object: Union[ndarray, csr_matrix] = None,
                 feature_strings: List[str] = None):
        self.level = level
        self.matrix_object = matrix_object
        self.feature_strings = feature_strings
        if matrix_object is None and feature_strings is None:
            raise Exception('either of matrix_object or feature_strings should be given')
        elif matrix_object is None:
            self.feature_object = feature_strings
        elif feature_strings is None:
            assert isinstance(matrix_object, (csr_matrix, ndarray))
            self.feature_object = matrix_object
        elif matrix_object is not None and feature_strings is not None:
            assert len(feature_strings) == len(matrix_object)

    def get_index_size(self)->int:
        if isinstance(self.feature_object, list):
            return len(self.feature_object)
        elif isinstance(self.feature_object, ndarray):
            return self.feature_object.shape[0]
        else:
            raise Exception()

    def __str__(self):
        __feature_type = type(self.feature_object)
        __ = "@level={} feature-type={}".format(self.level, __feature_type)
        if self.matrix_object is not None:
            __ += " matrix object with {} * {}".format(self.level, *self.matrix_object.shape)

        return __


class MultiFeatureMatrixObject(object):
    """Class for 1-input of recursive-clustering"""
    __slots__ = ("dict_level2feature_obj", "dict_index2label", "dict_index2attributes", "matrix_first_layer")

    def __init__(self,
                 matrix_objects: List[FeatureMatrixObject],
                 dict_index2label: Dict[int, str],
                 dict_index2attributes: Optional[Dict[int, Dict]]=None):
        """Init the class.

        :param matrix_objects: matrix object of a layer
        :param dict_index2label: key is index number of a matrix, value is label corresponding to a index.
        :param dict_index2attributes: key is index number of a matrix, value is any attributes.
        You could use it to put information anything you want.
        """
        assert isinstance(matrix_objects, list)
        assert isinstance(dict_index2label, dict)
        self.__check_level_zero(matrix_objects)
        matrix_index_sizes = [m_o.get_index_size() for m_o in matrix_objects]
        if len(set(matrix_index_sizes)) != 1:
            raise Exception("The given matrix size is not same. Sizes of the given matrix sizes are {}".format(
                set(matrix_index_sizes)))
        self.__check_match_index_size(matrix_index_sizes, dict_index2label)

        self.dict_level2feature_obj = {m_obj.level: m_obj.feature_object for m_obj in matrix_objects}
        self.dict_index2label = dict_index2label
        self.dict_index2attributes = dict_index2attributes

    @staticmethod
    def __check_level_zero(matrix_objects: List[FeatureMatrixObject]):
        for m_o in matrix_objects:
            if m_o.level == 0:
                return True
        raise Exception("There must be FeatureMatrixObject level = 0")

    @staticmethod
    def __check_match_index_size(matrix_index_sizes: List[int], dict_index2label: Dict[int, str]):
        _matrix_index_sizes = matrix_index_sizes[0]
        for i in range(0, _matrix_index_sizes):
            if i not in dict_index2label:
                raise Exception("label for index={} is not defined.".format(i))

    def __str__(self):
        return "#level={} #data-point={}".format(len(self.dict_level2feature_obj), len(self.dict_index2label))


class ClusteringOperator(object):
    def __init__(self,
                 level: int,
                 n_cluster: int,
                 instance_clustering):
        if not hasattr(instance_clustering, "fit"):
            raise Exception("instance_clustering MUST have method 'fit'")

        self.level = level
        self.n_cluster = n_cluster
        self.instance_clustering = instance_clustering

    def generate_clustering_instance(self):
        return self.instance_clustering

    def to_dict(self)->Dict[str, Any]:
        return {
            "level": self.level,
            "n_cluster": self.n_cluster,
            "instance_clustering": self.instance_clustering
        }

    def __str__(self):
        return "@level={}".format(self.level)


class MultiClusteringOperator(object):
    def __init__(self,
                 clustering_operators: List[ClusteringOperator]):
        self.__check_level_zero(clustering_operators)
        self.clustering_operators = clustering_operators
        self.dict_level2operator = {c_operator.level: c_operator
                                    for c_operator in clustering_operators}
        self.default_clustering = self.dict_level2operator[max(self.dict_level2operator.keys())]

    @staticmethod
    def __check_level_zero(clustering_operation: List[ClusteringOperator]):
        for m_o in clustering_operation:
            if m_o.level == 0:
                return True
        raise Exception("There must be FeatureMatrixObject level = 0")

    def get_clustering_instance(self, level: int):
        """It returns an instance which works clustering.
        If the given level is not in set, it returns the last instance.

        :param level: level of depth
        """
        if level in self.dict_level2operator:
            return self.dict_level2operator[level]
        else:
            return self.dict_level2operator[max(self.dict_level2operator.keys())]

    def get_default_clustering_algorithm(self)->ClusteringOperator:
        """it sets default clustering algorithm when level is not defined.
        by default, it uses clustering algorithm in max(level)
        """
        return self.default_clustering

    def to_dict(self):
        return {
            "operators": [c.to_dict() for c in self.clustering_operators]
        }


class ClusterObject(object):
    """Class for saving one-node of a tree"""
    __types__ = ('cluster_id', 'parent_cluster_id', 'data_ids',
                 'average_vector', 'child_cluster_ids', 'feature_matrix', 'matrix_depth_level',
                 'feature_type', 'feature_object',
                 'dict_submatrix_index2original_matrix_index', 'clustering_label')

    def __init__(self,
                 cluster_id: int,
                 parent_cluster_id: int,
                 average_vector: ndarray,
                 data_ids: List[int],
                 matrix_depth_level: int=None,
                 child_cluster_ids: List[int]=None,
                 feature_object: Union[List[str], ndarray] = None,
                 clustering_label: str=None,
                 dict_submatrix_index2original_matrix_index: Dict[int, int] = None,
                 feature_matrix: ndarray = None,):
        """

        :param cluster_id: id of this cluster node
        :param parent_cluster_id: id of parent cluster node
        :param data_ids:
        :param average_vector:
        :param matrix_depth_level: Depth level of matrix used for this cluster node
        :param child_cluster_ids:
        :param feature_matrix: matrix object
        :param clustering_label: name of clustering algorithm which is used to generate this node
        :param dict_submatrix_index2original_matrix_index:
        """
        if isinstance(feature_object, ndarray):
            self.feature_matrix = feature_object
            self.feature_type = ndarray
        elif isinstance(feature_object, list) and all([isinstance(_, str) for _ in feature_object]):
            self.feature_type = str
        else:
            raise Exception('Unexpected error. feature object expects either of str or ndarray.')

        self.cluster_id = cluster_id
        self.parent_cluster_id = parent_cluster_id
        self.data_ids = data_ids
        self.average_vector = average_vector
        self.child_cluster_ids = child_cluster_ids
        self.feature_matrix = feature_matrix
        self.matrix_depth_level = matrix_depth_level
        self.dict_submatrix_index2original_matrix_index = dict_submatrix_index2original_matrix_index
        self.clustering_label = clustering_label
        self.feature_object = feature_object

    def __str__(self):
        __ = "cluster-id={} feature-type={}".format(self.cluster_id, self.feature_type)
        if isinstance(self.feature_type, ndarray):
            __ += " with {} * {} matrix".format(*self.feature_matrix.shape)
        else:
            pass
        return  __


class ClusterTreeObject(object):
    """Class for saving tree information as a result of clustering"""
    def __init__(self,
                 dict_child_id2parent_id: Dict[int, int],
                 dict_depth2clustering_result: Dict[int, Dict[int, ClusterObject]],
                 multi_matrix_object: MultiFeatureMatrixObject,
                 multi_clustering_object: MultiClusteringOperator):
        """
        :param dict_child_id2parent_id: {child-cluster-id: parent-cluster-id}
        :param dict_depth2clustering_result: {depth-level: {cluster-id: {node-obj}}}
        :param multi_matrix_object:
        :param multi_clustering_object:
        """
        self.multi_matrix_object = multi_matrix_object
        self.dict_child_id2parent_id = dict_child_id2parent_id
        self.dict_depth2clustering_result = dict_depth2clustering_result
        self.dict_parent_cluster_id2children_cluster_id = \
            self.__generate_parent2children()  # type: Dict[int, List[int]]
        self.dict_cluster_id2cluster_obj = self.__generate_cluster_id2cluster_obj()  # type: Dict[int, ClusterObject]

        self.dict_cluster_id2cluster_agg_info = self.aggregate_node_information()  # type: Dict[int, Dict[str, Any]]
        self.updated_dict_child2parent = self.reshape_tree()  # type: Dict[int, int]

    def child_cluster_id2parent_cluster_id(self, child_cluster_id: int)->Union[int, bool]:
        """it returns parent-cluster-id
        """
        return self.dict_child_id2parent_id[child_cluster_id] \
            if child_cluster_id in self.dict_child_id2parent_id else False

    def __generate_parent2children(self) -> Dict[int, List[int]]:
        """it makes relationship parent-cluster-id: child-cluster-id

        :return: {parent-cluster-id: [child-cluster-id]}
        """
        d_parent_cluster_node2children = {}
        for p_node_id, g_obj in groupby(
                sorted(self.dict_child_id2parent_id.items(),
                       key=lambda t_parent_and_child: t_parent_and_child[1]),
                key=lambda t_parent_and_child: t_parent_and_child[1]):
            d_parent_cluster_node2children[p_node_id] = [t_pair[0] for t_pair in g_obj]
        else:
            return d_parent_cluster_node2children

    def __generate_cluster_id2cluster_obj(self) -> Dict[int, ClusterObject]:
        """it makes relationship of cluster-id: cluster-obj

        :return: {cluster-node-id: cluster-object}
        """
        cluster_id2cluster_obj = {cluster_obj.cluster_id: cluster_obj
                                  for _s_cluster_obj in self.dict_depth2clustering_result.values()
                                  for cluster_obj in _s_cluster_obj.values()}

        return cluster_id2cluster_obj

    @staticmethod
    def __update_child_cluster_nodes(dict_cluster_id2cluster_obj: Dict[int, ClusterObject],
                                     dict_parent_cluster_id2children_cluster_id: Dict[int, List[int]])->None:
        """it updates child-cluster information. This method is for updating, does not return values.

        :rtype: None
        """
        def __update(cluster_obj: ClusterObject, seq_children_node_id: List[int])->None:
            cluster_obj.child_cluster_ids = seq_children_node_id

        for cluster_node_id in dict_cluster_id2cluster_obj.keys():
            if cluster_node_id in dict_parent_cluster_id2children_cluster_id:
                __update(dict_cluster_id2cluster_obj[cluster_node_id],
                         dict_parent_cluster_id2children_cluster_id[cluster_node_id])
            else:
                pass
        else:
            pass

    # ---------------------------------------------------------------------------------------------------------

    def aggregate_node_information(self) -> Dict[int, Dict[str, Any]]:
        """Run aggregation of branch node information, especially "how many data-id the node has?"
        It updates with bottom-up way, in other words, it starts aggregation from leaf level nodes
        and go into branch level nodes.
        """
        def make_aggregation(_cluster_id: int,
                             _cluster_node_information: ClusterObject) -> None:
            """Aggregate node information.
            If _cluster_id is intermediate in a tree; then save information about nodes under the intermediate node.
            If _cluster_id is leaf; then saves information of the leaf node itself.
            """
            if _cluster_id in dict_parent_cluster_id2child_cluster_id:
                # if the _cluster_id is branch node
                # it computes frequency of labels
                _n_data_point = sum([len(self.dict_cluster_id2cluster_obj[c_id].data_ids)
                                     for c_id
                                     in dict_parent_cluster_id2child_cluster_id[_cluster_id]])
            else:
                # if the _cluster_id is leaf node
                _n_data_point = len(self.dict_cluster_id2cluster_obj[_cluster_id].data_ids)

            c_obj = Counter([self.multi_matrix_object.dict_index2label[d_id]
                             for d_id in _cluster_node_information.data_ids])
            dict_nodeid2agg_info[_cluster_id] = {
                "node-type": "branch",
                "#data": _n_data_point,
                "frequent_labels": c_obj.most_common(3)}

        # sort by cluster-id (reverse order), this can make it possible bottom up
        seq_cluster_node_information_obj = [
            t for t
            in sorted(self.dict_cluster_id2cluster_obj.items(), key=lambda t: t[0], reverse=True)
        ]  # type: List[Tuple[int, ClusterObject]]

        dict_nodeid2agg_info = {}
        t_parentid_g_obj = groupby(sorted(self.dict_child_id2parent_id.items(), key=lambda t: t[1]), key=lambda t: t[1])
        dict_parent_cluster_id2child_cluster_id = {
            parent_id: [t[0] for t in g_obj] for parent_id, g_obj in t_parentid_g_obj}  # type: Dict[int, List[int]]

        # call closure here
        for cluster_id, cluster_node_information in seq_cluster_node_information_obj:
            make_aggregation(_cluster_id=cluster_id, _cluster_node_information=cluster_node_information)

        dict_cluster_id2cluster_agg_info = {node_id: None for node_id in self.dict_cluster_id2cluster_obj.keys()}
        for node_id, cluster_node_information in seq_cluster_node_information_obj:
            dict_cluster_id2cluster_agg_info[node_id] = dict_nodeid2agg_info[node_id]
        return dict_cluster_id2cluster_agg_info

    # ---------------------------------------------------------------------------------------------------------
    # method to clean up a tree #

    def reshape_tree(self) -> Dict[int, int]:
        """Delete a cluster-node whose #child-cluster-node is only 1,
        and put the child-cluster-node into a parent cluster node.
        """
        def make_reshape_operation(_parent_cluster_id: int,
                                   _seq_child_cluster_id: List[int]) -> None:
            if len(_seq_child_cluster_id) != 1:
                tmp_dict_parent_clusterid2child_cluster_id[_parent_cluster_id] = _seq_child_cluster_id
            else:
                current_parent_cluster_id = _seq_child_cluster_id[0]
                while True:
                    if current_parent_cluster_id in dict_parent_clusterid2child_cluster_id:
                        seq_child_cluster_id_next_level = dict_parent_clusterid2child_cluster_id[
                            current_parent_cluster_id]
                        if len(seq_child_cluster_id_next_level) == 1:
                            # if cluster-size == 1; then go into deeply
                            set_stack_processed_id.add(current_parent_cluster_id)
                            set_stack_processed_id.add(seq_child_cluster_id_next_level[0])
                            current_parent_cluster_id = seq_child_cluster_id_next_level[0]
                        else:
                            # if cluster-size >= 2
                            set_stack_processed_id.add(current_parent_cluster_id)
                            tmp_dict_parent_clusterid2child_cluster_id[_parent_cluster_id] = [current_parent_cluster_id]
                            break
                    else:
                        # if it reaches leaf of a tree
                        set_stack_processed_id.add(current_parent_cluster_id)
                        tmp_dict_parent_clusterid2child_cluster_id[_parent_cluster_id] = [current_parent_cluster_id]
                        break

        # parent-node-cluster: child-node-clusters
        t_parentid_g_obj = groupby(sorted(self.dict_child_id2parent_id.items(), key=lambda t: t[1]), key=lambda t: t[1])
        dict_parent_clusterid2child_cluster_id = {parent_id: [t[0] for t in g_obj]
                                                  for parent_id, g_obj in t_parentid_g_obj}  # type: Dict[int,List[int]]

        tmp_dict_parent_clusterid2child_cluster_id = {}  # type: Dict[int,List[int]]
        set_stack_processed_id = set()
        for parent_cluster_id, seq_child_cluster_id in sorted(dict_parent_clusterid2child_cluster_id.items(),
                                                              key=lambda t: t[0]):
            if parent_cluster_id in set_stack_processed_id:
                continue
            make_reshape_operation(_parent_cluster_id=parent_cluster_id, _seq_child_cluster_id=seq_child_cluster_id)

        updated_dict_child2parent = {
            child_cluster_id: parent_cluster_id
            for parent_cluster_id, seq_child_cluster_id in tmp_dict_parent_clusterid2child_cluster_id.items()
            for child_cluster_id in seq_child_cluster_id}

        return updated_dict_child2parent

    # ---------------------------------------------------------------------------------------------------------
    # methods to show result into other data from #

    def __generate_branch_node_template(self,
                                        cluster_node_data: ClusterObject,
                                        dict_cluster_id2cluster_agg_info: Optional[Dict[int, Dict[str, Counter]]],
                                        sub_tree_information: Optional[List[Dict[str, Any]]] = None,
                                        is_child_with_underscore: bool=True) -> Dict[str, Any]:
        """
        """
        if sub_tree_information is None:
            # if target cluster node is branch
            children = [self.__generate_leaf_node_template(data_id, cluster_node_data)
                        for data_id in cluster_node_data.data_ids]  # type: List[Dict[str,Any]]
        else:
            children = sub_tree_information

        if is_child_with_underscore:
            child_field_name = "_children"
        else:
            child_field_name = "children"

        cluster_size = len(set(cluster_node_data.child_cluster_ids)) if \
            cluster_node_data.child_cluster_ids is not None else None

        return {
            "name": str(cluster_node_data.cluster_id),
            child_field_name: children,
            "cluster_size": cluster_size,
            "cluster-information": dict_cluster_id2cluster_agg_info[cluster_node_data.cluster_id],
            "clustering-method": cluster_node_data.clustering_label
        }

    def __generate_leaf_node_template(self, data_id: int, cluster_node_data: ClusterObject) -> Dict[str, Any]:
        """method to generate a dict object of leaf cluster node"""
        if isinstance(self.multi_matrix_object.dict_index2attributes, dict) and \
                data_id in self.multi_matrix_object.dict_index2attributes:
            information = self.multi_matrix_object.dict_index2attributes[data_id]
        else:
            information = {}

        return {
            "name": str(data_id),
            "label": self.multi_matrix_object.dict_index2label[data_id],
            "information": information,
            "clustering-method": cluster_node_data.clustering_label
        }

    def get_labels(self)->Dict[int, int]:
        """it gets a dict object of data-id: cluster-id"""
        return {
            data_id: c_id
            for c_id, c_obj in self.dict_cluster_id2cluster_obj.items()
            for data_id in c_obj.data_ids
        }

    def to_dict(self, is_child_with_underscore: bool=True)->Dict[str, Any]:
        """Show tree object with dict object. The dict object is suitable to collapsible tree in D3.js.

        :param is_child_with_underscore: True; close child cluster nodes in collapsible tree with default
        """
        self.dict_cluster_id2cluster_agg_info = self.aggregate_node_information()
        self.updated_dict_child2parent = self.reshape_tree()

        if len(self.updated_dict_child2parent) == 0:
            raise Exception(
                "There is 0 cluster for given dataset. This might be too small amount of data to construct clusters.")

        root_node_id = min(self.updated_dict_child2parent.values())

        # construct sub trees
        dict_subtree_pack = {
            parent_id: [t[0] for t in g_obj]
            for parent_id, g_obj
            in groupby(sorted(self.updated_dict_child2parent.items(), key=lambda t: t[1]), key=lambda t: t[1])}
        # dict to save subtree which is already completed of subtree construction
        dict_stack_constructed_subtree = {}  # type: Dict[int,List[Dict[str,Any]]]
        # start tree construction from bottom (starts from bigger node id)
        for subtree_parent_id, seq_subtree_node in sorted(dict_subtree_pack.items(), key=lambda t: t[0], reverse=True):
            stack_subtree = [None] * len(seq_subtree_node)
            # ===========================================================================================
            for i, sub_tree_cluster_id in enumerate(seq_subtree_node):
                cluster_information_data = self.dict_cluster_id2cluster_obj[sub_tree_cluster_id]
                if sub_tree_cluster_id in dict_stack_constructed_subtree:
                    generated_dict_object = self.__generate_branch_node_template(
                        cluster_node_data=cluster_information_data,
                        sub_tree_information=dict_stack_constructed_subtree[sub_tree_cluster_id],
                        dict_cluster_id2cluster_agg_info=self.dict_cluster_id2cluster_agg_info,
                        is_child_with_underscore=is_child_with_underscore)
                else:
                    # if sub_tree_cluster_id does not exist yet, that means sub_tree_cluster_id is leaf node
                    generated_dict_object = self.__generate_branch_node_template(
                        cluster_node_data=cluster_information_data,
                        dict_cluster_id2cluster_agg_info=self.dict_cluster_id2cluster_agg_info)
                stack_subtree[i] = generated_dict_object
            # ===========================================================================================
            dict_stack_constructed_subtree[subtree_parent_id] = stack_subtree

        child_field = "children"
        dict_d3_object = {"name": "Root", child_field: dict_stack_constructed_subtree[root_node_id], "cluster_size": 30}
        return dict_d3_object

    def to_html(self, visual_html: str=None):
        """generate html file with collapsible tree
        :param visual_html: html template which has collapsible tree in it.
        :return:
        """
        logger.debug("generating html document with template...")
        visual_template_html = pkgutil.get_data("flexible_clustering_tree",
                                                "resources/tree_template.html").decode("utf-8")
        jinja_environment = Environment(loader=BaseLoader)
        tpl = jinja_environment.from_string(visual_template_html)

        dict_d3_object = self.to_dict()
        dict_render_values = {
            "tree_object": json.dumps(dict_d3_object, ensure_ascii=False)
        }

        html = tpl.render(dict_render_values)
        return html

    def to_objects(self) -> Dict[str, Dict[str, Any]]:
        """Generates dict object which has 2 types of information. 'cluster_information' is relation of parent-child.
        'leaf_information' is object of leaf nodes.

        :return: {"cluster_information": [], "leaf_information": []}
        """
        dict_d3_object = self.to_dict()
        __ = {"cluster_information": {}, "leaf_information": {}}
        _cluster_relation_table_obj = {'cluster_id': None,
                                       'parent_id': None,
                                       'depth_level': None,
                                       'clustering_method': None}
        __cluster_relation_table_rows = []
        for child_cid, parent_cid in self.dict_child_id2parent_id.items():
            c_obj = self.dict_cluster_id2cluster_obj[child_cid]
            __relation_table_obj = deepcopy(_cluster_relation_table_obj)
            __relation_table_obj['cluster_id'] = child_cid
            __relation_table_obj['parent_id'] = parent_cid
            __relation_table_obj['depth_level'] = c_obj.matrix_depth_level
            __relation_table_obj['clustering_method'] = c_obj.clustering_label
            __cluster_relation_table_rows.append(__relation_table_obj)
        else:
            pass

        __node_table_rows = []
        _node_table_obj = {'leaf_id': None, 'cluster_id': None, 'label': None, 'args': None}
        d_id2c_id = {d_id: c_obj.cluster_id
                     for c_obj in self.dict_cluster_id2cluster_obj.values()
                     for d_id in c_obj.data_ids}
        for data_id, label in self.multi_matrix_object.dict_index2label.items():
            __leaf_object = None if self.multi_matrix_object.dict_index2attributes is None \
                else json.dumps(self.multi_matrix_object.dict_index2attributes[data_id], ensure_ascii=False)
            __node_table_obj = deepcopy(_node_table_obj)
            __node_table_obj['leaf_id'] = data_id
            __node_table_obj['cluster_id'] = d_id2c_id[data_id]
            __node_table_obj['label'] = self.multi_matrix_object.dict_index2label[data_id]
            __node_table_obj['args'] = __leaf_object
            __node_table_rows.append(__node_table_obj)
        else:
            pass

        __['cluster_information'] = __cluster_relation_table_rows
        __['leaf_information'] = __node_table_rows

        return __
