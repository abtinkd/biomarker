# -*- coding: utf-8 -*-
"""
Created on Wed July 06 09:32:06 2016

@author: Chad Kunde
"""

from __future__ import print_function

from itertools import count
from string import Template
import numpy as np
from functools import partial

np.set_printoptions(precision=6, suppress=True)

class Tree:
    def __init__(self, stump_func, leaf_func=None, max_depth=2, classes=["Alive", "Deceased"], treeNum=None):
        self.stump_func = stump_func
        self.stump_f = stump_func
        self.leaf_func = leaf_func if leaf_func is not None else stump_func
        self.leaf_f = leaf_func
        self.max_depth = max_depth
        self.count = count()
        self.classes = classes
        self.root = None
        self.labels = []
        self.treeNum = treeNum

    def get_params(self, deep=None):
        return { "stump_func" : self.stump_func,
                 "leaf_func" : self.leaf_func,
                 "max_depth" : self.max_depth,
                 "classes"   : self.classes  }

    def fit(self, data, class_lbl, min_split=None, labels=None, weights=None, pool=None):
        if min_split is not None:
            self.stump_func = partial(self.stump_f, min_split=min_split)
            self.leaf_func = partial(self.leaf_f, min_split=min_split)
        if weights is None:
            self.root = TreeNode(self, next(self.count), None, self.stump_func, data, class_lbl, np.ones(len(data)))
        else:
            self.root = TreeNode(self, next(self.count), None, self.stump_func, data, class_lbl, weights) 
        self.labels = labels
        self.fit_node(self.root, pool=pool)

    def fit_node(self, node, depth=1, pool=None):
        if depth > self.max_depth:
            return
        if depth == self.max_depth:
            node.split(self.count, pool, split_func=self.leaf_func)
            return
        left, right = node.split(self.count, pool, split_func=self.stump_func)
        self.fit_node(left, depth+1, pool)
        self.fit_node(right, depth+1, pool)
        return

    def test(self, data, class_lbl=None, weights=None):
        countr = count()
        if weights is None:
            root = TestNode(self, next(countr), None, self.root, data, class_lbl, np.ones(len(data)))
        else:
            root = TestNode(self, next(countr), None, self.root, data, class_lbl, weights)
        self.test_node(root, countr)
        return root
    
    def test_node(self, test_node, countr):
        if test_node.node.left is None:
            return
        left, right = test_node.split(countr)
        self.test_node(left, countr)
        self.test_node(right, countr)
        return

    def score(self, data, class_lbl):
        node = self.test(data, class_lbl)
        return node.test_score()

    def conf_matrix(self, node=None, mat=None):
        if mat is None:
            mat = np.zeros((2,2))
            if node is None:
                return self.conf_matrix(self.root, mat)
            else:
                return self.conf_matrix(node, mat)
        if node.left is None:
            mat[:,node.node_lbl] += node.counts
            return mat
        return self.conf_matrix(node.right, self.conf_matrix(node.left, mat))

    def print_tree(self, node=None):
        if node is None:
            node = self.root
        ret = '{0}'.format(node.desc(self.labels, self.classes))
        if node.parent is not None and node.parent.parent is None:
            if node == node.parent.left:
                ret += '{0} -> {1} [labeldistance=2.5, labelangle=45, headlabel="True"];\n'.format(node.parent.node_id, node.node_id)
            else:
                ret += '{0} -> {1} [labeldistance=2.5, labelangle=-45, headlabel="False"];\n'.format(node.parent.node_id, node.node_id)
        elif node.parent is not None:
            ret += "{0} -> {1};\n".format(node.parent.node_id, node.node_id)
        if node.left:
            ret += self.print_tree(node.left)
        if node.right:
            ret += self.print_tree(node.right)
        return ret
    def __repr__(self):
        return Template('''digraph Tree {
node [shape=box, style="rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;
$tree
}''').substitute(tree=self.print_tree(self.root))


class TreeNode:
    def __init__(self, tree, node_id, parent, split_func, data, class_lbl, weights):
        print(tree.treeNum, data.shape, weights.shape, weights[weights==1].sum())
        self.node_id = node_id
        self.parent = parent
        self.data = data
        self.class_lbl = class_lbl
        self.left = None
        self.right = None
        self.tree = tree

        if split_func is not None:
            self.split_func = split_func
        else:
            self.split_func = parent.split_func

        if len(data) == 0:
            self.counts = [0,0]
            self.node_lbl = 0
            self.class_props = 0
            self.weights = np.ones(0)
            self.splitr = None
            self.val = None
        else:
            self.counts = np.bincount(np.array(class_lbl, dtype=np.int32))
            self.node_lbl = np.argmax(self.counts)
            self.class_props = self.counts[self.node_lbl] / float(self.counts.sum())
            self.splitr = split_func(data, class_lbl, weights)
            self.val=self.splitr.score()
            self.weights = np.ones(len(data))if weights is None else weights
            
    def split(self, countr, pool, split_func=None):
        if split_func is not None:
            self.split_func = split_func
            self.splitr = split_func(self.data, self.class_lbl, self.weights)
            self.val=self.splitr.score()
        self.splitr.fit(pool)
        self.left = TreeNode(self.tree, next(countr), self, self.split_func,
                                 *self.splitr.mask_true(self.data, self.class_lbl, np.copy(self.weights)))
        self.right = TreeNode(self.tree, next(countr), self, self.split_func,
                                  *self.splitr.mask_false(self.data, self.class_lbl, np.copy(self.weights)))
        return self.left, self.right
    def label(self):
        return self.node_lbl
    def test_score(self, data, class_lbl):
        return self.splitr.test_score(data, class_lbl)
    def score(self):
        return self.class_props
    def desc(self, labels, classes):
        if labels is not None:
            lbl = classes[self.node_lbl]
        else:
            lbl = self.node_lbl
        return "{0} [label=<{1} <br/> {2} ({3}) <br/> {4:.4f}>];\n".format(self.node_id,
                                                                         self.splitr.full_desc(labels),
                                                                         self.counts,
                                                                         sum(self.counts),
                                                                         self.class_props)

class TestNode:
    def __init__(self, tree, node_id, parent, node, data, class_lbl, weights):
        self.tree = tree
        self.node_id = node_id
        self.parent = parent
        self.node = node
        self.node_lbl = node.node_lbl
        self.splitr = node.splitr
        self.data = data
        self.class_lbl = class_lbl
        self.weights = np.ones(len(data)) if weights is None else weights
        self.val = node.splitr.test_score(data, class_lbl, self.weights)
        if len(data) == 0:
            self.counts = np.zeros(2)
            self.class_props = np.nan
        else:
            self.counts = np.bincount(np.array(class_lbl, dtype=np.int64))
            self.counts.resize(2)
            self.class_props = self.counts[self.node_lbl] / float(self.counts.sum())
        self.left = None
        self.right = None

    def split(self, countr, split_func=None):
        if split_func is not None:
            self.split_func = split_func
            self.splitr = split_func(self.data, self.class_lbl)
        self.left = TestNode(self.tree, next(countr), self, self.node.left,
                                 *self.splitr.mask_true(self.data, self.class_lbl, self.weights))
        self.right = TestNode(self.tree, next(countr), self, self.node.right,
                                  *self.splitr.mask_false(self.data, self.class_lbl, self.weights))
        return self.left, self.right

    def score(self):
        return self.class_props
    def test_score(self):
        try:
            score = np.nan_to_num(self.splitr.score())
            if self.left is None:
                return abs(score-self.val)
            return abs(score-self.val)+self.left.test_score()+self.right.test_score()
        except:
            return 0
    def desc(self, labels, classes):
        if labels is not None:
            lbl = classes[self.node_lbl]
        else:
            lbl = self.node_lbl
        return "{0} [label=<{1} <br/> {2} ({3}) <br/> {4:.4f}>];\n".format(
            self.node_id,
            self.splitr.full_desc(labels, self.val),            
            self.counts,
            self.counts.sum(),
            self.class_props)
        # score = self.splitr.score()
        # val = score-self.val if score is not None else self.val
        # return "{0} [label=<{1} <br/> Diff: {2:.6f} <br/> {3} ({4}) <br/> {5:.4f}>];\n".format(
        #     self.node_id,
        #     self.splitr.full_desc(labels, self.val),
        #     val,
        #     self.counts,
        #     self.counts.sum(),
        #     self.class_props)
    def __repr__(self):
        return Template('''digraph Tree {
node [shape=box, style="rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;
$tree
}''').substitute(tree=self.tree.print_tree(self))
