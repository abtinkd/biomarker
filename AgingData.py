# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:09:08 2016

@author: Chad Kunde
"""

from __future__ import print_function, division

import os, sys
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, cross_val_predict
from sklearn.externals.six import StringIO
import pydot_ng as pydot
import Splitters as spl
import functools
import time
from CustomTree import Tree as CustomTree
from datetime import datetime
#import cPickle as pickle
import pickle
import PatternFinder as ptfd

class DTtest:
    def __init__(self, args):
        pd.set_option('display.max_columns', 10)
        pd.set_option('precision', 3)
        np.set_printoptions(precision=3,suppress=True)

        self.raw = pd.read_csv(args.sorc, na_values=args.naval)
        
        # Drop all rows with missing values
        self.data = self.raw.dropna(subset=[args.prog_var, args.ev_state, args.ev_time])\
                .round(args.prec)\
                .copy()

        self.train, self.test = train_test_split(self.data, train_size=0.80, random_state=7)
        self.labels = list(self.data)

        return


    # not used!
    def MakeTree(self, crit, name, depth=None, leaf_samples=1, leaf_nodes=None):
        rf = tree.DecisionTreeClassifier(criterion=crit,
                                    max_depth=depth,
                                    min_samples_leaf=leaf_samples,
                                    max_leaf_nodes=leaf_nodes)

        rf.fit(self.train[:,2:-2], self.train[:,-2])

        print("Test:", name)
        print("5-way cross-val scores:", cross_val_score( rf, self.data.iloc[:, 2:-2], self.data.iloc[:,-2], cv = 5 ))
        print("Training score:", rf.score( self.train[:,2:-2], self.train[:,-2] ))
        print("Testing score:", rf.score( self.test[:,2:-2],  self.test[:,-2]  ))

        dot_data = StringIO()
        tree.export_graphviz(rf, out_file=dot_data,
                             feature_names=list(self.data)[2:-2],
                             class_names=["Alive", "Deceased"],
                             filled=True, rounded=True,
                             special_characters=True)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf(name+".pdf")
        graph.write_png(name+".png")

        return

def Summary(trNode, dpth, trNum, side):
     ret = pd.DataFrame.from_dict({ "Feature": [trNode.node.splitr.feature],
                                         "Value": [trNode.node.splitr.value],
                                         "Level": [dpth],
                                         "TrainLeft": [trNode.node.left.splitr.coef],
                                         "TrainRight": [trNode.node.right.splitr.coef],
                                         "TrainDiff": [trNode.node.left.splitr.coef - trNode.node.right.splitr.coef],
                                         "TestLeft": [trNode.left.val],
                                         "TestRight": [trNode.right.val],
                                         "TestDiff": [trNode.left.val - trNode.right.val],
                                         "TrainBase": [trNode.node.splitr.coef],
                                         "TestBase": [trNode.val],
                                         "TreeNum": [trNum],
                                         "Side": [side],
                                         })
     ret["Direction"] = ~((ret.TestDiff<0) ^ (ret.TrainDiff<0))
     ret["Difference"] = abs(ret.TestDiff/ret.TrainDiff)
     ret["LeftScale"] = abs(ret.TestLeft/ret.TrainLeft)
     ret["RightScale"] = abs(ret.TestRight/ret.TrainRight)
     return ret
    
def Testing():
    test = DTtest()
    data = test.data[list(test.data)[3:-6]]
    from Splitters import SplitCoef_statsmod as smod
    
    from CustomTree import TreeNode, Tree as CustomTree

    filler = np.zeros(1)
    
    tr = CustomTree(smod, smod, max_depth=2)
    tr.root = TreeNode(0, None, smod, filler, filler, filler)
    tr.root.splitr = smod(filler, filler, filler)
    tr.root.left = TreeNode(0,tr.root, smod, filler,filler,filler)
    tr.root.left.splitr = smod(filler, filler, filler)
    tr.root.right = TreeNode(0,tr.root, smod, filler,filler,filler)
    tr.root.right.splitr = smod(filler, filler, filler)
    return
    
    train, test = train_test_split(data, train_size=0.60, random_state=7)
    tr.fit(train[~np.isnan(train.dsst)], train.death[~np.isnan(train.dsst)])
    result = tr.test(test[~np.isnan(test.dsst)], test.death[~np.isnan(test.dsst)])

    summaries = pd.DataFrame()

    summaries = summaries.append(Summary(result,1))
    summaries = summaries.append(Summary(result.left,2))
    summaries = summaries.append(Summary(result.right,2))

    print(summaries)

def cross_val(idxs, args, data, pool, min_split):
    n, (train_idx, test_idx) = idxs

    split_func = functools.partial(spl.SplitCoef_statsmod, args= args)

    tr = CustomTree(split_func, split_func, max_depth=2, treeNum=n)
    tr.fit(data.iloc[train_idx], data[args.ev_state].iloc[train_idx], pool=pool, min_split=min_split)
    t = tr.test(data.iloc[test_idx], data[args.ev_state].iloc[test_idx])
    print(datetime.now().time())
    return t

def finish(res, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    summaries = pd.DataFrame()
    sub_vs_orig_DF = pd.DataFrame()
    sub_vs_sub_DF = pd.DataFrame()
    trees = list()
    for t in res:
        try:
            graph = pydot.graph_from_dot_data(repr(t.tree))
            graph.write_svg("{0}/CrossVal{1}_train.svg".format(dest, t.tree.treeNum))
            graph = pydot.graph_from_dot_data(repr(t))
            graph.write_svg("{0}/CrossVal{1}_test.svg".format(dest, t.tree.treeNum))
        except:
            print("Unexpected error:", t.tree.treeNum, sys.exc_info()[0])
        summaries = summaries.append(Summary(t, 1, t.tree.treeNum, "Root"))
        summaries = summaries.append(Summary(t.left, 2, t.tree.treeNum, "Left"))
        summaries = summaries.append(Summary(t.right, 2, t.tree.treeNum, "Right"))
        s_v_o, s_v_s = ptfd.patterns(t, 2)
        sub_vs_orig_DF = sub_vs_orig_DF.append(s_v_o)
        sub_vs_sub_DF = sub_vs_sub_DF.append(s_v_s)
        trees.append(t)
 
    print(summaries)
    summaries.to_csv("./{0}/summary.csv".format(dest))
    sub_vs_orig_DF.to_csv("./{0}/pattern_sub_orig.csv".format(dest))
    sub_vs_sub_DF.to_csv("./{0}/pattern_sub_sub.csv".format(dest))
    with open("{0}/trees.pkl".format(dest), 'wb') as f:
        pickle.dump(trees, f, pickle.HIGHEST_PROTOCOL)


def process(args):
    test = DTtest(args)
    labels = list(set(list(test.data))-set(args.filter))
    data = test.data[labels]
    print('Involved variables:')
    print(sorted(labels))

    if args.sub is not None:
        data = data[:args.sub]

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    ss = ShuffleSplit(n_splits=args.nsplits, test_size=args.test, random_state=args.seed)    
    build_tree = functools.partial(cross_val, args=args, data=data, min_split=args.min_split)


    if args.dask is not None:
        from distributed import Client
        client = Client(args.dask)
        print(client)
        build_tree = functools.partial(cross_val, args=args, data=data, pool=client, min_split=args.min_split)
    elif args.nprocs is not None:
        from multiprocessing import Pool
        pool = Pool(processes=args.nprocs)
        build_tree = functools.partial(cross_val, args=args, data=data, pool=pool, min_split=args.min_split)
    else:
        from multiprocessing import Pool
        pool = Pool(processes=1)
        build_tree = functools.partial(cross_val, args=args, data=data, pool=pool, min_split=args.min_split)
        print(hasattr(pool, "gather"))
        
    print(datetime.now().time())

    # cv_scores = [build_tree(idx) for idx in enumerate(ss.split(data))]
    ds = pickle.load(open('/home/abt/Downloads/ds.pkl', 'rb'))
    cv_scores = [build_tree(idx) for idx in ds]
    finish(cv_scores, args.dest)
    return
    
    
if __name__ == "__main__":
#    Testing()
#    sys.exit(0)

    import argparse
    parser = argparse.ArgumentParser(description='Build decision trees using Cox Proportional Hazard models.')
    parser.add_argument('--dest', help='Output folder prefix for trees and summary.', default='CrossVal')
    parser.add_argument('--sorc', help='Input dataset path. (Default = ./Biomarker_Data_Fern_8_29_16.csv)',\
                        default='./Biomarker_Data_Fern_8_29_16.csv')
    parser.add_argument('--ev_time', help='Name of the variable that shows the time to the event of interest. (Default= ttodeath)', default='ttodeath')
    parser.add_argument('--ev_state', help='Name of the variable that shows the state of the event of interest. (Default= death', default='death')
    parser.add_argument('--prog_var', help='Name of the prognostic variable. (Default= sbp)', default='sbp')
    parser.add_argument('--naval', help='NA values in the dataset. (Default= .m .e .a .t)',\
                        default= ['.m', '.e', '.a', '.t'], nargs = '+')    
    parser.add_argument('--prec', help='Rounding precision for all features. (Default=2)', default=2, type=int)
    parser.add_argument('--filter', help='Ignore these variables during the experiments. (Default= id study site stroke ttostroke mi ttomi hf ttohf)', \
                        default=['id', 'study', 'site', 'stroke', 'ttostroke', 'mi', 'ttomi', 'hf', 'ttohf'], nargs= '+')
    parser.add_argument('--dask', help='Dask scheduler (ip:port)', default=None)
    parser.add_argument('--nprocs', help='Parallel processes (for local only).', default=None, type=int)
    parser.add_argument('--sub', help='Subset of data to process.', default=None, type=int) 
    parser.add_argument('--nsplits', help='Number of random subsets.', default=2, type=int)
    parser.add_argument('--test', help='Portion of data in test set.', default=0.4, type=float)
    parser.add_argument('--seed', help='Random seed for splits.', default=7, type=int)
    parser.add_argument('--min_split', help='Minimum portion of data in a split. (Default = 0.25)', default=0.25, type=float)
    args = parser.parse_args()    
    
    args.dest = (args.dest + "_n" + str(args.nsplits) +
                     "_s" + str(args.seed) +
                     "_" + datetime.now().strftime('%Y%m%d_%H%M%Z'))
    print(args.dest)
    process(args)