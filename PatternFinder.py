# -*- coding: utf-8 -*-
"""

@author: Abtin Khodadadi
"""
import pickle
import copy
import pandas as pd
import numpy as np
import sys

def generate_sub_orig_pattern(trNode, path_info, path_num):
    sub_vs_orig_pattrn_dic = {"PathCode": [str(path_info['TreeNum'])+':'+str(path_num), str(path_info['TreeNum'])+':'+str(path_num+1)],
        "TreeNum":[path_info['TreeNum'], path_info['TreeNum']],
        "PathLength":[path_info['level'], path_info['level']],
        "MaxDepth":[path_info['max_length'], path_info['max_length']],
        "TrainBaseCoef":[path_info['TrainBaseCoef'], path_info['TrainBaseCoef']],
        "TestBaseCoef":[path_info['TestBaseCoef'], path_info['TestBaseCoef']],
        "TrainSubCoef":[trNode.node.left.splitr.coef, trNode.node.right.splitr.coef],
        "TestSubCoef":[trNode.left.val, trNode.right.val],
        "TrainCoefDiff":[path_info['TrainBaseCoef'] - trNode.node.left.splitr.coef, path_info['TrainBaseCoef'] - trNode.node.right.splitr.coef],
        "TestCoefDiff":[path_info['TestBaseCoef'] - trNode.left.val, path_info['TestBaseCoef'] - trNode.right.val],
        "TrainBasePopulation":[path_info['TrainBasePopulation'], path_info['TrainBasePopulation']],
        "TrainBaseAlivePopulation":[path_info['TrainBaseAlivePopulation'], path_info['TrainBaseAlivePopulation']],
        "TrainBaseDeadPopulation":[path_info['TrainBaseDeadPopulation'], path_info['TrainBaseDeadPopulation']],
        "TestBasePopulation":[path_info['TestBasePopulation'], path_info['TestBasePopulation']],
        "TestBaseAlivePopulation":[path_info['TestBaseAlivePopulation'], path_info['TestBaseAlivePopulation']],
        "TestBaseDeadPopulation":[path_info['TestBaseDeadPopulation'], path_info['TestBaseDeadPopulation']],
        "TrainSubPopulation":[len(trNode.node.left.splitr.data), len(trNode.node.right.splitr.data)],
        "TrainSubAlivePopulation":[np.sum(trNode.node.left.splitr.class_lbl == 0), np.sum(trNode.node.right.splitr.class_lbl == 0)],
        "TrainSubDeadPopulation":[np.sum(trNode.node.left.splitr.class_lbl == 1), np.sum(trNode.node.right.splitr.class_lbl == 1)],
        "TestSubPopulation":[len(trNode.left.data), len(trNode.right.data)],
        "TestSubAlivePopulation":[trNode.left.counts[0], trNode.right.counts[0]],
        "TestSubDeadPopulation":[trNode.left.counts[1], trNode.right.counts[1]]}
    
    pt = "("
    for f in range(len(path_info['Features'])):
        pt += path_info['Path'][f] + '@'+ path_info['Features'][f] + '&'
        sub_vs_orig_pattrn_dic['feature'+str(f+1)] = [str(path_info['Features'][f]), str(path_info['Features'][f])]
        sub_vs_orig_pattrn_dic['feature'+str(f+1)+'_value'] = [str(path_info['Thresholds'][f]), str(path_info['Thresholds'][f])]
        
    sub_vs_orig_pattrn_dic['Pattern'] = [pt  +'leq)', pt + 'grt)']
    ret = pd.DataFrame.from_dict(sub_vs_orig_pattrn_dic)
    return ret



def generate_sub_sub_pattern(trNode, path_info, path_num):
    sub_vs_sub_pattrn_dic = {"PathCode": [str(path_info['TreeNum'])+':'+str(path_num)],
        "TreeNum":[path_info['TreeNum']],
        "PathLength":[path_info['level']],
        "MaxDepth":[path_info['max_length']],
        "TrainLeftCoef":[trNode.node.left.splitr.coef],
        "TestLeftCoef":[trNode.left.val],
        "TrainRightCoef":[trNode.node.right.splitr.coef],
        "TestRightCoef":[trNode.right.val],
        "TrainCoefDiff":[trNode.node.left.splitr.coef-trNode.node.right.splitr.coef],
        "TestCoefDiff":[trNode.left.val-trNode.right.val],
        "TrainBasePopulation":[path_info['TrainBasePopulation']],
        "TrainBaseAlivePopulation":[path_info['TrainBaseAlivePopulation']],
        "TrainBaseDeadPopulation":[path_info['TrainBaseDeadPopulation']],
        "TestBasePopulation":[path_info['TestBasePopulation']],
        "TestBaseAlivePopulation":[path_info['TestBaseAlivePopulation']],
        "TestBaseDeadPopulation":[path_info['TestBaseDeadPopulation']],
        "TrainLeftPopulation":[len(trNode.node.left.splitr.data)],
        "TrainLeftAlivePopulation":[np.sum(trNode.node.left.splitr.class_lbl == 0)],
        "TrainLeftDeadPopulation":[np.sum(trNode.node.left.splitr.class_lbl == 1)],
        "TrainRightPopulation":[len(trNode.node.right.splitr.data)],
        "TrainRightAlivePopulation":[np.sum(trNode.node.right.splitr.class_lbl == 0)],
        "TrainRightDeadPopulation":[np.sum(trNode.node.right.splitr.class_lbl == 1)],
        "TestLeftPopulation":[len(trNode.left.data)],
        "TestLeftAlivePopulation":[trNode.left.counts[0]],
        "TestLeftDeadPopulation":[trNode.left.counts[1]],
        "TestRightPopulation":[len(trNode.right.data)],
        "TestRightAlivePopulation":[trNode.right.counts[0]],
        "TestRightDeadPopulation":[trNode.right.counts[1]]}
    
    path_list = []
    for f in range(len(path_info['Features'])):
        path_list.append(path_info['Path'][f] + '@'+ path_info['Features'][f])
        sub_vs_sub_pattrn_dic['feature'+str(f+1)] = [str(path_info['Features'][f])]
        sub_vs_sub_pattrn_dic['feature'+str(f+1)+'_value'] = [str(path_info['Thresholds'][f])]
        
    ix = sorted(range(len(path_list)), key=lambda k: path_list[k], reverse = True)
    pt = "{"
    for i in ix:
        pt += path_list[i] + '&'
    pt += "}"
    sub_vs_sub_pattrn_dic['Pattern'] = pt
    ret = pd.DataFrame.from_dict(sub_vs_sub_pattrn_dic)
    return ret


def patterns(trNode, max_path_length, path_info = None, level = 0, sub_vs_orig_DF = None, sub_vs_sub_DF = None,  save_sub_patt = False, sub_vs_orig = True, sub_vs_sub = True):
    if max_path_length > trNode.tree.max_depth:
        return
    if level == 0:
        patterns.path_num = 0
        sub_vs_orig_DF = pd.DataFrame()
        sub_vs_sub_DF = pd.DataFrame()
        path_info = {'Features':[],
            'level': level,
            'max_length': max_path_length,
            'Thresholds': [],
            'Path': ['root'],
            'TrainBaseCoef': trNode.node.splitr.coef,
            'TestBaseCoef':trNode.val,
            'TrainBasePopulation': len(trNode.node.splitr.data),
            'TrainBaseAlivePopulation': np.sum(trNode.node.splitr.class_lbl == 0),
            'TrainBaseDeadPopulation': np.sum(trNode.node.splitr.class_lbl == 1),
            'TestBasePopulation': len(trNode.data),
            'TestBaseAlivePopulation': trNode.counts[0],
            'TestBaseDeadPopulation': trNode.counts[1],
            'TreeNum':trNode.tree.treeNum}
    
    level += 1
    path_info['Features'].append(trNode.node.splitr.feature)
    path_info['level'] = level
    path_info['Thresholds'].append(trNode.node.splitr.value)
    
    if level == max_path_length:
        s_v_o = sub_vs_orig_DF.append(generate_sub_orig_pattern(trNode, copy.deepcopy(path_info), patterns.path_num))
        s_v_s = sub_vs_sub_DF.append(generate_sub_sub_pattern(trNode, copy.deepcopy(path_info), patterns.path_num))
        patterns.path_num += 2
        return s_v_o, s_v_s

    left_path_info = copy.deepcopy(path_info)
    left_path_info['Path'].append('leq')
    right_path_info = copy.deepcopy(path_info)
    right_path_info['Path'].append('grt')
    l_s_v_o, l_s_v_s = patterns(trNode.left, max_path_length, left_path_info, level, sub_vs_orig_DF, sub_vs_sub_DF)
    r_s_v_o, r_s_v_s = patterns(trNode.right, max_path_length, right_path_info, level, sub_vs_orig_DF, sub_vs_sub_DF)

    sub_vs_orig_DF = sub_vs_orig_DF.append(l_s_v_o)
    sub_vs_orig_DF = sub_vs_orig_DF.append(r_s_v_o)
    sub_vs_sub_DF = sub_vs_sub_DF.append(l_s_v_s)
    sub_vs_sub_DF = sub_vs_sub_DF.append(r_s_v_s)
    return sub_vs_orig_DF, sub_vs_sub_DF
            

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract patterns from trees.')
    parser.add_argument('--sorc', help='Input file of the trees. (Default= ./trees.pkl)', default='./trees.pkl')
    parser.add_argument('--dest', help='Output folder to save the patterns. (Default= ./Patterns)', default='Patterns')
    args = parser.parse_args()
    
    file_name = args.sorc
    print('Reading from: '+ file_name)
    trees = pickle.load(open(file_name, "rb"), encoding='latin1')
    print ('Number of trees: {}'.format(len(trees)))

    sub_vs_orig_DF = pd.DataFrame()
    sub_vs_sub_DF = pd.DataFrame()    
    for t in trees:
        s_v_o, s_v_s = patterns(t, 2)
        sub_vs_orig_DF = sub_vs_orig_DF.append(s_v_o)
        sub_vs_sub_DF = sub_vs_sub_DF.append(s_v_s)

    #exculde results with different direction between train and test
    # sub_vs_orig_DF = sub_vs_orig_DF[np.sign(sub_vs_orig_DF['TrainCoefDiff']) == np.sign(sub_vs_orig_DF['TestCoefDiff'])]
    # sub_vs_sub_DF = sub_vs_sub_DF[np.sign(sub_vs_sub_DF['TrainCoefDiff']) == np.sign(sub_vs_sub_DF['TestCoefDiff'])]

    # add a column for showing the sign direction between train and test
    sub_vs_orig_DF['TrainTestSameDirection'] = pd.Series(
        np.sign(sub_vs_orig_DF['TrainCoefDiff']) == np.sign(sub_vs_orig_DF['TestCoefDiff']), index = sub_vs_orig_DF.index)
    sub_vs_sub_DF['TrainTestSameDirection'] = pd.Series(
        np.sign(sub_vs_sub_DF['TrainCoefDiff']) == np.sign(sub_vs_sub_DF['TestCoefDiff']), index = sub_vs_sub_DF.index)

    # population
    sub_vs_orig_DF['TrainSubPopFraction'] = pd.Series(
        sub_vs_orig_DF['TrainSubPopulation']/sub_vs_orig_DF['TrainBasePopulation'], index = sub_vs_orig_DF.index)
    sub_vs_orig_DF['TestSubPopFraction'] = pd.Series(
        sub_vs_orig_DF['TestSubPopulation']/sub_vs_orig_DF['TestBasePopulation'], index = sub_vs_orig_DF.index)

    sub_vs_orig_DF['TrainAliveSubPopFraction'] = pd.Series(
        sub_vs_orig_DF['TrainSubAlivePopulation']/sub_vs_orig_DF['TrainBaseAlivePopulation'], index = sub_vs_orig_DF.index)
    sub_vs_orig_DF['TestAliveSubPopFraction'] = pd.Series(
        sub_vs_orig_DF['TestSubAlivePopulation']/sub_vs_orig_DF['TestBaseAlivePopulation'], index = sub_vs_orig_DF.index)
    
    sub_vs_sub_DF['TrainSubPopFraction'] = pd.Series(
        (sub_vs_sub_DF['TrainLeftPopulation']+sub_vs_sub_DF['TrainRightPopulation'])/sub_vs_sub_DF['TrainBasePopulation'], index = sub_vs_sub_DF.index)
    sub_vs_sub_DF['TestSubPopFraction'] = pd.Series(
        (sub_vs_sub_DF['TestLeftPopulation']+sub_vs_sub_DF['TestRightPopulation'])/sub_vs_sub_DF['TestBasePopulation'], index = sub_vs_sub_DF.index)

    sub_vs_sub_DF['TrainAliveSubPopFraction'] = pd.Series(
        (sub_vs_sub_DF['TrainLeftAlivePopulation']+sub_vs_sub_DF['TrainRightAlivePopulation'])/sub_vs_sub_DF['TrainBaseAlivePopulation'], index = sub_vs_sub_DF.index)
    sub_vs_sub_DF['TestAliveSubPopFraction'] = pd.Series(
        (sub_vs_sub_DF['TestLeftAlivePopulation']+sub_vs_sub_DF['TestRightAlivePopulation'])/sub_vs_sub_DF['TestBaseAlivePopulation'], index = sub_vs_sub_DF.index)

    sub_vs_orig_DF.to_csv("{0}/pkl_pattern_sub_orig.csv".format(args.dest))
    sub_vs_sub_DF.to_csv("{0}/pkl_pattern_sub_sub.csv".format(args.dest))
