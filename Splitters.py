# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:46:51 2016

@author: Chad Kunde
"""
from __future__ import print_function, division

from functools import partial
from statsmodels.duration.hazard_regression import PHReg
import numpy as np
import pandas as pd


def testCoef(args, node):
    (i, v, mask, nanmsk) = args
    Coef1 = node.cox_coef(node.data[mask & nanmsk],
                              node.class_lbl[mask & nanmsk],
                              node.weights[np.array(mask & nanmsk, dtype=bool)])
    Coef2 = node.cox_coef(node.data[(~mask) & nanmsk],
                              node.class_lbl[(~mask) & nanmsk],
                              node.weights[np.array((~mask) & nanmsk, dtype=bool)])
    Diff = abs(Coef1-Coef2)
    return [i, v, Diff, Coef1, Coef2, mask.sum(), (~nanmsk).sum()]

class SplitCoef_statsmod(object):
    def __init__(self, data, class_lbl, weights, scale=0.5, min_split=0.25):
        """
        data      : features
        class_lbl : classifier
        weights   : per-case weight
        scale     : weight scaling factor for missing data
        data, class_lbl, and weights must be the same length.
        """
        self.feature = None
        self.value = None
        self.idx = None
        self.labels = list(data)
        self.coef = np.nan
        self.countr = 0
        self.nancount = np.nan
        self.ratio = scale
        if float(np.sum(class_lbl))/float(np.sum(1-class_lbl)) in[0,1]:
            self.data = None
            self.coef = None
            self.loglikelihood = None
            self.weights = None
            return
        if len(weights) != len(data):
            print("Weight length mismatch:", len(weights), len(data))
        self.weights = weights
        self.min_split = min_split if min_split < 0.5 else 1 - min_split
        self.data = data
        self.class_lbl = class_lbl
        self.coef, self.loglikelihood = self.cox_coef(data, class_lbl, weights, llf=True)

    def fit(self, pool):
        if self.data is None:
            return
        tests = list() 
        
        for i in range(self.data.shape[1]-2): 
            unq_val = np.unique(self.data[self.labels[i]])
            lbl = self.labels[i]
            if lbl in ["sbp", "death", "ttodeath"]  or len(unq_val[~np.isnan(unq_val)]) == 1:
                continue
            for v in (unq_val[~np.isnan(unq_val)])[:-1]:
                # Fitting cox models
                # Select on largest magnitude difference in sbp coefficients
                mask = (self.data[lbl] <= v)
                nanmsk = ~np.isnan(self.data[lbl])
                if not (self.min_split < mask.sum()/nanmsk.sum() < 1-self.min_split and
                    self.min_split < mask.sum()/self.data.shape[0] < 1-self.min_split and
                    mask.sum() > (~nanmsk).sum() < (self.data.shape[0] - mask.sum())):
                    continue
                tests.append(( i, v, mask, nanmsk ))
                
        self.countr = len(tests)
        testMasks = partial(testCoef, node=self)
        rslt = pool.map(testMasks, tests)
        if hasattr(pool, "gather"):
            rslt = pool.gather(rslt)
        # rslt = [testMasks(a) for a in tests]
        rslt = np.array(rslt)
        self.idx = rslt[rslt.argmax(axis=0)[2]]
        self.nancount = self.idx[6]
        print("Curves Fit:", self.countr, " Nan:", self.nancount)
        if self.idx is None:
            return
        self.feature = self.labels[int(self.idx[0])]
        self.value = self.idx[1].astype(self.data[self.feature].dtype)
        self.ratio = self.idx[5]/(self.data.shape[0]-self.nancount)
        
    def cox_coef_jit(self, data, class_lbl, weights):
        mod = PHReg(data.ttodeath, data.sbp, status=class_lbl)
        return mod.fit_regularized(alpha=weights, warn_convergence=False)

    def cox_coef(self, data, class_lbl, weights, llf=False):
        mod = self.cox_coef_jit(data, class_lbl, weights)
        if llf:
            return (mod.summary().tables[1]['log HR'][0], mod.llf)
        return mod.summary().tables[1]['log HR'][0]

    def splitr(self, dat):
        return (dat[self.feature]<=self.value)

    def split_vals(self):
        if self.idx is None:
            return None
        return self.idx[-2:]

    def mask_true(self, data, class_lbl, weights=None):
        if self.splitr is None:
            return data, class_lbl
        mask = self.splitr(data) | pd.isnull(data[self.feature])
        if weights is not None:
            nanLst = np.array(np.isnan(data[self.feature]), dtype=bool)
            weights[nanLst] = weights[nanLst] * self.ratio
            return data[mask], class_lbl[mask], weights[np.array(mask, dtype=bool)]
        return data[mask], class_lbl[mask]

    def mask_false(self, data, class_lbl, weights=None):
        if self.splitr is None:
            return data, class_lbl
        mask = (~self.splitr(data)) | pd.isnull(data[self.feature])
        if weights is not None:
            nanLst = np.array(np.isnan(data[self.feature]), dtype=bool)
            weights[nanLst] = weights[nanLst] * (1-self.ratio)
            return data[mask], class_lbl[mask], weights[np.array(mask, dtype=bool)]
        return data[mask], class_lbl[mask]

    def test_score(self, data, class_lbl, weights):
        try:
            return self.cox_coef(data, class_lbl, weights)
        except:
            return np.nan
    def score(self):
        return self.coef
    def desc(self, val=None):
        print(self.coef)
        if val is None:
            return "Coefficient={0}".format(self.coef)
        return "Coefficient={0}".format(val)
    def desc_ft(self, labels):
        print(self.feature, self.value)
        if labels:
            return "{0} &le; {1}".format(labels[int(self.feature)], self.value)
        return "{0} &le; {1}".format(self.feature, self.value)
    def full_desc(self, labels=None, val=None):
        if self.feature is not None:
            return "{0} <br/> {1}".format(self.desc(val), self.desc_ft(labels))
        return "{0}".format(self.desc(val))

