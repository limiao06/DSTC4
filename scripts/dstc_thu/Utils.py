#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
Utility functions for dstc4


Miao Li
limiaogg@126.com
'''


import logging, sys
import json
import os
import re
from math import log


def GetLogLevel(log_level_key):
    key = log_level_key.upper()
    if key == 'CRITICAL':
        return logging.CRITICAL
    elif key == 'ERROR':
        return logging.ERROR
    elif key == 'WARNING':
        return logging.WARNING
    elif key == 'INFO':
        return logging.INFO
    elif key == 'DEBUG':
        return logging.DEBUG
    elif key == 'NOTSET':
        return logging.NOTSET
    else:
        raise Exception('Unknown log level key!')



# feature chosen methods
def Stat_CHI_MultiLabel(feature_keys, labels):
    '''
    feature_keys is a list, each item is the features of a sample
    labels is a list, each item is a list of multi-labels of a sample
    '''
    if not len(feature_keys) == len(labels):
        raise Exception('Error: Stat_CHI_MultiLabel: feature_keys and labels must have same size! %d != %d' %(len(feature_keys), len(labels)))
    # stat labels
    label_dict = {}
    for label in labels:
        for lb in label:
            if lb not in label_dict:
                label_dict[lb] = 1

    # stat features
    feature_dict = {}
    for feature in feature_keys:
        for f in feature:
            if f not in feature_dict:
                feature_dict[f] = 1

    N = len(feature_keys)

    A = {}
    B = {}
    C = {}
    D = {}

    CHI_f_label = {}
    CHI_f = {}

    for f in feature_dict.keys():
        CHI_f[f] = 0.0
        for label in label_dict.keys():
            A[(f,label)] = 0
            B[(f,label)] = 0
            C[(f,label)] = 0
            D[(f,label)] = 0
            CHI_f_label[(f,label)] = 0.0

    for feature, label in zip(feature_keys, labels):
        feature_set = set(feature)
        label_set = set(label)

        unseen_feature_set = set(feature_dict.keys()) - feature_set
        unseen_label_set = set(label_dict.keys()) - label_set

        for f in list(feature_set):
            for label in list(label_set):
                A[(f,label)] += 1

        for f in list(feature_set):
            for label in list(unseen_label_set):
                B[(f,label)] += 1

        for f in list(unseen_feature_set):
            for label in list(label_set):
                C[(f,label)] += 1

        for f in list(unseen_feature_set):
            for label in list(unseen_label_set):
                D[(f,label)] += 1

    for key in CHI_f_label:
        CHI_f_label[key] = N * (A[key] * D[key] - C[key] * B[key])**2 / ((A[key] + C[key]) * (B[key] + D[key]) * (A[key] + B[key]) * (C[key] + D[key]))

    for f in feature_dict.keys():
        CHI_f[f] = max([CHI_f_label[(f,label)] for label in label_dict.keys()])

    return CHI_f

def ChooseFromCHI(CHI, percent=0.5):
    new_f_vec = []
    chose_num = int(len(CHI) * percent)
    sorted_chi = sorted(CHI.items(), key = lambda x:x[1], reverse = True)
    for i, (key,value) in enumerate(sorted_chi):
        if i >= chose_num:
            break
        new_f_vec.append(key)
    return new_f_vec




