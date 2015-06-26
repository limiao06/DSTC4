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


def EvalMultiLabel(labels, output_labels):
    '''
    labels and output labels are lists of multi-label
    each item is a list of label
    '''
    label_dict = {}
    for label, output_label in zip(labels, output_labels):
        for l in label:
            if l not in label_dict:
                label_dict[l] = 1
        for l in output_label:
            if l not in label_dict:
                label_dict[l] = 1

    sample_results = []
    label_results = {}
    for l in label_dict:
        label_results[l] = {'label':0, 'out':0, 'right':0, 'precision':0.0, 'recall':0.0, 'f1':0.0}

    micro_label_results = {'label':0, 'out':0, 'right':0, 'precision':0.0, 'recall':0.0, 'f1':0.0}

    for label, output_label in zip(labels, output_labels):
        label_count = len(label)
        output_count = len(output_label)
        right_count = 0
        for l in label:
            if l in output_label:
                right_count += 1
        if label_count == 0 or output_count == 0 or right_count == 0:
            s_precision = 0.0
            s_recall = 0.0
            s_fscore = 0.0
        else:
            s_precision = right_count * 1.0 / output_count
            s_recall = right_count * 1.0 / label_count
            s_fscore =  2 * s_precision * s_recall / (s_precision + s_recall)

        micro_label_results['label'] += label_count
        micro_label_results['out'] += output_count
        micro_label_results['right'] += right_count

        sample_results.append((s_precision, s_recall, s_fscore))
        for l in label:
            label_results[l]['label'] += 1
            if l in output_label:
                label_results[l]['right'] += 1

        for l in output_label:
            label_results[l]['out'] += 1

    for l in label_results:
        if label_results[l]['right'] == 0 or label_results[l]['right'] == 0 or label_results[l]['right'] == 0:
            pass
        else:
            label_results[l]['precision'] = label_results[l]['right'] * 1.0 / label_results[l]['out']
            label_results[l]['recall'] = label_results[l]['right'] * 1.0 / label_results[l]['label']
            label_results[l]['f1'] = 2 * label_results[l]['precision'] * label_results[l]['recall'] / (label_results[l]['precision'] + label_results[l]['recall'])

    micro_label_results['precision'] = micro_label_results['right'] * 1.0 / micro_label_results['out']
    micro_label_results['recall'] = micro_label_results['right'] * 1.0 / micro_label_results['label']
    micro_label_results['f1'] = 2 * micro_label_results['precision'] * micro_label_results['recall'] / (micro_label_results['precision'] + micro_label_results['recall'])

    Macro_label_results = {}
    Macro_label_results['precision'] = sum([v['precision'] for k,v in label_results.items()]) / len(label_results)
    Macro_label_results['recall'] = sum([v['recall'] for k,v in label_results.items()]) / len(label_results)
    Macro_label_results['f1'] = sum([v['f1'] for k,v in label_results.items()]) / len(label_results)
 
    sample_precision_all = sum([a[0] for a in sample_results]) / len(sample_results)
    sample_recall_all = sum([a[1] for a in sample_results]) / len(sample_results)
    sample_fscore_all = sum([a[2] for a in sample_results]) / len(sample_results)

    print '# sample performance'
    print '%.3f %.3f %.3f' %(seg_precison_all, seg_recall_all, seg_fscore_all)

    print  '# label performance'
    print 'Macro-average'
    print '%.3f %.3f %.3f' %(Macro_label_results['precision'], Macro_label_results['recall'], Macro_label_results['f1'])
    print 'Micro-average'
    print '%.3f %.3f %.3f' %(micro_label_results['precision'], micro_label_results['recall'], micro_label_results['f1'])

    print '\nlabel details'
    sorted_label_results = sorted(label_results.items(), key= lambda x:x[1]['f1'], reverse=True)
    for k,v in sorted_label_results:
        print '%s: %d %d %d %.3f %.3f %.3f' %(k, v['out'], v['label'], v['right'], v['precision'], v['recall'], v['f1'])








