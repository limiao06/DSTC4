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


class Tuple_Extractor(object):
    MY_ID = 'Tuple_Extractor'
    '''
    read a config file
    know which slot is enumerable and which is non-enumerable

    then it can extract tuple from Frame_Label
    '''
    def __init__(self, slot_config_file = None):
        '''
        slot_config_file tells while slot is enumerable and which is not
        '''
        self.config = GetConfig()
        self.appLogger = logging.getLogger(self.MY_ID)

        if not slot_config_file:
            self.appLogger.debug('Slot config file is not assigned, so use the default config file')
        slot_config_file = self.config.get(self.MY_ID,'slot_config_file')
        slot_config_file = os.path.join(os.path.dirname(__file__),'../config/', slot_config_file)
        self.appLogger.debug('Slot config file: %s' %(slot_config_file))

        input = codecs.open(slot_config_file, 'r', 'utf-8')
        self.slot_config = json.load(input)
        input.close()

    def enumerable(self, slot):
        if slot not in self.slot_config:
            self.appLogger.error('Error: Unknown slot: %s' %(slot))
            raise Exception('Error: Unknown slot: %s' %(slot))
        else:
            return self.slot_config[slot]

    def extract_tuple(self, frame_label):
        output_tuple = []
        for slot in frame_label:
            output_tuple.append('root:%s' %(slot))
            if self.enumerable(slot): 
                for value in frame_label[slot]:
                    output_tuple.append('%s:%s' %(slot, value))
        return list(set(output_tuple))

    def generate_frame(self, tuples, t_probs, mode = 'hr'):
        '''
        generate frame based on tuples
        there are two generate modes:
        high-precision mode: 'hp'
        high-recall mode: 'hr'
        '''
        if mode != 'hp' and mode != 'hr':
            self.appLogger.error('Error: Unknown generate mode: %s' %(mode))
            raise Exception('Error: Unknown generate mode: %s' %(mode))

        add_tuples = []
        for t in tuples:
            tokens = t.split(':')
            assert(len(tokens) == 2)
            add_tuples.append(tuple(tokens))

        probs = [p for p in t_probs]

        frame_label = {}

        while True:
            current_size = len(add_tuples)
            if current_size == 0:
                break
            remove_index = []
            for i, t in enumerate(add_tuples):
                if t[0] == 'root':
                    if t[1] not in frame_label:
                        frame_label[t[1]] = {'prob': probs[i], 'values':{}}
                    else:
                        if probs[i] > frame_label[t[1]]['prob']:
                            frame_label[t[1]]['prob'] = probs[i]
                    remove_index.append(i)
                else:
                    if t[0] in frame_label:
                        new_prob = 1 - (1-probs[i])*(1- frame_label[t[0]]['prob'])
                        if t[1] not in frame_label[t[0]]['values']:
                            frame_label[t[0]]['values'][t[1]] = new_prob
                        else:
                            if new_prob > frame_label[t[0]]['values'][t[1]]:
                                frame_label[t[0]]['values'][t[1]] = new_prob
                        remove_index.append(i)

            add_tuples = [t for i,t in enumerate(add_tuples) if i not in remove_index]
            probs = [p for i,p in enumerate(probs) if i not in remove_index]
            if len(add_tuples) == current_size:
                break
        if mode == 'hp':
            return frame_label
        else :
            for t, prob in zip(add_tuples, probs):
                if t[0] not in frame_label:
                    frame_label[t[0]] = {'prob': -1, 'values':{}}
                if t[1] not in frame_label[t[0]]['values']:
                    frame_label[t[0]]['values'][t[1]] = prob
                else:
                    if prob > frame_label[t[0]]['values'][t[1]]:
                        frame_label[t[0]]['values'][t[1]] = prob
            return frame_label

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
    print '%.3f %.3f %.3f' %(sample_precision_all, sample_recall_all, sample_fscore_all)

    print  '# label performance'
    print 'Macro-average'
    print '%.3f %.3f %.3f' %(Macro_label_results['precision'], Macro_label_results['recall'], Macro_label_results['f1'])
    print 'Micro-average'
    print '%.3f %.3f %.3f' %(micro_label_results['precision'], micro_label_results['recall'], micro_label_results['f1'])

    print '\nlabel details'
    sorted_label_results = sorted(label_results.items(), key= lambda x:x[1]['f1'], reverse=True)
    for k,v in sorted_label_results:
        print '%s: %d %d %d %.3f %.3f %.3f' %(k, v['out'], v['label'], v['right'], v['precision'], v['recall'], v['f1'])








