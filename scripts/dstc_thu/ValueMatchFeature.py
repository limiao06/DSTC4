import argparse, sys, time, json
from fuzzywuzzy import fuzz
import copy

'''
value match feature 
use fuzz.partial_ratio to match input utterance with values in ontology
return the match ratio that above than the low_thres
'''

class ValueMatchFeature(object):
	def __init__(self, tagsets, low_thres=0.3, case_sensitive=False):
		self.low_thres = low_thres
		self.topic_value_id_map = {}
		self.case_sensitive=case_sensitive
		self.value_num=0
		for topic in tagsets:
			self.topic_value_id_map[topic] = {}
			for slot in tagsets[topic]:
				for value in tagsets[topic][slot]:
					if not self.case_sensitive:
						value = value.lower()
					if value not in self.topic_value_id_map[topic]:
						self.topic_value_id_map[topic][value] = self.value_num+1
						self.value_num += 1

	def GetFeatureSize(self):
		return self.value_num


	def extract_trans_feature(self, trans, topic):
		transcript = trans.replace('Singapore', '')
		feature_vec={}
		if topic in self.topic_value_id_map:
			for value,idx in self.topic_value_id_map[topic].items():
				if not self.case_sensitive:
					transcript = transcript.lower()
				ratio = fuzz.partial_ratio(value, transcript) * 1.0 / 100
				if ratio > self.low_thres:
					feature_vec[idx] = ratio
		return feature_vec

	def Merge2Features(self, f1, f2):
		feature_vec = copy.deepcopy(f1)
		for idx,value in f2.items():
			if idx in feature_vec and value > feature_vec[idx]:
				feature_vec[idx] = value
			elif idx not in feature_vec:
				feature_vec[idx] = value
		return feature_vec

	def Save(self):
		'''
		output the variables into a json format
		'''
		outjson = {'name':'ValueMatchFeature'}
		outjson['low_thres'] = self.low_thres
		outjson['case_sensitive'] = self.case_sensitive
		outjson['value_num'] = self.value_num
		outjson['topic_value_id_map'] = self.topic_value_id_map
		return outjson

	def Load(self, in_json):
		if in_json['name'] != 'ValueMatchFeature':
			raise Exception('Input Json is not a proper format for load!')
		self.low_thres = in_json['low_thres']
		self.case_sensitive = in_json['case_sensitive']
		self.value_num = in_json['value_num']
		self.topic_value_id_map = in_json['topic_value_id_map']



