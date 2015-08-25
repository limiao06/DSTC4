#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
given a slot,
find the most likely value in a utter
'''

import argparse, sys, time, json, os, copy
from collections import defaultdict
from fuzzywuzzy import fuzz

class value_extractor(object):
	def __init__(self, tagsets, threshold = 0, max_num = 2, case_sensitive=False):
		#'''
		self.tagsets = {}
		for topic in tagsets:
			self.tagsets[topic] = {}
			for slot in tagsets[topic]:
				self.tagsets[topic][slot] = {}
				if slot == 'CUISINE':
					for value in tagsets[topic][slot]:
						match_value = value.replace(' cuisine', '')
						if case_sensitive:
							self.tagsets[topic][slot][match_value] = value
						else:
							self.tagsets[topic][slot][match_value.lower()] = value
				elif slot == 'STATION':
					for value in tagsets[topic][slot]:
						match_value = value.replace(' Station', '')
						if case_sensitive:
							self.tagsets[topic][slot][match_value] = value
						else:
							self.tagsets[topic][slot][match_value.lower()] = value
				else:
					for value in tagsets[topic][slot]:
						match_value = value
						if case_sensitive:
							self.tagsets[topic][slot][match_value] = value
						else:
							self.tagsets[topic][slot][match_value.lower()] = value
		#'''
		#self.tagsets = tagsets
		self.threshold = threshold * 100
		self.max_num = max_num

	def ExtractValue(self, topic, slot, transcript):
		value_dict = self._ExtractValueDict(topic, slot, transcript)
		if len(value_dict) == 0:
			return []
		sort_value_dict = sorted(value_dict.items(), key=lambda x:x[1], reverse=True)
		if self.threshold == 0:
			return [sort_value_dict[0]]
		else:
			value_list = []
			for value, ratio in sort_value_dict:
				if ratio >= self.threshold:
					value_list.append((value, ratio))
					if self.max_num > 0 and len(value_list) >= self.max_num:
						break
				else:
					break
			return value_list

	def _ExtractValueDict(self, topic, slot, transcript):
		value_dict = defaultdict(float)
		if slot == 'CUISINE' or slot == 'DRINK':
			pass
		else:
			transcript = transcript.replace('Singapore', '')
		if topic in self.tagsets:
			if slot in self.tagsets[topic]:
				for match_value, value in self.tagsets[topic][slot].items():
					if case_sensitive:
						ratio = fuzz.partial_ratio(match_value, transcript)
					else:
						ratio = fuzz.partial_ratio(match_value, transcript.lower())
					if ratio > value_dict[value]:
						value_dict[value] = ratio
		return value_dict


