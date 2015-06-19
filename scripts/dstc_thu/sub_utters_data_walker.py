#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
sub_utters_data_walker
read sub utters data

'''

import argparse, sys, json
import re


class sub_utters_data_walker(object):
	def __init__(self, sub_utters_data_file):
		input = open(sub_utters_data_file, 'r')
		self.data = json.load(input)
		input.close()

	def ReadUtter(self):
		for i, cul_utter in enumerate(self.data['utterances']):
			if i == 0:
				pre_utter = None
			else: 
				pre_utter = self.data['utterances'][i-1]
			yield (pre_utter, cul_utter)

