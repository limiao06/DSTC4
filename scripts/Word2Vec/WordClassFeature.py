#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
word class feature 
read the output of WordVecClustering.py

Miao Li
limiaogg@126.com
'''


class Word2VecClass(object):
	def __init__(self, WordClassFile):
		self.dict = {}
		input = file(WordClassFile)
		while True:
			line = input.readline()
			if not line:
				break
			l = line.strip()
			tokens = l.split('\t')
			if len(tokens) != 2:
				continue
			self.dict[tokens[0]] = int(tokens[1])

	def GetWordClass(self, word):
		if word not in self.dict:
			return -1
		else:
			return self.dict[word]

