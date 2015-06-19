#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
preprocessor
'''

import sys
sys.path.append('../')
import nltk
import re
from GlobalConfig import GetConfig

class stemmer(object):
	MY_ID = 'STEMMER'
	def __init__(self):
		#self.config = GetConfig()
		#from nltk.stem.lancaster import LancasterStemmer as Stemmer
		from nltk.stem.porter import PorterStemmer as Stemmer
		self.st = Stemmer()

	def stem(self, string):
		return self.st.stem(string)

class tokenizer(object):
	MY_ID = 'TOKENIZER'
	def __init__(self,mode=None):
		self.config = GetConfig()
		if mode:
			self.mode = mode
		else:
			if self.config.has_option(self.MY_ID,'mode'):
				self.mode = self.config.get(self.MY_ID,'mode')
			else:
				self.mode = 'NLTK'
		if self.mode == 'STANFORD':
			from nltk.tokenize.stanford import StanfordTokenizer as Tokenizer
			self.tokenizer = Tokenizer()
		elif self.mode == 'NLTK':
			pass
		elif self.mode == 'MINE':
			self.spacePunct = re.compile(ur'[`~!@#\$%\^&\*\(\)\[\]{}_\+\-=\|\\:;\"\'<>,\?/]')
			self.removePunct = re.compile(ur'\.')
		else:
			raise Exception('Error: tokenizer, Unknown mode %s!' %(self.mode))

	def tokenize(self, sent):
		if sent.endswith('-') or sent.endswith('~'):
			sent += ' '
		sent = sent.replace('~ ', ' ~ ')
		sent = sent.replace('- ', ' - ')
		if self.mode == 'STANFORD':
			tokens = self.tokenizer.tokenize(sent.strip())
		elif self.mode == 'NLTK':
			tokens = nltk.word_tokenize(sent.strip())
		elif self.mode == 'MINE':
			new_sent = sent.strip()
			new_sent = self.spacePunct.sub(' ', new_sent)
			new_sent = self.removePunct.sub('', new_sent)
			tokens = new_sent.split()
		p_sent = ' '.join(tokens)
		p_sent = p_sent.replace('% ', '%')
		p_sent = p_sent.replace('``', '\"')
		p_sent = p_sent.replace('\'\'', '\"')
		p_tokens = p_sent.split(' ')
		return p_tokens

if __name__ =="__main__":
	preproc = tokenizer('MINE')
	sents = ["%uh because this is a East-West Line %um you will have to change %uh the line in City Hall.",
			"So you actually get to view %uh you know the- some parts %uh of the island %uh and you know if you feel like stopping you can always stop-",
			"You are quite %uh you know what's going on from your airport, you take the subway-"
			]
	for sent in sents:
		tokens = preproc.tokenize(sent)
		print ' '.join(tokens)

