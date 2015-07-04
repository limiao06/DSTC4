#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
Semantic_Tag_Data_Extractor.py

read sub_utters data
convert to crf like format
'''

import argparse, sys, time, json, os, codecs, logging
import re
from collections import defaultdict

import nltk
from preprocessor import *
from GlobalConfig import *
from Utils import *



from sub_utters_data_walker import sub_utters_data_walker

class SemTagExtractor(object):
	def __init__(self):
		self.tokenizer = tokenizer()
		self.stemmer = stemmer()
		self.reset()

	def ProcSent(self, sent):
		self.reset()
		self.sent = sent
		self._process()

	def reset(self):
		self.sent = None
		self.ori_TokenList = []
		self.ori_TagList = []
		self.ori_BIOList = []

		self.tok_TokenList = []
		self.tok_LemmaList = []
		self.tok_POSList = []
		self.tok_TagList = []
		self.tok_BIOList = []

		self.tokenized_tokens_map = []
		self.success = False



	def process_utters(self,utter):
		sub_utters_list = []
		for sub_utter_tag in utter['sub_tag_list']:
			sub_utters_list.append(self._process_sub_utter_tag(sub_utter_tag))
		return sub_utters_list

	def _process_sub_utter_tag(self, sub_utter_tag):
		pass

	def _process(self):
		# extract semantic tags:
		(self.ori_TokenList, self.ori_TagList, self.ori_BIOList) = SemTagExtractor._ReadSentTags(self.sent)
		if self.ori_TokenList == None:
			return

		# build token map
		self.tok_TokenList = self._tokenize(' '.join(self.ori_TokenList))
		word_tags = nltk.pos_tag(self.tok_TokenList)
		self.tok_LemmaList = []
		self.tok_POSList = []
		for word, tag in word_tags:
			self.tok_LemmaList.append(self.stemmer.stem(word.lower()))
			self.tok_POSList.append(tag)

		self.tokenized_tokens_map = SemTagExtractor._map_tokens(self.ori_TokenList, self.tok_TokenList)
		if self.tokenized_tokens_map == None:
			return


		# map tags
		pre_map_id = -1
		for map_id in self.tokenized_tokens_map:
			self.tok_TagList.append(self.ori_TagList[map_id])
			if self.ori_BIOList[map_id][-1] == 'B':
				BIO_tag = self.ori_BIOList[map_id][:-1]
				if map_id != pre_map_id:
					self.tok_BIOList.append(BIO_tag + 'B')
				else:
					self.tok_BIOList.append(BIO_tag + 'I')
			else:
				self.tok_BIOList.append(self.ori_BIOList[map_id])
		'''
		print self.ori_TokenList
		print self.ori_TagList
		print self.ori_BIOList

		print self.tok_TokenList
		print self.tok_TagList
		print self.tok_BIOList
		'''
		self.success = True
		return

	@staticmethod
	def _map_tokens(ori_token, tok_token):
		tokens_map = [0] * len(tok_token)
		ori_token_idx = 0
		cover_num = 0
		for i, token in enumerate(tok_token):
			if not ori_token[ori_token_idx].startswith(token,cover_num):
				print ori_token
				print tok_token
				print ori_token[ori_token_idx]
				print token
				print cover_num
				return None
			tokens_map[i] = ori_token_idx
			cover_num += len(token)
			if cover_num == len(ori_token[ori_token_idx]):
				cover_num = 0
				ori_token_idx += 1
		return tokens_map


	@staticmethod
	def _ReadSentTags(sent):
		TokenList = []
		TagList = []
		BIOList = []
		pos = 0
		while True:
			if pos >= len(sent):
				break
			if sent[pos] == '<':	# meet semantic tags
				space_pos = sent.find(' ', pos)
				if space_pos == -1:
					print sent
					return (None,None,None)
				TagName = sent[pos+1:space_pos]
				end_pos = sent.find('>', pos)
				FullTag = ','.join(sent[pos:end_pos+1].split(' '))

				endTag_pos = sent.find('</', end_pos +1)
				ContentInTag = sent[end_pos+1:endTag_pos]
				tokens = ContentInTag.strip().split(' ')
				for i, token in enumerate(tokens):
					TokenList.append(token)
					TagList.append(FullTag)
					if i == 0:
						BIOList.append('%s-B' %(TagName))
					else:
						BIOList.append('%s-I' %(TagName))
				pos = endTag_pos + 3 + len(TagName)
			elif sent[pos] == ' ':
				pos += 1
			else:					# normal words	
				TagList.append('NULL')
				BIOList.append('O')
				space_pos = sent.find(' ', pos)
				tag_pos = sent.find('<', pos)

				if tag_pos == -1:
					if space_pos == -1:
						next_pos = -1
					else:
						next_pos = space_pos
				else:
					if space_pos == -1:
						next_pos = tag_pos
					else:
						next_pos = min(tag_pos, space_pos)

				if next_pos == -1:
					TokenList.append(sent[pos:])
					break
				else:
					TokenList.append(sent[pos:next_pos])
					pos = next_pos
		return (TokenList, TagList, BIOList)



	def _tokenize(self, sent):
		return self.tokenizer.tokenize(sent)







def main(argv):
	# 读取配置文件
	InitConfig()
	config = GetConfig()
	config.read([os.path.join(os.path.dirname(__file__),'../config/msiip_simple.cfg')])

	# 设置logging
	log_level_key = config.get('logging','level')
	run_code_name = os.path.basename(sys.argv[0])[0:-3]
	logging.basicConfig(filename = os.path.join(os.path.dirname(__file__), '../../output/logs', '%s_%s.log' %(run_code_name,time.strftime('%Y-%m-%d',time.localtime(time.time())))), \
    					level = GetLogLevel(log_level_key), 
    					format = '%(asctime)s %(levelname)8s %(lineno)4d %(module)s:%(name)s.%(funcName)s: %(message)s')
	
	parser = argparse.ArgumentParser(description='Extract Semantic Tag Data.')
	parser.add_argument('sub_utters_file', help='sub_utters_file')
	parser.add_argument('output', help='Output file')

	args = parser.parse_args()

	extractor = SemTagExtractor()
	output = codecs.open(args.output, 'w', 'utf-8')

	walker = sub_utters_data_walker(args.sub_utters_file)
	count = 0
	for (pre_utter, cul_utter) in walker.ReadUtter():
		for sub_tag in cul_utter['sub_tag_list']:
			count += 1
			if count % 100 == 0:
				sys.stderr.write('%d\n' %(count))
			extractor.ProcSent(sub_tag)
			if extractor.success:
				for (token, lemma, POStag, SemTag, BIOTag) in zip(extractor.tok_TokenList, extractor.tok_LemmaList, extractor.tok_POSList, extractor.tok_TagList, extractor.tok_BIOList):
					if token.istitle():
						title_flag = 'Y'
					else:
						title_flag = 'N'
					print >>output, '%s\t%s\t%s\t%s\t%s\t%s' %(token, lemma, POStag, title_flag, SemTag, BIOTag)
				print >>output
	output.close()

	
	

	#sent = "why we want to go to <AREA FROM-TO=\"TO\" REL=\"NONE\" CAT=\"CITY\">Singapore</AREA> is really %um to really try out the <FOOD CAT=\"MAIN\" FROM-TO=\"NONE\" REL=\"NONE\">food</FOOD>."
	#sent = "what kind of <FOOD CAT=\"MAIN\" FROM-TO=\"NONE\" REL=\"NONE\">food</FOOD> does he like then?"
	#extractor = SemTagExtractor(sent)


if __name__ =="__main__":
	main(sys.argv)

