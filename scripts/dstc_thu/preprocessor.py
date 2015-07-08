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
	def __init__(self, flag=True):
		#self.config = GetConfig()
		#from nltk.stem.lancaster import LancasterStemmer as Stemmer
		self.flag = flag
		if self.flag:
			from nltk.stem.porter import PorterStemmer as Stemmer
			self.st = Stemmer()

	def stem(self, string):
		if self.flag and string and string[0] != '$':
			return self.st.stem(string)
		else:
			return string

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


class NGRAM_builder(object):
	MY_ID='ngram_builder'
	def __init__(self, remove_stopwords=True, remove_punctuation=False, replace_num=True):
		self.remove_stopwords = remove_stopwords
		self.remove_punctuation = remove_punctuation
		self.replace_num = replace_num

		self.stopwords = set(['okay','yah','yes','alright','oh','uh','i','s','right','see','that','so','it','huh','hm','the','you',
							'ah','um','a','to','and','in','sure','of','can','is','but','great','this','correct','have','think','also',
							'me','m','be','very','for','ll','now','do','good','let','then','get','no','just','know','sorry','what',
							'my','would','two','really','well','welcome','one','are','we','will','on','because','with','not','wow',
							'please','understand','if','or','your','ooh','cool','here','want','yeah','re','quay','take','about','o',
							'up','nice','all','how','actually','yup','part','need','eight','fine','ve','day','they','t','guess','lot',
							'again','kinda','ten','from','give','may','course','bye','mean','eh','got','anyway','going','at','five',
							'by','ok','problem','other','exactly','v','along','still','too','r','were','say','any','don','definitely',
							'only','him','she','hmm','while','them','bit','d','was','an','as','hello','particular','did','p','k','a',
							'about','above','after','again','against','all','an','and','any','are','as','at','be','been','before','being',
							'below','between','both','but','by','cannot','could','did','do','does','doing','down','during','each','few',
							'for','from','had','has','have','having','he','her','hers','herself','him','himself','his','how','i','if',
							'in','into','is','it','its','itself','me','more','most','my','myself','no','nor','not','of','off','on','once',
							'only','or','other','ought','our','ours','ourselves','out','over','own','same','she','should','so','some',
							'such','than','that','the','their','theirs','them','themselves','then','these','they','this','those','through',
							'to','too','under','until','up','very','was','we','were','what','when','where','which','while','who','whom',
							'with','would','you','your','yours','yourself','yourselves'])
		self.re_puntuation = re.compile("^\W+$")
		self.card_numbers = set(['thousand','thousands','hundred','hundreds','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety',
								'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen',
								'zero','one','two','three','four','five','six','seven','eight','nine'])
		self.ordi_numbers = set(['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth',
								'tenth','eleventh','twelfth','thirteenth','fourteenth','fifteenth','sixteenth','seventeenth','eighteenth','nineteenth',
								'twentieth','thirtieth','fortieth','fiftieth','sixtieth','seventieth','eightieth','ninetieth'])

		self.day_name = set(['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])
		self.month_name = set(['january','february','march','april','may','june','july','august','september','october','november','december'])

	def RemoveStopWords(self, tokens):
		new_tokens = []
		for t in tokens:
			if t.lower() in self.stopwords:
				new_tokens.append('$ST')
			else:
				new_tokens.append(t)
		return new_tokens

	def RemovePunct(self, tokens):
		new_tokens = []
		for t in tokens:
			if self.re_puntuation.match(t):
				new_tokens.append('$PT')
			else:
				new_tokens.append(t)
		return new_tokens

	def ReplaceNum(self, tokens):
		new_tokens = []
		for t in tokens:
			tl = t.lower()
			if tl in self.card_numbers:
				new_tokens.append('$CNUM')
			elif tl in self.ordi_numbers:
				new_tokens.append('$ONUM')
			elif tl in self.day_name:
				new_tokens.append('$DAY')
			elif tl in self.month_name:
				new_tokens.append('$MONTH')
			else:
				new_tokens.append(t)
		return new_tokens

	def PreReplace(self, tokens):
		new_tokens = tokens
		if self.remove_stopwords:
			new_tokens = self.RemoveStopWords(new_tokens)
		if self.remove_punctuation:
			new_tokens = self.RemovePunct(new_tokens)
		if self.replace_num:
			new_tokens = self.ReplaceNum(new_tokens)
		return new_tokens

	def GenerateNGRAM(self, tokens, n):
		ngram_list = []
		new_tokens = tuple(tokens)
		for i in range(len(new_tokens) - (n - 1)):
			ngram = new_tokens[i:i+n]
			str_ngram = str(ngram)
			if self.remove_stopwords and str(ngram).find('$ST') != -1:
				continue
			if self.remove_punctuation and str(ngram).find('$PT') != -1:
				continue
			ngram_list.append(ngram)
		return ngram_list




if __name__ =="__main__":
	preproc = tokenizer('MINE')
	sents = ["%uh because this is a East-West Line %um you will have to change %uh the line in City Hall.",
			"So you actually get to view %uh you know the- some parts %uh of the island %uh and you know if you feel like stopping you can always stop-",
			"You are quite %uh you know what's going on from your airport, you take the subway-"
			]
	for sent in sents:
		tokens = preproc.tokenize(sent)
		print ' '.join(tokens)

