#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
slot value classifier
similiar to the STC method in (Mairesse et al., 2009).
first read from a config file
find which slot is enumeratable which is un-enumeratable
'''

import argparse, sys, time, json, os, math
import logging
import re
import codecs
import shutil
from collections import defaultdict
from ctypes import c_double


from Utils import *
from GlobalConfig import *

from preprocessor import *
from temp_baseline import BaselineTracker as SubSegBaselineTracker

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import ontology_reader
import dataset_walker
from baseline import BaselineTracker as BaselineTracker


sys.path.append('/home/limiao/open_tools/liblinear-1.96/python')
from liblinearutil import save_model
from liblinearutil import load_model
from liblinear import *


class Tuple_Extractor(object):
	MY_ID = 'Tuple_Extractor'
	'''
	read a config file
	know which slot is enumeratable and which is un-enumeratable

	then it can extract tuple from Frame_Label
	'''
	def __init__(self, slot_config_file = None):
		'''
		slot_config_file tells while slot is enumeratable and which is not
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

	def extract_tuple(self, frame_label):
		output_tuple = []
		for slot in frame_label:
			output_tuple.append('root:%s' %(slot))
			if self.slot_config[slot]:
				for value in frame_label[slot]:
					output_tuple.append('%s:%s' %(slot, value))
		return list(set(output_tuple))


class feature(object):
	MY_ID = 'svc_feature'
	def __init__(self, tagsets, tokenizer_mode=None, use_stemmer=None):
		self.config = GetConfig()
		self.appLogger = logging.getLogger(self.MY_ID)

		# tokenizer
		if tokenizer_mode:
			self.tokenizer_mode = tokenizer_mode
		else:
			self.tokenizer_mode = self.config.get(self.MY_ID,'tokenizer_mode')
		self.appLogger.debug('tokenizer mode: %s' %(self.tokenizer_mode))
		self.tokenizer = tokenizer(self.tokenizer_mode)

		# stemmer
		if use_stemmer == None:
			use_stemmer = self.tokenizer_mode = self.config.getboolean(self.MY_ID,'use_stemmer')
		self.appLogger.debug('use stemmer ? %s' %(use_stemmer))	
		self.use_stemmer = use_stemmer
		self.stemmer = stemmer(use_stemmer)

		self.tagsets = tagsets

		self.feature_list = None
		self.unigram = False
		self.bigram = False
		self.trigram = False
		

		# feature vector
		self.UNI_LEX = None
		self.BI_LEX = None
		self.TRI_LEX = None

		self.UNI_LEX_weight = None
		self.BI_LEX_weight = None
		self.TRI_LEX_weight = None

		self.TOPIC_LEX = None
		self.BASELINE_LEX = None
		
		self.TOPIC_LEX_offset = 0
		self.UNI_LEX_offset = 0
		self.BI_LEX_offset = 0
		self.TRI_LEX_offset = 0
		self.BASELINE_LEX_offset = 0
		self.is_set = False

	def _set_offset(self):
		self.TOPIC_LEX_offset = 0
		self.UNI_LEX_offset = 0
		self.BI_LEX_offset = 0
		self.TRI_LEX_offset = 0
		self.BASELINE_LEX_offset = 0

		if 'TOPIC' in self.feature_list:
			self.UNI_LEX_offset = self.TOPIC_LEX_offset + len(self.TOPIC_LEX)

		if self.unigram:
			self.BI_LEX_offset = self.UNI_LEX_offset + len(self.UNI_LEX)

		if self.bigram:
			self.TRI_LEX_offset = self.BI_LEX_offset + len(self.BI_LEX)

		if self.trigram:
			self.BASELINE_LEX_offset = self.TRI_LEX_offset + len(self.TRI_LEX)

	def _preprocessing(self, sent):
		'''
		convert to lower type
		tokenization and stemming
		'''
		sent = sent.lower()
		tokens = self.tokenizer.tokenize(sent)
		new_tokens = [self.stemmer.stem(tk) for tk in tokens]
		return new_tokens


	def _prepare_resources(self):
		self._set_offset()
		self.is_set = True

	def load_Lexicon(self, Lexicon_file):
		input = codecs.open(Lexicon_file, 'r', 'utf-8')
		in_json = json.load(input)
		input.close()
		self.feature_list = in_json['feature_list']
		self.unigram = in_json['feature_unigram']
		self.bigram = in_json['feature_bigram']
		self.trigram = in_json['feature_trigram']
		self.UNI_LEX = in_json['UNI_LEX']
		self.BI_LEX = in_json['BI_LEX']
		self.TRI_LEX = in_json['TRI_LEX']
		self.UNI_LEX_weight = in_json['UNI_LEX_weight']
		self.BI_LEX_weight = in_json['BI_LEX_weight']
		self.TRI_LEX_weight = in_json['TRI_LEX_weight']

		self.TOPIC_LEX = in_json['TOPIC_LEX']
		self.BASELINE_LEX = in_json['BASELINE_LEX']

		self.tokenizer_mode = in_json['tokenizer_mode']
		self.tokenizer = tokenizer(self.tokenizer_mode)

		self.use_stemmer = in_json['use_stemmer']
		self.stemmer = stemmer(self.use_stemmer)

		self._prepare_resources()


	def save_Lexicon(self, Lexicon_file):
		output = codecs.open(Lexicon_file, 'w', 'utf-8')
		out_json = {}
		out_json['tokenizer_mode'] = self.tokenizer_mode
		out_json['use_stemmer'] = self.use_stemmer
		out_json['feature_list'] = self.feature_list
		out_json['feature_unigram'] = self.unigram
		out_json['feature_bigram'] = self.bigram
		out_json['feature_trigram'] = self.trigram

		out_json['UNI_LEX'] = self.UNI_LEX
		out_json['BI_LEX'] = self.BI_LEX
		out_json['TRI_LEX'] = self.TRI_LEX
		out_json['UNI_LEX_weight'] = self.UNI_LEX_weight
		out_json['BI_LEX_weight'] = self.BI_LEX_weight
		out_json['TRI_LEX_weight'] = self.TRI_LEX_weight

		out_json['TOPIC_LEX'] = self.TOPIC_LEX
		out_json['BASELINE_LEX'] = self.BASELINE_LEX
		json.dump(out_json, output, indent=4)
		output.close()


	def Stat_Lexicon(self, train_samples, label_samples,  feature_list = ['TOPIC', 'NGRAM_u:b', 'BASELINE']):
		'''
		train samples is a list of samples
		each item is a list , each item of the list is correspond to the feature list
		'''
		if len(train_samples) != len(label_samples):
			self.appLogger.error('Error: size of train samples and label samples mismatch! %d : %d' %(len(train_samples), len(label_samples)))
			raise Exception('Error: size of train samples and label samples mismatch! %d : %d' %(len(train_samples), len(label_samples)))
		if len(train_samples) == 0:
			self.appLogger.error('Error: No samples!')
			raise Exception('Error: No samples!')

		self.feature_list = feature_list
		sample_field_num = len(train_samples[0])
		if sample_field_num != len(self.feature_list):
			self.appLogger.error('Error: size of sample field num and feature list mismatch! %d : %d' %(sample_field_num, len(self.feature_list)))
			raise Exception('Error: size of sample field num and feature list mismatch! %d : %d' %(sample_field_num, len(self.feature_list)))
		'''
		print feature_list
		print train_samples[0]
		print label_samples[0]
		'''

		for feature in feature_list:
			if feature.startswith('NGRAM'):
				ngram_feature = feature[6:]
				tokens = ngram_feature.split(':')
				for t in tokens:
					if t == 'u':
						self.unigram = True
						continue
					elif t == 'b':
						self.bigram = True
						continue
					elif t == 't':
						self.trigram = True
						continue
					else:
						self.appLogger.error('Unknown ngram feature! %s' %(ngram_feature))
						raise Exception('Unknown ngram feature! %s' %(ngram_feature))

		for i, feature in enumerate(self.feature_list):
			if feature == 'TOPIC':
				#print i
				topic_samples = [train_sample[i] for train_sample in train_samples]
				#print topic_samples[0:3]
				self.TOPIC_LEX = self._stat_lexicon(topic_samples, threshold = 0)
			elif feature == 'BASELINE':
				#print i
				baseline_samples = [train_sample[i] for train_sample in train_samples]
				#print baseline_samples[0:3]
				self.BASELINE_LEX = self._stat_lexicon(baseline_samples, threshold = 0)
			elif feature.startswith('NGRAM'):
				#print i
				sent_samples = [train_sample[i] for train_sample in train_samples]
				#print sent_samples[0:3]
				unigram_lists = []
				bigram_lists = []
				trigram_lists = []
				for sent in sent_samples:
					#print sent
					tokens = self._preprocessing(sent)
					if self.unigram:
						unigram_lists.append(tokens)
					tokens.insert(0,'*')
					tokens.insert(0,'*')
					tokens.append('*')
					tokens.append('*')
					if self.bigram:
						bigram_tokens = []
						for j in range(1, len(tokens)-2):
							key = '%s, %s' %(tokens[i],tokens[i+1])
							bigram_tokens.append(key)
						bigram_lists.append(bigram_tokens)
					if self.trigram:
						trigram_tokens = []
						for j in range(len(len(tokens)-2)):
							key = '%s, %s, %s'%(tokens[i],tokens[i+1],tokens[i+2])
							trigram_tokens.append(key)
						trigram_lists.append(trigram_tokens)

				if self.unigram:
					self.UNI_LEX = self._stat_lexicon(unigram_lists, threshold=2)
					self.UNI_LEX_weight = self._calc_feature_weight(unigram_lists, label_samples, self.UNI_LEX, 'simple')
				if self.bigram:
					self.BI_LEX = self._stat_lexicon(bigram_lists, threshold=2)
					self.BI_LEX_weight = self._calc_feature_weight(bigram_lists, label_samples, self.BI_LEX,'simple')
				if self.trigram:
					self.TRI_LEX = self._stat_lexicon(trigram_lists, threshold=2)
					self.TRI_LEX_weight = self._calc_feature_weight(trigram_lists, label_samples, self.TRI_LEX,'simple')
			else:
				self.appLogger.error('Unknown feature! %s' %(feature))
				raise Exception('Unknown feature! %s' %(feature))
		return

	def _calc_feature_weight(self, feature_lists, label_samples, lexcion, method = 'simple'):
		lexicon_weight = {}
		if method == 'simple':
			for key in lexcion:
				lexicon_weight[key] = 1
		elif method == 'IDF':
			for key in lexcion:
				lexicon_weight[key] = 0.0
			N = len(feature_lists)
			for feature_list in feature_lists:
				f_list = list(set(feature_list))
				for f in f_list:
					if f in lexcion:
						lexicon_weight[f] += 1
			for f in lexicon_weight:
				lexicon_weight[f] = math.log(N/lexicon_weight[f])
		else:
			self.appLogger.error('Unknown weight calculate method! %s' %(method))
			raise Exception('Unknown weight calculate method! %s' %(method))

		return lexicon_weight

	def _stat_lexicon(self, feature_lists, threshold):
		lexicon_count = {}
		for feature in feature_lists:
			for f in feature:
				if f in lexicon_count:
					lexicon_count[f] += 1
				else:
					lexicon_count[f] = 0

		lexicon_out = {}
		for f, count in lexicon_count.items():
			if count > threshold:
				lexicon_out[f] = len(lexicon_out) + 1
		return lexicon_out

	def ExtractFeatureFromTuple(self, feature_tuple):
		if len(feature_tuple) != len(self.feature_list):
			self.appLogger.error('size of feature_tuple and the feature_list mismatch! %d : %d' (len(feature_tuple), len(feature_list)))
			raise Exception('size of feature_tuple and the feature_list mismatch! %d : %d' (len(feature_tuple), len(feature_list)))
		feature_vector = {}
		for i, feature in self.feature_list:
			if feature == 'TOPIC':
				for f in feature_tuple[i]:
					if f in self.TOPIC_LEX:
						idx = self.TOPIC_LEX_offset + self.TOPIC_LEX[f]
						if idx in feature_vector:
							feature_vector[idx] += 1
						else:
							feature_vector[idx] = 1
			elif feature == 'BASELINE':
				for f in feature_tuple[i]:
					if f in self.BASELINE_LEX:
						idx = self.BASELINE_LEX_offset + self.BASELINE_LEX[f]
						if idx in feature_vector:
							feature_vector[idx] += 1
						else:
							feature_vector[idx] = 1
			elif feature.startswith('NGRAM'):
				sent = feature_tuple[i]
				tokens = self._preprocessing(sent)
				if self.unigram:
					for tk in tokens:
						if tk in self.UNI_LEX:
							idx = self.UNI_LEX_offset + self.UNI_LEX[tk]
							weight = self.UNI_LEX_weight[tk]
							if idx in feature_vector:
								feature_vector[idx] += weight
							else:
								feature_vector[idx] = weight
				tokens.insert(0,'*')
				tokens.insert(0,'*')
				tokens.append('*')
				tokens.append('*')
				if self.bigram:
					for j in range(1, len(tokens)-2):
						key = '%s, %s' %(tokens[i],tokens[i+1])
						if key in self.BI_LEX:
							idx = self.BI_LEX_offset + self.BI_LEX[key]
							weight = self.BI_LEX_weight[key]
							if idx in feature_vector:
								feature_vector[idx] += weight
							else:
								feature_vector[idx] = weight

				if self.trigram:
					for j in range(len(len(tokens)-2)):
						key = '%s, %s, %s'%(tokens[i],tokens[i+1],tokens[i+2])
						if key in self.TRI_LEX:
							idx = self.TRI_LEX_offset + self.TRI_LEX[key]
							weight = self.TRI_LEX_weight[key]
							if idx in feature_vector:
								feature_vector[idx] += weight
							else:
								feature_vector[idx] = weight
		return feature_vector


def CompareCompareResult(c1,c2):
	if c1[1][5] > c2[1][5]:
		return 1
	elif c1[1][5] < c2[1][5]:
		return -1
	else:
		return cmp(c1[1][0], c2[1][0])


class slot_value_classifier(object):
	MY_ID = 'SLOT_VALUE_CLASSIFIER'
	def __init__(self):
		self.config = GetConfig()
		self.appLogger = logging.getLogger(self.MY_ID)
		self.models = {}
		self.model_keys = []
		self.ontology_file = ''
		self.tagsets = None
		self.feature = None
		self.is_set = False

	def _prepare_resources(self):
		if self.tagsets:
			self.baseline = BaselineTracker(self.tagsets)
		else:
			self.appLogger.error('Error: _prepare_resources(): Ontology tagsets not ready!')
			raise Exception('Error: _prepare_resources(): Ontology tagsets not ready!')

	def _extract_utter_tuple(self, utter, feature_list):
		train_sample = []
		topic = utter['segment_info']['topic']
		for i, feature in enumerate(feature_list):
			if feature == 'TOPIC':
				train_sample.append([topic])
			elif feature == 'BASELINE':
				self.baseline.addUtter(utter)
				baseline_out_label = self.baseline.frame
				train_sample.append(tuple_extractor.extract_tuple(baseline_out_label))
			elif feature.startswith('NGRAM'):
				train_sample.append(utter['transcript'])
			else:
				self.appLogger.error('Unknown feature: %s' %(feature))
				raise Exception('Unknown feature: %s' %(feature))
		return train_sample


	def TrainFromDataSet(self, ontology_file, feature_list, dataset, model_dir, tokenizer_mode, use_stemmer):
		# deal with model dir
		if os.path.exists(model_dir):
			shutil.rmtree(model_dir,True)
		os.mkdir(model_dir)

		self.ontology_file = ontology_file
		self.tagsets = ontology_reader.OntologyReader(ontology_file).get_tagsets()
		self._prepare_resources()


		# stat train samples
		tuple_extractor = Tuple_Extractor()
		label_samples = []
		train_samples = []
		for call in dataset:
			for (log_utter, label_utter) in call:
				if 'frame_label' in label_utter:
					frame_label = label_utter['frame_label']
					label_samples.append(tuple_extractor.extract_tuple(frame_label))
					train_samples.append(self._extract_utter_tuple(log_utter, feature_list))
		# stat lexicon
		self.feature = feature(self.tagsets, tokenizer_mode, use_stemmer)
		self.feature.Stat_Lexicon(train_samples, label_samples, feature_list)
		# extract feature, build training data
		for labels in label_samples:
			for label in labels:
				if label not in self.models:
					self.models[label] = None
		self.model_keys = self.models.keys()

		train_feature_samples = []
		for train_sample in train_samples:
			train_feature_samples.append(self.feature.ExtractFeatureFromTuple(train_sample))

		train_labels = {}
		for key in self.model_keys:
			train_labels[key] = [0] * len(train_feature_samples)
		for i,labels in enumerate(label_samples):
			for key in list(set(labels)):
				train_labels[key][i] = 1
		# begin train
		print 'train svm models...'
		for model_key in self.model_keys:
			print 'Train tuple: %s' %(model_key)
			prob = problem(train_labels[model_key], train_feature_samples)
			param = parameter('-s 0 -c 1')
			self.models[model_key] = liblinear.train(prob, param)
		
		# save model
		print 'save models'
		out_json = {}
		out_json['tuples'] = self.model_keys
		out_json['train_samples_file'] = 'train_samples.json'
		out_json['feature_lexicon_file'] = 'feature_lexicon.json'
		out_json['ontology_file'] = 'ontology.json'
		output = codecs.open(os.path.join(model_dir, 'config.json'), 'w', 'utf-8')
		json.dump(out_json, output, indent=4)
		output.close()

		# save ontology file
		shutil.copyfile(self.ontology_file, os.path.join(model_dir,out_json['ontology_file']))

		# save train samples
		output = codecs.open(os.path.join(model_dir, out_json['train_samples_file']), 'w', 'utf-8')
		out_json = {}
		out_json['train_samples'] = train_samples
		out_json['label_samples'] = label_samples
		out_json['train_feature_samples'] = train_feature_samples
		out_json['train_labels'] = train_labels
		json.dump(out_json, output, indent=4)
		output.close()

		# save feature
		self.feature.save_Lexicon(os.path.join(model_dir, out_json['feature_lexicon_file']))

		# save svm models
		for model_key in self.model_keys:
			save_model(os.path.join(model_dir, '%s.svm.m' %(model_key)), self.models[model_key])

		print 'Done!'

	def TestFromDataSet(self, dataset, model_dir):
		self.LoadMode(model_dir)
		if not self.is_set:
			raise Exception('Can not load model from :%s' %(model_dir))

	def LoadMode(self, model_dir):
		# load config
		input = codecs.open(os.path.join(model_dir,'config.json'), 'r', 'utf-8')
		config_json = json.load(input)
		input.close()
		self.model_keys = config_json['tuples']
		# load ontology
		self.ontology_file = os.path.join(model_dir,config_json['ontology_file'])
		self.tagsets = ontology_reader.OntologyReader(self.ontology_file).get_tagsets()
		# load feature
		self.feature = feature(self.tagsets)
		self.feature.load_Lexicon(os.path.join(model_dir,config_json['feature_lexicon_file']))
		if not self.feature.is_set:
			raise Exception('Fail to load feature module!')
		# load svm model
		for key in self.model_keys:
			self.models[key] = load_model(os.path.join(model_dir, '%s.svm.m' %(key)))
		self.is_set = True

	def PredictUtter(self, Utter, feature_list):
		sample_tuple = self._extract_utter_tuple(Utter, feature_list)
		feature_vector = self.feature.ExtractFeatureFromTuple(sample_tuple)

		result = {}
		result_prob = {}
		for key in self.model_keys:
			(label, prob) = self.svm_predict(self.models[key], feature_vector)
			result[key] = label
			result_prob[key] = prob
			# print label, prob
		return result, result_prob

	def svm_predict(self, model, feature_vector):
		is_prob_model = model.is_probability_model()
		x, idx = gen_feature_nodearray(feature_vector)
		nr_class = model.get_nr_class()
		if not is_prob_model:
			if nr_class <= 2:
				nr_classifier = 1
			else:
				nr_classifier = nr_class
			dec_values = (c_double * nr_classifier)()

			label = liblinear.predict_values(model, x, dec_values)
			values = dec_values[:nr_classifier]
			return (label, values)
		else:
			prob_estimates = (c_double * nr_class)()
			label = liblinear.predict_probability(model, x, prob_estimates)
			probs = prob_estimates[:nr_class]
			return (label, probs)




def GetFeatureList(feature_code):
	if feature_code == None or feature_code == '':
		self.appLogger.error('Error: Empty feature code!')
		raise Exception('Error: Empty feature code!')

	feature_list = []
	feature_code_list = list(feature_code)
	u_flag = False
	b_flag = False
	t_flag = False

	if 'T' in feature_code_list:
		feature_list.append('TOPIC')
	if 'u' in feature_code_list:
		u_flag = True
	if 'b' in feature_code_list:
		b_flag = True
	if 't' in feature_code_list:
		t_flag = True
	if u_flag or b_flag or t_flag:
		NGRAM_key = 'NGRAM_'
		if u_flag:
			NGRAM_key += 'u:'
		if b_flag:
			NGRAM_key += 'b:'
		if t_flag:
			NGRAM_key += 't:'
		NGRAM_key = NGRAM_key[:-1]
		feature_list.append(NGRAM_key)
	if 'B' in feature_code_list:
		feature_list.append('BASELINE')	
	if not feature_list:
		self.appLogger.error('Error: Empty feature list!')
		raise Exception('Error: Empty feature list!')
	return feature_list




	




def main(argv):
	
	# 读取配置文件
	InitConfig()
	config = GetConfig()
	config.read([os.path.join(os.path.dirname(__file__),'../config/msiip_simple.cfg')])

	# 设置logging
	log_level_key = config.get('logging','level')
	run_code_name = os.path.basename(sys.argv[0])[0:-3]
	logging.basicConfig(filename = os.path.join(os.path.dirname(__file__), '%s_%s.log' %(run_code_name,time.strftime('%Y-%m-%d',time.localtime(time.time())))), \
    					level = GetLogLevel(log_level_key), 
    					format = '%(asctime)s %(levelname)8s %(lineno)4d %(module)s:%(name)s.%(funcName)s: %(message)s')
	

	parser = argparse.ArgumentParser(description='STC like slot value classifier.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('model_dir',metavar='PATH', help='The output model dir')
	parser.add_argument('--train',dest='train',action='store_true', help='train or test.')
	parser.add_argument('--ontology',dest='ontology',action='store', help='Ontology file.')
	parser.add_argument('--feature',dest='feature',action='store', help='feature to use. Example: TubB')
	parser.add_argument('--mode',dest='mode',action='store', help='tokenizer mode')
	parser.add_argument('--UseST',dest='UseST',action='store_true', help='use stemmer or not.')
	parser.add_argument('--test',dest='test',action='store_true', help='train or test.')	
	args = parser.parse_args()

	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)
	feature_list = GetFeatureList(args.feature)

	svc = slot_value_classifier()

	if args.test and args.train:
		sys.stderr.write('Error: train and test can not be both ture!')
	elif not (args.test or args.train):
		sys.stderr.write('Error: train and test can not be both false!')
	elif args.test:
		print 'Test!'
		svc.TestFromDataSet(dataset,args.model_dir)
	else:
		print 'Train'
		svc.TrainFromDataSet(args.ontology, feature_list, dataset, args.model_dir, args.mode, args.UseST)

if __name__ =="__main__":
	main(sys.argv)

