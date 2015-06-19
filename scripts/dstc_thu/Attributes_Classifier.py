#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
temp attributes classifier
train svm model
'''


'''
test, little change
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

sys.path.append('/home/limiao/open_tools/liblinear-1.96/python')
from liblinearutil import save_model
from liblinearutil import load_model
from liblinear import *



class attr_feature(object):
	MY_ID = 'attr_feature'
	def __init__(self, feature_list = ['UNIGRAM', 'BIGRAM'], percent = 0.8, tokenizer_mode=None):
		'''
		available feature:
			UNIGRAM
			BIGRAM
		'''
		self.config = GetConfig()
		if tokenizer_mode:
			self.tokenizer_mode = tokenizer_mode
		else:
			self.tokenizer_mode = self.config.get(self.MY_ID,'tokenizer_mode')

		self.tokenizer = tokenizer(self.tokenizer_mode)
		self.stemmer = stemmer()


		self.feature_list = feature_list
		self.percent = percent
		self.UNI_LEX = None
		self.BI_LEX = None
		self.UNI_LEX_weight = None
		self.BI_LEX_weight = None

		self.UNI_LEX_offset = 0
		self.BI_LEX_offset = 0
		self.is_set = False

		self.appLogger = logging.getLogger(self.MY_ID)
	
	def _stat_unigram_lexicon(self, attr_data):
		feature_samples = []
		label_samples = []
		sys.stderr.write('prepare unigram stat corpus ...')

		for i, utter in enumerate(attr_data['sub_utter_data']):
			feature_sample = []
			label_sample = []
			tokens = self._preprocessing(utter)
			feature_sample.extend(tokens)

			for attr in attr_data['attr_data_index']:
				if i in attr_data['attr_data_index'][attr]:
					label_sample.append(attr)

			feature_samples.append(feature_sample)
			label_samples.append(label_sample)

		sys.stderr.write('Stat unigram CHI ...')
		self.appLogger.debug('UNIGRAM feature_samples len: %d, label_samples len: %d' %(len(feature_samples), len(label_samples)))
		unigram_chi = Stat_CHI_MultiLabel(feature_samples, label_samples)
		# test codes
		sorted_unigram_chi = sorted(unigram_chi.items(), key = lambda x:x[1], reverse = True)

		self.appLogger.debug('unigram chi:')
		for key, value in sorted_unigram_chi:
			self.appLogger.debug('%s, %s' %(key.encode('utf-8'), value)) 

		chosen_unigram = ChooseFromCHI(unigram_chi, self.percent)
		sys.stderr.write('Finish stat unigram CHI ...')

		uni_lex = {}
		for uni in chosen_unigram:
			uni_lex[uni] = len(uni_lex) + 1
		return uni_lex


	def _stat_unigram_weight(self, attr_data, UNI_LEX):
		UNI_LEX_weight = {}
		for key in UNI_LEX:
			UNI_LEX_weight[key] = 0.0
		N = 0.0
		for i, utter in enumerate(attr_data['sub_utter_data']):
			N += 1
			unigram_list = []
			tokens = self._preprocessing(utter)
			unigram_list = list(set(tokens))
			for uni in unigram_list:
				if uni in UNI_LEX:
					UNI_LEX_weight[uni] += 1

		for uni in UNI_LEX_weight:
			UNI_LEX_weight[uni] = math.log(N/UNI_LEX_weight[uni])

		return UNI_LEX_weight

	def _stat_bigram_lexicon(self, attr_data):
		feature_samples = []
		label_samples = []
		sys.stderr.write('prepare bigram stat corpus ...')

		for i, utter in enumerate(attr_data['sub_utter_data']):
			feature_sample = []
			label_sample = []
			tokens = self._preprocessing(utter)
			tokens.insert(0,'*')
			tokens.append('*')
			for j in range(len(tokens)-1):
				key = '%s, %s' %(tokens[j],tokens[j+1])
				feature_sample.append(key)

			for attr in attr_data['attr_data_index']:
				if i in attr_data['attr_data_index'][attr]:
					label_sample.append(attr)

			self.appLogger.debug('%s' %(feature_sample.__str__()))
			self.appLogger.debug('%s' %(label_sample.__str__()))

			feature_samples.append(feature_sample)
			label_samples.append(label_sample)

		sys.stderr.write('Stat bigram CHI ...')
		self.appLogger.debug('BIGRAM feature_samples len: %d, label_samples len: %d' %(len(feature_samples), len(label_samples)))
		bigram_chi = Stat_CHI_MultiLabel(feature_samples, label_samples)
		# test codes
		sorted_bigram_chi = sorted(bigram_chi.items(), key = lambda x:x[1], reverse = True)
		self.appLogger.debug('bigram chi:')
		for key, value in sorted_bigram_chi:
			self.appLogger.debug('%s, %s' %(key.encode('utf-8'), value)) 

		chosen_bigram = ChooseFromCHI(bigram_chi, self.percent)
		sys.stderr.write('Finish stat bigram CHI ...')
		bi_lex = {}
		for bi in chosen_bigram:
			bi_lex[bi] = len(bi_lex) + 1
		return bi_lex


		

	def _stat_bigram_weight(self, attr_data, BI_LEX):
		BI_LEX_weight = {}
		for key in BI_LEX:
			BI_LEX_weight[key] = 0.0
		N = 0.0
		for i, utter in enumerate(attr_data['sub_utter_data']):
			N += 1
			bigram_list = []
			tokens = self._preprocessing(utter)
			tokens.insert(0,'*')
			tokens.append('*')
			for i in range(len(tokens)-1):
				key = '%s, %s' %(tokens[i],tokens[i+1])
				bigram_list.append(key)

			bigram_list = list(set(bigram_list))
			for bi in bigram_list:
				if bi in BI_LEX:
					BI_LEX_weight[bi] += 1

		for bi in BI_LEX_weight:
			BI_LEX_weight[bi] = math.log(N/BI_LEX_weight[bi])

		return BI_LEX_weight


	

	# test function
	def _temp_stat_Lexicon(self, attr_data):
		self._stat_unigram_lexicon(attr_data)
		#self._stat_bigram_lexicon(attr_data)
	
	def stat_Lexicon(self, attr_data):
		'''
		stat_Lexicon
		input attr_data
		'''

		if 'UNIGRAM' in self.feature_list:
			self.UNI_LEX = self._stat_unigram_lexicon(attr_data)
			self.UNI_LEX_weight = self._stat_unigram_weight(attr_data, self.UNI_LEX)


		if 'BIGRAM' in self.feature_list:
			self.BI_LEX = self._stat_bigram_lexicon(attr_data)
			self.BI_LEX_weight = self._stat_bigram_weight(attr_data, self.BI_LEX)		

		self._prepare_resources()


	def load_Lexicon(self, Lexicon_file):
		input = codecs.open(Lexicon_file, 'r', 'utf-8')
		in_json = json.load(input)
		input.close()
		self.feature_list = in_json['feature_list']
		self.percent = in_json['percent']
		self.UNI_LEX = in_json['UNI_LEX']
		self.BI_LEX = in_json['BI_LEX']
		self.UNI_LEX_weight = in_json['UNI_LEX_weight']
		self.BI_LEX_weight = in_json['BI_LEX_weight']

		self.tokenizer_mode = in_json['tokenizer_mode']
		self.tokenizer = tokenizer(self.tokenizer_mode)

		self._prepare_resources()


	def save_Lexicon(self, Lexicon_file):
		output = codecs.open(Lexicon_file, 'w', 'utf-8')
		out_json = {}
		out_json['tokenizer_mode'] = self.tokenizer_mode
		out_json['feature_list'] = self.feature_list
		out_json['percent'] = self.percent
		out_json['UNI_LEX'] = self.UNI_LEX
		out_json['BI_LEX'] = self.BI_LEX
		out_json['UNI_LEX_weight'] = self.UNI_LEX_weight
		out_json['BI_LEX_weight'] = self.BI_LEX_weight
		json.dump(out_json, output, indent=4)
		output.close()


	def ExtractFeatureFromSent(self, sent):
		'''
		extract feature vector based on the Lexicon
		sent is the transcript of one sentence(utterance)
		'''
		if not self.is_set:
			raise Exception("Error: feature module's lexicon is not ready, please stat in a training corpus of read from file!")
		feature_vector = {}

		tokens = self._preprocessing(sent)
		if 'UNIGRAM' in self.feature_list:
			for token in tokens:
				if token in self.UNI_LEX:
					f_id = self.UNI_LEX[token] + self.UNI_LEX_offset
					weight = self.UNI_LEX_weight[token]
					if f_id not in feature_vector:
						feature_vector[f_id] = weight
					else:
						feature_vector[f_id] += weight	

		if 'BIGRAM' in self.feature_list:
			tokens.insert(0,'*')
			tokens.append('*')
			for i in range(len(tokens)-1):
				key = '%s, %s' %(tokens[i],tokens[i+1])
				if key in self.BI_LEX:
					f_id = self.BI_LEX[key] + self.BI_LEX_offset
					weight = self.BI_LEX_weight[key]
					if f_id not in feature_vector:
						feature_vector[f_id] = weight
					else:
						feature_vector[f_id] += weight
		return feature_vector

	
	def ExtractFeatureFromUtter(self, utter):
		'''
		extract feature vector based on the Lexicon
		'''
		if not self.is_set:
			raise Exception("Error: feature module's lexicon is not ready, please stat in a training corpus of read from file!")
		feature_vector = {}

		'''
		if 'TOPIC' in self.feature_list:
			topic = utter['segment_info']['topic']
			if topic in self.TOPIC_LEX:
				f_id = self.TOPIC_LEX[topic] + self.TOPIC_LEX_offset
				feature_vector[f_id] = 1
		'''


		transcript = utter['transcript']
		self.appLogger.debug('%s' %(transcript))
		
		tokens = self._preprocessing(transcript)
		if 'UNIGRAM' in self.feature_list:
			for token in tokens:
				if token in self.UNI_LEX:
					f_id = self.UNI_LEX[token] + self.UNI_LEX_offset
					weight = self.UNI_LEX_weight[token]
					if f_id not in feature_vector:
						feature_vector[f_id] = weight
					else:
						feature_vector[f_id] += weight	

		if 'BIGRAM' in self.feature_list:
			tokens.insert(0,'*')
			tokens.append('*')
			for i in range(len(tokens)-1):
				key = '%s, %s' %(tokens[i],tokens[i+1])
				if key in self.BI_LEX:
					f_id = self.BI_LEX[key] + self.BI_LEX_offset
					weight = self.BI_LEX_weight[key]
					if f_id not in feature_vector:
						feature_vector[f_id] = weight
					else:
						feature_vector[f_id] += weight
		
		return feature_vector

	def _preprocessing(self, sent):
		'''
		convert to lower type
		tokenization and stemming
		'''
		sent = sent.lower()
		tokens = self.tokenizer.tokenize(sent)
		new_tokens = [self.stemmer.stem(tk) for tk in tokens]
		return new_tokens

	

	def _set_offset(self):
		self.UNI_LEX_offset = 0
		self.BI_LEX_offset = 0

		if 'UNIGRAM' in self.feature_list:
			self.BI_LEX_offset = self.UNI_LEX_offset + len(self.UNI_LEX)


	def _prepare_resources(self):
		self._set_offset()
		self.is_set = True


def CompareCompareResult(c1,c2):
	if c1[1][5] > c2[1][5]:
		return 1
	elif c1[1][5] < c2[1][5]:
		return -1
	else:
		return cmp(c1[1][0], c2[1][0])


class attributes_classifier(object):
	MY_ID = 'ATTRIBUTE_CLASSIFIER'
	def __init__(self):
		self.config = GetConfig()
		self.feature = None
		self.models = {}
		self.attrs = []
		self.is_set = False
		self.appLogger = logging.getLogger(self.MY_ID)

		self.train_samples = None
		self.attrs_labels = None


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

	def TrainFromAttrDictFiles(self, attr_dict_file, feature_list, percent, tokenizer_mode, model_dir):
		# deal with model dir
		if os.path.exists(model_dir):
			shutil.rmtree(model_dir,True)
		os.mkdir(model_dir)

		input = codecs.open(attr_dict_file, 'r', 'utf-8')
		attr_data = json.load(input)
		input.close()

		# stat train samples
		print 'stat train samples'
		self.feature = attr_feature(feature_list, percent, tokenizer_mode)
		self.feature.stat_Lexicon(attr_data)
		self.attrs = attr_data['attr_data_index'].keys()

		self.train_samples = []
		for utter in attr_data['sub_utter_data']:
			feature_vector = self.feature.ExtractFeatureFromSent(utter)
			self.train_samples.append(feature_vector)

		self.attrs_labels = {}
		for attr in self.attrs:
			self.attrs_labels[attr] = [0] * len(self.train_samples)
			for index in attr_data['attr_data_index'][attr]:
				self.attrs_labels[attr][index] = 1


		# train svm model
		print 'train svm models'
		for attr in self.attrs:
			print 'Train attr: %s' %(attr)
			prob = problem(self.attrs_labels[attr], self.train_samples)
			param = parameter('-s 0 -c 1')
			self.models[attr] = liblinear.train(prob, param)

		# save model
		print 'save models'
		out_json = {}
		out_json['attrs'] = self.attrs
		out_json['train_samples_file'] = 'train_samples.json'
		out_json['feature_lexicon_file'] = 'feature_lexicon.json'
		out_json['ontology_file'] = 'ontology.json'
		output = codecs.open(os.path.join(model_dir, 'config.json'), 'w', 'utf-8')
		json.dump(out_json, output, indent=4)
		output.close()

		# save train samples
		output = codecs.open(os.path.join(model_dir, out_json['train_samples_file']), 'w', 'utf-8')
		train_sample_json = {}
		train_sample_json['samples'] = self.train_samples
		train_sample_json['label_index'] = self.attrs_labels
		json.dump(train_sample_json, output, indent=4)
		output.close()

		# save feature
		self.feature.save_Lexicon(os.path.join(model_dir, out_json['feature_lexicon_file']))

		# save svm models
		for attr in self.attrs:
			save_model(os.path.join(model_dir, '%s.svm.m' %(attr)), self.models[attr])

		print 'Done!'


	def TestAttrDictFiles(self, attr_data_file, model_dir):
		self.LoadModel(model_dir)
		if not self.is_set:
			raise Exception('Error: Fail to load model!')
		input = codecs.open(attr_data_file, 'r', 'utf-8')
		attr_data = json.load(input)
		input.close()

		utter_result = []
		compare_result = {}
		compare_result_all = [0] * 6
		for attr in self.attrs:
			compare_result[attr] = [0] * 6

		for i, utter in enumerate(attr_data['sub_utter_data']):
			gold_attr_label = {}
			for attr in self.attrs:
				if i in attr_data['attr_data_index'][attr]:
					gold_attr_label[attr] = 1
				else:
					gold_attr_label[attr] = 0

			result_attr,_ = self.PredictSent(utter)

			result_count = 0
			gold_count = 0
			right_count = 0

			for attr in self.attrs:
				gold_count += gold_attr_label[attr]
				result_count += result_attr[attr]

				compare_result[attr][0] += gold_attr_label[attr]
				compare_result[attr][1] += result_attr[attr]
				if gold_attr_label[attr] == result_attr[attr] and gold_attr_label[attr] == 1:
					compare_result[attr][2] += 1
					right_count += 1

			if result_count == 0 or gold_count == 0 or right_count == 0:
				utter_precison = 0.0
				utter_recall = 0.0
				utter_fscore = 0.0
			else:
				utter_precison = right_count * 1.0 / result_count
				utter_recall = right_count * 1.0 / gold_count
				utter_fscore = 2 * utter_precison * utter_recall / (utter_precison + utter_recall)

			utter_result.append([utter_precison, utter_recall, utter_fscore])

			compare_result_all[0] += gold_count
			compare_result_all[1] += result_count
			compare_result_all[2] += right_count

		# fix from here!!!
		utter_precison_all = sum([a[0] for a in utter_result]) / len(utter_result)
		utter_recall_all = sum([a[1] for a in utter_result]) / len(utter_result)
		utter_fscore_all = sum([a[2] for a in utter_result]) / len(utter_result)
		print '# utteracne performance:'
		print '%.3f %.3f %.3f' %(utter_precison_all, utter_recall_all, utter_fscore_all)

		for attr in self.attrs:
			if compare_result[attr][0] == 0 or compare_result[attr][1] == 0 or compare_result[attr][2] == 0:
				pass
			else:
				compare_result[attr][3] = compare_result[attr][2] * 1.0 / compare_result[attr][1]
				compare_result[attr][4] = compare_result[attr][2] * 1.0 / compare_result[attr][0]
				compare_result[attr][5] = 2 * compare_result[attr][3] * compare_result[attr][4] / (compare_result[attr][3] + compare_result[attr][4])

		compare_result_all[3] = compare_result_all[2] * 1.0 / compare_result_all[1]
		compare_result_all[4] = compare_result_all[2] * 1.0 / compare_result_all[0]
		compare_result_all[5] = 2 * compare_result_all[3] * compare_result_all[4] / (compare_result_all[3] + compare_result_all[4])


		print '# attr performance:'
		print 'Macro-average:'
		attr_Macro_average_precision = sum([v[3] for k,v in compare_result.items()]) / len(compare_result)
		attr_Macro_average_recall = sum([v[4] for k,v in compare_result.items()]) / len(compare_result)
		attr_Macro_average_fscore = sum([v[5] for k,v in compare_result.items()]) / len(compare_result)
		print '%.3f %.3f %.3f' %(attr_Macro_average_precision, attr_Macro_average_recall, attr_Macro_average_fscore)

		print 'Micro-average:\n%s' %(' '.join(['%.3f' %(i) for i in compare_result_all]))
		sorted_result = sorted(compare_result.items(), cmp=CompareCompareResult, reverse=True)
		for k,v in sorted_result:
			print '%s: %s' %(k, ' '.join(['%.3f' %(i) for i in v])) 



	def PredictSent(self, sent):
		if not self.is_set:
			raise Exception('Error: Model is not ready for predict!')
		result = {}
		result_prob = {}
		feature_vector = self.feature.ExtractFeatureFromSent(sent)
		self.appLogger.debug('%s' %(feature_vector.__str__()))
		for attr in self.attrs:
			(label, prob) = self.svm_predict(self.models[attr], feature_vector)
			result[attr] = label
			result_prob[attr] = prob
			# print label, prob
		return result, result_prob


	def LoadModel(self, model_dir):
		# load config
		input = codecs.open(os.path.join(model_dir,'config.json'), 'r', 'utf-8')
		config_json = json.load(input)
		input.close()
		self.attrs = config_json['attrs']
		# load feature
		self.feature = attr_feature()
		self.feature.load_Lexicon(os.path.join(model_dir,config_json['feature_lexicon_file']))
		if not self.feature.is_set:
			raise Exception('Fail to load feature module!')
		# load svm model
		for attr in self.attrs:
			self.models[attr] = load_model(os.path.join(model_dir, '%s.svm.m' %(attr)))
		self.is_set = True





def GetFeatureList(feature_code):
	if feature_code == None or feature_code == '':
		feature_code = 'ub'
	feature_list = []
	feature_code_list = list(feature_code)


	if 'u' in feature_code_list:
		feature_list.append('UNIGRAM')
	if 'b' in feature_code_list:
		feature_list.append('BIGRAM')
	return feature_list
	



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
	

	parser = argparse.ArgumentParser(description='liblinear attributes classifier.')
	parser.add_argument('attr_data_file', help='attr_data_file')
	parser.add_argument('model_dir',metavar='PATH', help='The output model dir')
	parser.add_argument('--train',dest='train',action='store_true', help='train or test.')
	parser.add_argument('--feature',dest='feature',action='store', help='feature to use. Example: TubB')
	parser.add_argument('--mode',dest='mode',action='store', help='tokenizer mode')
	parser.add_argument('--percent',dest='percent',type=float,action='store', help='CHI chosen percent')
	parser.add_argument('--test',dest='test',action='store_true', help='train or test.')
	# parser.add_argument('--TestPre',dest='test_preprocessing',action='store_true', help='train or test.')
	
	args = parser.parse_args()

	feature_list = GetFeatureList(args.feature)

	ac = attributes_classifier()

	if args.test and args.train:
		sys.stderr.write('Error: train and test can not be both ture!')
	elif not (args.test or args.train):
		sys.stderr.write('Error: train and test can not be both false!')
	elif args.test:
		print 'Test!'
		ac.TestAttrDictFiles(args.attr_data_file,args.model_dir)
	else:
		print 'Train'
		ac.TrainFromAttrDictFiles(args.attr_data_file, feature_list, args.percent, args.mode, args.model_dir)
	
		

if __name__ =="__main__":
	main(sys.argv)

