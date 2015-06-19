#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
temp slot value classifier
use sub_segments file as input
extract train file for svm
train svm model
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
from baseline import BaselineTracker as BaselineTracker


sys.path.append('/home/limiao/open_tools/liblinear-1.96/python')
from liblinearutil import save_model
from liblinearutil import load_model
from liblinear import *



class feature(object):
	MY_ID = 'svc_feature'
	def __init__(self, tagsets, feature_list = ['TOPIC', 'UNIGRAM', 'BIGRAM' , 'BASELINE'], percent = 0.8, tokenizer_mode=None):
		'''
		available feature:
			UNIGRAM
			BIGRAM
			BASELINE
		'''
		self.config = GetConfig()
		if tokenizer_mode:
			self.tokenizer_mode = tokenizer_mode
		else:
			self.tokenizer_mode = self.config.get(self.MY_ID,'tokenizer_mode')

		self.tokenizer = tokenizer(self.tokenizer_mode)
		self.stemmer = stemmer()

		self.SubSeg_baseline = None
		self.baseline = None
		self.tagsets = tagsets

		self.feature_list = feature_list
		self.percent = percent
		self.UNI_LEX = None
		self.BI_LEX = None
		self.UNI_LEX_weight = None
		self.BI_LEX_weight = None

		self.TOPIC_LEX = None
		self.BASELINE_LEX = None
		
		self.TOPIC_LEX_offset = 0
		self.UNI_LEX_offset = 0
		self.BI_LEX_offset = 0
		self.BASELINE_LEX_offset = 0
		self.is_set = False

		self.utter_transcripts = []
		self.appLogger = logging.getLogger(self.MY_ID)
	
	def _stat_unigram_lexicon(self, sub_segments):
		feature_samples = []
		label_samples = []
		sys.stderr.write('prepare unigram stat corpus ...')

		for session in sub_segments['sessions']:
			for sub_seg in session['sub_segments']:
				feature_sample = []
				label_sample = []

				utter_list = sub_seg['utter_sents']
				for utter in utter_list:
					utter = utter[utter.find(' ') + 1:]
					tokens = self._preprocessing(utter)
					feature_sample.extend(tokens)

				for slot, values in sub_seg['frame_label'].items():
					if slot == 'INFO':
						for value in values:
							slot_name = '%s:%s' %(slot,value)
							label_sample.append(slot_name)
					else:
						label_sample.append(slot)

				feature_samples.append(feature_sample)
				label_samples.append(label_sample)

		sys.stderr.write('Stat unigram CHI ...')
		unigram_chi = Stat_CHI_MultiLabel(feature_samples, label_samples)
		chosen_unigram = ChooseFromCHI(unigram_chi, self.percent)
		uni_lex = {}
		for uni in chosen_unigram:
			uni_lex[uni] = len(uni_lex) + 1

		'''
		sys.stderr.write('Sort and output unigram CHI ...')
		# test codes
		sorted_unigram_chi = sorted(unigram_chi.items(), key = lambda x:x[1], reverse = True)

		print 'unigram chi:'
		for key, value in sorted_unigram_chi:
			print key.encode('utf-8'), value
		'''

		return uni_lex

	def _stat_unigram_weight(self,sub_segments, UNI_LEX):
		UNI_LEX_weight = {}
		for key in UNI_LEX:
			UNI_LEX_weight[key] = 0.0
		N = 0.0
		for session in sub_segments['sessions']:
			for sub_seg in session['sub_segments']:
				N += 1
				unigram_list = []
				utter_list = sub_seg['utter_sents']
				for utter in utter_list:
					utter = utter[utter.find(' ') + 1:]
					tokens = self._preprocessing(utter)
					unigram_list.extend(tokens)
				unigram_list = list(set(unigram_list))
				for uni in unigram_list:
					if uni in UNI_LEX:
						UNI_LEX_weight[uni] += 1

		for uni in UNI_LEX_weight:
			UNI_LEX_weight[uni] = math.log(N/UNI_LEX_weight[uni])

		return UNI_LEX_weight

	def _stat_bigram_weight(self,sub_segments, BI_LEX):
		BI_LEX_weight = {}
		for key in BI_LEX:
			BI_LEX_weight[key] = 0.0
		N = 0.0
		for session in sub_segments['sessions']:
			for sub_seg in session['sub_segments']:
				N += 1
				bigram_list = []
				utter_list = sub_seg['utter_sents']
				for utter in utter_list:
					utter = utter[utter.find(' ') + 1:]
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




	def _stat_bigram_lexicon(self, sub_segments):
		feature_samples = []
		label_samples = []

		for session in sub_segments['sessions']:
			for sub_seg in session['sub_segments']:
				feature_sample = []
				label_sample = []

				utter_list = sub_seg['utter_sents']
				for utter in utter_list:
					utter = utter[utter.find(' ') + 1:]
					tokens = self._preprocessing(utter)
					tokens.insert(0,'*')
					tokens.append('*')
					for i in range(len(tokens)-1):
						key = '%s, %s' %(tokens[i],tokens[i+1])
						feature_samples.append(key)

				for slot, values in sub_seg['frame_label'].items():
					if slot == 'INFO':
						for value in values:
							slot_name = '%s:%s' %(slot,value)
							label_sample.append(slot_name)
					else:
						label_sample.append(slot)

				feature_samples.append(feature_sample)
				label_samples.append(label_sample)

		bigram_chi = Stat_CHI_MultiLabel(feature_samples, label_samples)

		chosen_bigram = ChooseFromCHI(bigram_chi)
		bi_lex = {}
		for bi in chosen_bigram:
			bi_lex[bi] = len(bi_lex) + 1

		'''
		sys.stderr.write('Sort and output unigram CHI ...')
		# test codes
		sorted_unigram_chi = sorted(unigram_chi.items(), key = lambda x:x[1], reverse = True)

		print 'unigram chi:'
		for key, value in sorted_unigram_chi:
			print key.encode('utf-8'), value
		'''

		return uni_lex

	# test function
	def _temp_stat_Lexicon(self, sub_segments):
		self._stat_unigram_lexicon(sub_segments)
		#self._stat_bigram_lexicon(sub_segments)
	
	def stat_Lexicon(self, sub_segments):
		'''
		stat_Lexicon
		input sub_segments
		'''

		if 'TOPIC' in self.feature_list:
			self.TOPIC_LEX = {}
			for topic in self.tagsets:
				if topic not in self.TOPIC_LEX:
					self.TOPIC_LEX[topic] = len(self.TOPIC_LEX) + 1

		if 'UNIGRAM' in self.feature_list:
			self.UNI_LEX = self._stat_unigram_lexicon(sub_segments)
			self.UNI_LEX_weight = self._stat_unigram_weight(sub_segments, self.UNI_LEX)


		if 'BIGRAM' in self.feature_list:
			self.BI_LEX = self._stat_bigram_lexicon(sub_segments)
			self.BI_LEX = self._stat_bigram_weight(sub_segments, self.BI_LEX)

		# baseline lex
		if 'BASELINE' in self.feature_list:
			self.BASELINE_LEX = {}
			for topic in self.tagsets:
				for slot in self.tagsets[topic]:
					if slot == 'INFO':
						for value in self.tagsets[topic][slot]:
							slot_name = '%s:%s' %(slot, value)
							if slot_name not in self.BASELINE_LEX:
								self.BASELINE_LEX[slot_name] = len(self.BASELINE_LEX) + 1
					else:
						if slot not in self.BASELINE_LEX:
							self.BASELINE_LEX[slot] = len(self.BASELINE_LEX) + 1		

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
		self.TOPIC_LEX = in_json['TOPIC_LEX']
		self.BASELINE_LEX = in_json['BASELINE_LEX']

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

		out_json['TOPIC_LEX'] = self.TOPIC_LEX
		out_json['BASELINE_LEX'] = self.BASELINE_LEX
		json.dump(out_json, output, indent=4)
		output.close()


	def ExtractFeatureFromSubseg(self, sub_seg):
		'''
		extract feature vector based on the Lexicon
		'''
		if not self.is_set:
			raise Exception("Error: feature module's lexicon is not ready, please stat in a training corpus of read from file!")
		feature_vector = {}

		if 'TOPIC' in self.feature_list:
			topic = sub_seg['topic']
			if topic in self.TOPIC_LEX:
				f_id = self.TOPIC_LEX[topic] + self.TOPIC_LEX_offset
				feature_vector[f_id] = 1

		utter_list = sub_seg['utter_sents']
		for utter in utter_list:
			utter = utter[utter.find(' ') + 1:]
			tokens = self._preprocessing(utter)
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

		if 'BASELINE' in self.feature_list:
			baseline_out = self.SubSeg_baseline.addSubSeg(sub_seg)
			for slot in baseline_out:
				if slot == 'INFO':
					for value in baseline_out[slot]:
						slot_name = '%s:%s' %(slot,value)
						if slot_name in self.BASELINE_LEX:
							f_id = self.BASELINE_LEX[slot_name] + self.BASELINE_LEX_offset
							if f_id not in feature_vector:
								feature_vector[f_id] = 1
							else:
								feature_vector[f_id] += 1
				else:
					if slot in self.BASELINE_LEX:
						f_id = self.BASELINE_LEX[slot] + self.BASELINE_LEX_offset
						feature_vector[f_id] = 1
		
		return feature_vector

	def ExtractFeatureFromUtter(self, utter):
		'''
		extract feature vector based on the Lexicon
		'''
		if not self.is_set:
			raise Exception("Error: feature module's lexicon is not ready, please stat in a training corpus of read from file!")
		feature_vector = {}

		if utter['segment_info']['target_bio'] == 'B':
			self.utter_transcripts = []

		if 'TOPIC' in self.feature_list:
			topic = utter['segment_info']['topic']
			if topic in self.TOPIC_LEX:
				f_id = self.TOPIC_LEX[topic] + self.TOPIC_LEX_offset
				feature_vector[f_id] = 1


		transcript = utter['transcript']
		self.utter_transcripts.append(transcript)

		self.appLogger.debug('%s' %(self.utter_transcripts.__str__()))
		
		for ut in self.utter_transcripts:
			tokens = self._preprocessing(ut)
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

		if 'BASELINE' in self.feature_list:
			self.baseline.addUtter(utter)
			baseline_out = self.baseline.frame
			for slot in baseline_out:
				if slot == 'INFO':
					for value in baseline_out[slot]:
						slot_name = '%s:%s' %(slot,value)
						if slot_name in self.BASELINE_LEX:
							f_id = self.BASELINE_LEX[slot_name] + self.BASELINE_LEX_offset
							if f_id not in feature_vector:
								feature_vector[f_id] = 1
							else:
								feature_vector[f_id] += 1
				else:
					if slot in self.BASELINE_LEX:
						f_id = self.BASELINE_LEX[slot] + self.BASELINE_LEX_offset
						feature_vector[f_id] = 1
		
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
		self.TOPIC_LEX_offset = 0
		self.UNI_LEX_offset = 0
		self.BI_LEX_offset = 0
		self.BASELINE_LEX_offset = 0

		if 'TOPIC' in self.feature_list:
			self.UNI_LEX_offset = self.TOPIC_LEX_offset + len(self.TOPIC_LEX)

		if 'UNIGRAM' in self.feature_list:
			self.BI_LEX_offset = self.UNI_LEX_offset + len(self.UNI_LEX)

		if 'BIGRAM' in self.feature_list:
			self.BASELINE_LEX_offset = self.BI_LEX_offset + len(self.BI_LEX)


	def _prepare_resources(self):
		self._set_offset()
		if self.tagsets:
			self.SubSeg_baseline = SubSegBaselineTracker(self.tagsets)
			self.baseline = BaselineTracker(self.tagsets)
		else:
			raise Exception('Error: _prepare_resources(): Ontology tagsets not ready!')
		self.is_set = True


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

		self.feature = None
		self.models = {}
		self.train_samples = {}
		self.slots = []
		self.ontology_file = ''
		self.tagsets = None
		self.is_set = False
		self.appLogger = logging.getLogger(self.MY_ID)

	
	'''
	def __del__(self):
		if self.models and len(self.models) > 0:
			for slot, model in self.models.items():
				#liblinear.free_and_destroy_model(model)
				liblinear.free_model_content(model)
		self.models = {}
	'''
	

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


	def TrainFromSubsegFiles(self, ontology_file, sub_segments_file, feature_list, percent, tokenizer_mode, model_dir):
		# deal with model dir
		if os.path.exists(model_dir):
			shutil.rmtree(model_dir,True)
		os.mkdir(model_dir)

		self.ontology_file = ontology_file
		self.tagsets = ontology_reader.OntologyReader(ontology_file).get_tagsets()

		input = codecs.open(sub_segments_file, 'r', 'utf-8')
		sub_segments = json.load(input)
		input.close()

		# stat train samples
		print 'stat train samples'
		self.feature = feature(self.tagsets, feature_list, percent, tokenizer_mode)
		self.feature.stat_Lexicon(sub_segments)
		self.slots = self._stat_slot(sub_segments)
		for slot in self.slots:
			self.train_samples[slot] = [[],[]]

		for session in sub_segments['sessions']:
			print '%d' %(session['session_id'])
			for sub_seg in session['sub_segments']:
				feature_vector = self.feature.ExtractFeatureFromSubseg(sub_seg)
				mentioned_slots = []
				for slot, values in sub_seg['frame_label'].items():
					if slot == 'INFO':
						for value in values:
							slot_name = '%s:%s' %(slot,value)
							if slot_name not in mentioned_slots:
								mentioned_slots.append(slot_name)
					else:
						if slot not in mentioned_slots:
							mentioned_slots.append(slot)
				for slot, train_samples in self.train_samples.items():
					if slot in mentioned_slots:
						train_samples[0].append(1)
					else:
						train_samples[0].append(0)

					train_samples[1].append(feature_vector)

		# train svm model
		print 'train svm models'
		for slot, train_samples in self.train_samples.items():
			print 'Train slot: %s' %(slot)
			prob = problem(train_samples[0], train_samples[1])
			param = parameter('-s 0 -c 1')
			self.models[slot] = liblinear.train(prob, param)

		# save model
		print 'save models'
		out_json = {}
		out_json['slots'] = self.slots
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
		json.dump(self.train_samples, output, indent=4)
		output.close()

		# save feature
		self.feature.save_Lexicon(os.path.join(model_dir, out_json['feature_lexicon_file']))

		# save svm models
		for slot in self.slots:
			save_model(os.path.join(model_dir, '%s.svm.m' %(slot)), self.models[slot])

		print 'Done!'

	def TestSubsegFiles(self, sub_segments_file, model_dir):
		self.LoadModel(model_dir)
		if not self.is_set:
			raise Exception('Error: Fail to load model!')
		input = codecs.open(sub_segments_file, 'r', 'utf-8')
		sub_segments = json.load(input)
		input.close()

		seg_result = []
		compare_result = {}
		compare_result_all = [0] * 6
		for slot in self.slots:
			compare_result[slot] = [0] * 6

		for session in sub_segments['sessions']:
			for sub_seg in session['sub_segments']:
				# get gold frame
				gold_frame = {}
				mentioned_slots = []
				for slot, values in sub_seg['frame_label'].items():
					if slot == 'INFO':
						for value in values:
							slot_name = '%s:%s' %(slot,value)
							mentioned_slots.append(slot_name)
					else:
						mentioned_slots.append(slot)
				for slot in self.slots:
					if slot in mentioned_slots:
						gold_frame[slot] = 1
					else:
						gold_frame[slot] = 0

				result_frame,_ = self.PredictSubseg(sub_seg)

				result_count = 0
				gold_count = 0
				right_count = 0
				for slot in self.slots:
					gold_count += gold_frame[slot]
					result_count += result_frame[slot]

					compare_result[slot][0] += gold_frame[slot]
					compare_result[slot][1] += result_frame[slot]
					if gold_frame[slot] == result_frame[slot] and gold_frame[slot] == 1:
						compare_result[slot][2] += 1
						right_count += 1

				if result_count == 0 or gold_count == 0 or right_count == 0:
					seg_precison = 0.0
					seg_recall = 0.0
					seg_fscore = 0.0
				else:
					seg_precison = right_count * 1.0 / result_count
					seg_recall = right_count * 1.0 / gold_count
					seg_fscore = 2 * seg_precison * seg_recall / (seg_precison + seg_recall)

				seg_result.append([seg_precison, seg_recall, seg_fscore])

				compare_result_all[0] += gold_count
				compare_result_all[1] += result_count
				compare_result_all[2] += right_count

		seg_precison_all = sum([a[0] for a in seg_result]) / len(seg_result)
		seg_recall_all = sum([a[1] for a in seg_result]) / len(seg_result)
		seg_fscore_all = sum([a[2] for a in seg_result]) / len(seg_result)
		print '# segment performance:'
		print '%.3f %.3f %.3f' %(seg_precison_all, seg_recall_all, seg_fscore_all)

		for slot in self.slots:
			if compare_result[slot][0] == 0 or compare_result[slot][1] == 0 or compare_result[slot][2] == 0:
				pass
			else:
				compare_result[slot][3] = compare_result[slot][2] * 1.0 / compare_result[slot][1]
				compare_result[slot][4] = compare_result[slot][2] * 1.0 / compare_result[slot][0]
				compare_result[slot][5] = 2 * compare_result[slot][3] * compare_result[slot][4] / (compare_result[slot][3] + compare_result[slot][4])

		compare_result_all[3] = compare_result_all[2] * 1.0 / compare_result_all[1]
		compare_result_all[4] = compare_result_all[2] * 1.0 / compare_result_all[0]
		compare_result_all[5] = 2 * compare_result_all[3] * compare_result_all[4] / (compare_result_all[3] + compare_result_all[4])


		print '# slot performance:'
		print 'Macro-average:'
		slot_Macro_average_precision = sum([v[3] for k,v in compare_result.items()]) / len(compare_result)
		slot_Macro_average_recall = sum([v[4] for k,v in compare_result.items()]) / len(compare_result)
		slot_Macro_average_fscore = sum([v[5] for k,v in compare_result.items()]) / len(compare_result)
		print '%.3f %.3f %.3f' %(slot_Macro_average_precision, slot_Macro_average_recall, slot_Macro_average_fscore)

		print 'Micro-average:\n%s' %(' '.join(['%.3f' %(i) for i in compare_result_all]))
		sorted_result = sorted(compare_result.items(), cmp=CompareCompareResult, reverse=True)
		for k,v in sorted_result:
			print '%s: %s' %(k, ' '.join(['%.3f' %(i) for i in v])) 

		#print json.dumps(compare_result, indent=4)



	def PredictSubseg(self, sub_seg):
		if not self.is_set:
			raise Exception('Error: Model is not ready for predict!')
		result = {}
		result_prob = {}
		feature_vector = self.feature.ExtractFeatureFromSubseg(sub_seg)

		for slot in self.slots:
			(label, prob) = self.svm_predict(self.models[slot], feature_vector)
			result[slot] = label
			result_prob[slot] = prob
			# print label, prob
		return result, result_prob

	def PredictUtter(self, utter):
		if not self.is_set:
			raise Exception('Error: Model is not ready for predict!')
		result = {}
		result_prob = {}
		feature_vector = self.feature.ExtractFeatureFromUtter(utter)
		self.appLogger.debug('%s' %(feature_vector.__str__()))
		for slot in self.slots:
			(label, prob) = self.svm_predict(self.models[slot], feature_vector)
			result[slot] = label
			result_prob[slot] = prob
			# print label, prob
		return result, result_prob
	

	def LoadModel(self, model_dir):
		# load config
		input = codecs.open(os.path.join(model_dir,'config.json'), 'r', 'utf-8')
		config_json = json.load(input)
		input.close()
		self.slots = config_json['slots']
		# load ontology
		self.ontology_file = os.path.join(model_dir,config_json['ontology_file'])
		self.tagsets = ontology_reader.OntologyReader(self.ontology_file).get_tagsets()
		# load feature
		self.feature = feature(self.tagsets)
		self.feature.load_Lexicon(os.path.join(model_dir,config_json['feature_lexicon_file']))
		if not self.feature.is_set:
			raise Exception('Fail to load feature module!')
		# load svm model
		for slot in self.slots:
			self.models[slot] = load_model(os.path.join(model_dir, '%s.svm.m' %(slot)))
		self.is_set = True

	def _stat_slot(self, sub_segments):
		slots = defaultdict(int)
		for session in sub_segments['sessions']:
			for sub_seg in session['sub_segments']:
				for slot, values in sub_seg['frame_label'].items():
					if slot == 'INFO':
						for value in values:
							slot_name = '%s:%s' %(slot,value)
							slots[slot_name] += 1
					else:
						slots[slot] += 1
		return dict(slots)









def GetFeatureList(feature_code):
	if feature_code == None or feature_code == '':
		feature_code = 'TubB'
	feature_list = []
	feature_code_list = list(feature_code)

	if 'T' in feature_code_list:
		feature_list.append('TOPIC')
	if 'u' in feature_code_list:
		feature_list.append('UNIGRAM')
	if 'b' in feature_code_list:
		feature_list.append('BIGRAM')
	if 'B' in feature_code_list:
		feature_list.append('BASELINE')	
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
	

	parser = argparse.ArgumentParser(description='liblinear slot value classifier.')
	parser.add_argument('sub_segments_file', help='sub_segments_file')
	parser.add_argument('model_dir',metavar='PATH', help='The output model dir')
	parser.add_argument('--train',dest='train',action='store_true', help='train or test.')
	parser.add_argument('--ontology',dest='ontology',action='store', help='Ontology file.')
	parser.add_argument('--feature',dest='feature',action='store', help='feature to use. Example: TubB')
	parser.add_argument('--mode',dest='mode',action='store', help='tokenizer mode')
	parser.add_argument('--percent',dest='percent',type=float,action='store', help='CHI chosen percent')
	parser.add_argument('--test',dest='test',action='store_true', help='train or test.')
	# parser.add_argument('--TestPre',dest='test_preprocessing',action='store_true', help='train or test.')
	
	args = parser.parse_args()

	feature_list = GetFeatureList(args.feature)

	svc = slot_value_classifier()
	'''
	if args.test_preprocessing:
		input = codecs.open(args.sub_segments_file, 'r', 'utf-8')
		sub_segments = json.load(input)
		input.close()
		feat = feature(None)
		count = 0
		for session in sub_segments['sessions']:
			for sub_seg in session['sub_segments']:
				utter_list = sub_seg['utter_sents']
				for utter in utter_list:
					count += 1
					if count % 100 == 0:
						sys.stderr.write('%d\n' %(count))
					utter = utter[utter.find(' ') + 1:]
					tokens = feat._preprocessing(utter)
					print ' ;'.join(tokens).encode('utf-8')
		return
	'''
	if args.test:
		svc.LoadModel(args.model_dir)
		return
	
	if args.train:
		print 'Train'
		if args.ontology:
			svc.TrainFromSubsegFiles(args.ontology, args.sub_segments_file, feature_list, args.percent, args.mode, args.model_dir)
		else:
			print 'Error: No ontology file!'
	else:
		print 'Test!'
		svc.TestSubsegFiles(args.sub_segments_file,args.model_dir)

if __name__ =="__main__":
	main(sys.argv)

