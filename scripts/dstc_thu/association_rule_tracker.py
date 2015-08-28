#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
The association rule tracker

it first use crf tagger to find semtag

then output frame based on the association rules
'''


import argparse, sys, time, json, os, copy, codecs
import logging
from fuzzywuzzy import fuzz
import math

from GlobalConfig import *

from BeliefState import BeliefState
from dstc4_rules import DSTC4_rules
from Utils import *
from preprocessor import *


sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import ontology_reader, dataset_walker

import CRFPP


class SemanticTagger(object):
	MY_ID = 'SemanticTagger'
	def __init__(self, model):
		self.appLogger = logging.getLogger(self.MY_ID)
		self.tagger = CRFPP.Tagger("-m %s -v 3" %(model))
		self.tokenizer = tokenizer()
		self.stemmer = stemmer()

	def ExtractFeature(self, word_tokens):
		word_tags = nltk.pos_tag(word_tokens)
		lemma_tokens = []
		POS_tags = []
		First_Upper = []
		for word, tag in word_tags:
			lemma_tokens.append(self.stemmer.stem(word.lower()))
			POS_tags.append(tag)
			if word.istitle():
				First_Upper.append('Y')
			else:
				First_Upper.append('N')
		return zip(word_tokens, lemma_tokens, POS_tags, First_Upper)


	def ParseSent(self, sent):
		semtags = []

		sent = sent.strip()
		ori_tokens = sent.split(' ')
		ori_tokens = [tk for tk in ori_tokens if len(tk) > 0]
		tok_tokens = self.tokenizer.tokenize(sent)
		self.appLogger.debug('ori_tokens: %s' %(' '.join(ori_tokens)))
		self.appLogger.debug('tok_tokens: %s' %(' '.join(tok_tokens)))

		tokens_map = SemanticTagger._map_tokens(ori_tokens, tok_tokens)
		if not tokens_map:
			return semtags


		self.tagger.clear()
		for feature in self.ExtractFeature(tok_tokens):
			self.tagger.add(' '.join(feature).encode('utf-8'))
		self.tagger.parse()

		tok_tags = []
		for i in range(self.tagger.size()):
			tok_tags.append(self.tagger.y2(i))

		ori_tags = self._get_ori_tags(tokens_map, tok_tags)

		self.appLogger.debug('ori_tags: %s' %(' '.join(ori_tags)))
		self.appLogger.debug('tok_tags: %s' %(' '.join(tok_tags)))

		
		
		cur_tag = {}
		cur_tag_flag = False
		for word, stag in zip(ori_tokens, ori_tags):
			if stag[-1] == 'B':
				if cur_tag_flag:
					semtags.append(cur_tag)
					cur_tag_flag = False
				cur_tag_flag = True
				cur_tag = {}
				cur_tag['tag'] = stag[0:-2]
				cur_tag['content'] = [word]
			elif stag[-1] == 'I':
				if cur_tag_flag:
					if stag[0:-2] == cur_tag['tag']:
						cur_tag['content'].append(word)
						continue

					semtags.append(cur_tag)
					cur_tag_flag = False

				cur_tag_flag = True
				cur_tag = {}
				cur_tag['tag'] = stag[0:-2]
				cur_tag['content'] = [word]

			else:
				if cur_tag_flag:
					semtags.append(cur_tag)
					cur_tag_flag = False

		self.appLogger.debug('semtags: %s' %(semtags.__str__()))
		return semtags

	@staticmethod
	def _get_ori_tags(tokens_map, tok_tags):
		ori_tags = []
		cur_pos = -1
		for i, tk_map in enumerate(tokens_map):
			if tk_map != cur_pos:
				assert tk_map == len(ori_tags)
				ori_tags.append(tok_tags[i])
				cur_pos = tk_map
		return ori_tags

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





class association_rule_tracker(object):
	MY_ID = 'association_rule_tracker'
	def __init__(self, tagsets, association_rule_file, semtagger_model, prob_threshold = 0.8, mode = 'exact', bs_mode = 'max', bs_alpha = 0.0, unified_thres=0.5):
		self.appLogger = logging.getLogger(self.MY_ID)		
		self.tagsets = tagsets

		self.unified_thres = unified_thres
		self.prob_threshold = prob_threshold
		self.prob_thres_factor = math.log(self.unified_thres, self.prob_threshold)

		input = codecs.open(association_rule_file, 'r', 'utf-8')
		self.association_rules = json.load(input)
		input.close()
		self.semtagger = SemanticTagger(semtagger_model)
		self.mode = mode

		self.beliefstate = BeliefState(bs_mode,bs_alpha)
		self.rules = DSTC4_rules(tagsets)
		self.reset()
		


	def addUtter(self, utter):
		self.appLogger.debug('utter_index: %d' % (utter['utter_index']))
		output = {'utter_index': utter['utter_index']}
		topic = utter['segment_info']['topic']
		if topic in self.tagsets:
			self._UpdateFrameProb(utter)
			self._UpdateFrame()
			self.frame = self.rules.prune_frame_label(topic, self.frame)
			output['frame_prob'] = copy.deepcopy(self.beliefstate.state)
			output['frame_label'] = copy.deepcopy(self.frame)
		return output

	def _get_prob(self, prob, occurrence_times, alpha = 2):
		return prob * (1.0 - 1.0 / (1 + occurrence_times**alpha) )

	def _get_frame(self, semtag, topic):
		candidate_frames = []
		semtag_key = 'SemTag_%s(%s)' %(semtag['tag'], '_'.join(semtag['content']))

		target_rules = self.association_rules[topic]

		if self.mode == 'exact':
			if semtag_key in target_rules['v']['rules']:
				for frame_key, frame_value in target_rules['v']['rules'][semtag_key].items():
					if frame_key.startswith('Frame'):
						slot = frame_key[frame_key.find('_') + 1: frame_key.find('(')]
						value = ' '.join(frame_key[frame_key.find('(') + 1 : frame_key.find(')')].split('_'))
						prob = self._get_prob(frame_value['prob'], frame_value['occurrence'])
						candidate_frames.append((slot,value,prob))
		elif self.mode == 'fuzzy':
			for key, frame_candidates in target_rules['v']['rules'].items():
				ratio = fuzz.ratio(semtag_key[7:], key[7:]) * 1.0 / 100

				for frame_key, frame_value in frame_candidates.items():
					if frame_key.startswith('Frame'):
						slot = frame_key[frame_key.find('_') + 1: frame_key.find('(')]
						value = ' '.join(frame_key[frame_key.find('(') + 1 : frame_key.find(')')].split('_'))
						prob = self._get_prob(frame_value['prob'], frame_value['occurrence']) * ratio
						candidate_frames.append((slot,value,prob))

		else:
			raise Exception('Error: unknown mode %s!' %(mode))

		return candidate_frames
	
	def _UpdateFrameProb(self, utter):
		topic = utter['segment_info']['topic']
		if utter['segment_info']['target_bio'] == 'B':
			self.frame = {}
			#self.frame_prob = {}
			self.beliefstate.reset()
		
		if topic in self.tagsets:
			transcript = utter['transcript']
			semtags = self.semtagger.ParseSent(transcript)
			for semtag in semtags:
				candidate_frames = self._get_frame(semtag, topic)
				for (slot, value, prob) in candidate_frames:
					#self._AddSLot2FrameProb(slot, -1)
					#self._AddSlotValue2FrameProb(slot,value,prob)
					self.beliefstate._AddSLot2State(slot, -1)
					normalised_prob = math.pow(prob, self.prob_thres_factor)
					self.beliefstate._AddSlotValue2State(slot,value,normalised_prob)



	def _UpdateFrame(self):
		self.frame = {}
		for slot in self.beliefstate.state:
			for value, prob in self.beliefstate.state[slot]['values'].items():
				if prob >= self.unified_thres:
					self._AddSlotValue2Frame(slot,value)
			

	def _AddSlotValue2Frame(self,slot,value):
		if slot not in self.frame:
			self.frame[slot] = []
		if value not in self.frame[slot]:
			self.frame[slot].append(value)





	def reset(self):
		self.frame = {}
		#self.frame_prob = {}
		self.memory = {}
		self.beliefstate.reset()



def main(argv):
	parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('--ar',dest='association_rules',action='store',required=True,metavar='JSON_FILE', help='association_rules')
	parser.add_argument('--stm',dest='semantic_tagger_model',action='store',required=True, help='semantic_tagger_model')
	parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
	parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')
	parser.add_argument('--pt',dest='prob_threshold',type=float,action='store',default=0.8,help='prob_threshold')
	parser.add_argument('--exact',dest='exact',action='store_true',help='exact mode of fuzz mode')
	args = parser.parse_args()


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



	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=False)
	tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()

	track_file = open(args.trackfile, "wb")
	track = {"sessions":[]}
	track["dataset"]  = args.dataset
	start_time = time.time()

	if args.exact:
		mode = 'exact'
	else:
		mode = 'fuzzy'
	tracker = association_rule_tracker(tagsets, args.association_rules, args.semantic_tagger_model, args.prob_threshold, mode)
	for call in dataset:
		this_session = {"session_id":call.log["session_id"], "utterances":[]}
		tracker.reset()
		for (utter,_) in call:
			sys.stderr.write('%d:%d\n'%(call.log['session_id'], utter['utter_index']))
			tracker_result = tracker.addUtter(utter)
			if tracker_result is not None:
				this_session["utterances"].append(tracker_result)
		track["sessions"].append(this_session)
	end_time = time.time()
	elapsed_time = end_time - start_time
	track['wall_time'] = elapsed_time

	json.dump(track, track_file, indent=4)

	track_file.close()

if __name__ =="__main__":
	main(sys.argv)