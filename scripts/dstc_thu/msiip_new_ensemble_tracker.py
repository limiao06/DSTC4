#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
The ensemble tracker
Combine multiple trackers output a result 
'''


import argparse, sys, time, json, os, copy, codecs

import logging
from GlobalConfig import *


from BeliefState import BeliefState
from dstc4_rules import DSTC4_rules
from Utils import *

#from msiip_nsvc_tracker import msiip_nsvc_tracker
#from association_rule_tracker import association_rule_tracker

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import ontology_reader, dataset_walker


'''
采用后融合的方式，以其他输出的结果作为输入
'''

class msiip_ensemble_tracker(object):
	MY_ID = 'msiip_ensemble'
	def __init__(self, tagsets, dataset, base_dir, config_file, \
				weight_key='f1', unified_threshold=0.5, slot_specific=True):
		'''
		weight_key can be 'precision', 'recall' or 'f1'
		config_file has many lines
		each line indicates a tracker out json file and its performance file on dev set
		'''

		self.config = GetConfig()
		self.appLogger = logging.getLogger(self.MY_ID)
		self.rules = DSTC4_rules(tagsets)
		self.unified_threshold = unified_threshold
		self.tagsets = tagsets

		self.logs, self.perform_list = self._read_config(config_file, base_dir)

		# calc weight list
		self.weight_key = weight_key
		self.weight_dict = {}
		for topic in self.tagsets:
			self.weight_dict[topic] = {}
			for slot in self.tagsets[topic]:
				temp_weight = []
				for i, perform_dict in enumerate(self.perform_list):
					if topic in perform_dict and slot in perform_dict[topic]:
						value = perform_dict[topic][slot][self.weight_key]
						if value != 'NULL':
							temp_weight.append(value)
						else:
							temp_weight.append(0.0)
					else:
						temp_weight.append(0.0)
				if sum(temp_weight) == 0.0:
					temp_weight = []
					for perform_dict in self.perform_list:
						value = perform_dict[topic]['all'][self.weight_key]
						if value != 'NULL':
							temp_weight.append(value)
						else:
							temp_weight.append(0.0)
				if sum(temp_weight) == 0.0:
					temp_weight = []
					for perform_dict in self.perform_list:
						value = perform_dict['all']['all'][self.weight_key]
						if value != 'NULL':
							temp_weight.append(value)
						else:
							temp_weight.append(0.0)
				sum_weight = sum(temp_weight)
				self.weight_dict[topic][slot] = [value / sum_weight for value in temp_weight]
		temp_weight=[]
		for perform_dict in self.perform_list:
			value = perform_dict['all']['all'][self.weight_key]
			if value != 'NULL':
				temp_weight.append(value)
			else:
				temp_weight.append(0.0)
		sum_weight = sum(temp_weight)
		self.default_weight = [value / sum_weight for value in temp_weight]


		self.topic_dic = {}
		for call in dataset:
			session_id = call.log["session_id"]
			for (utter,_) in call:
				utter_index = utter['utter_index']
				self.topic_dic[(session_id, utter_index)] = utter["segment_info"]['topic']

		for i in range(len(self.logs)-1):
			if self.logs[i]['dataset'] != self.logs[i+1]['dataset']:
				raise Exception('Dataset of input logs mismatch!')

		self.dataset = self.logs[0]['dataset']
		self.appLogger.info('dataset: %s' %(self.dataset))
		self.frame_prob = {}
		self.frame = {}

	def _read_config(self, config_file, base_dir):
		'''
		config file indicates which systems to ensemble,
		each line has two parts, 
			first part is a json output file from the system
			second part is a score file on dev set of the system
		'''
		logs = []
		perform_list = []
		input = file(config_file)
		lines = input.readlines()
		input.close()
		for line in lines:
			l = line.strip()
			if not l:
				break
			tokens = l.split('\t')
			assert len(tokens) == 2
			filename = os.path.join(base_dir, tokens[0])
			log_input = codecs.open(filename, 'r', 'utf-8')
			logs.append(json.load(log_input))
			log_input.close()

			perform_file_name = os.path.join(base_dir, tokens[1])
			perform_list.append(self._read_score_file(perform_file_name))
		return logs, perform_list

	def _read_score_file(self, score_file):
		input=file(score_file)
		score_dict = {}
		line = input.readline()
		if not line.strip() == 'topic, slot, schedule, stat, N, result':
			raise Exception('Error: the input file is not a dstc score file!')
		while True:
			line = input.readline()
			if not line:
				break
			tokens = [t.strip() for t in line.split(',')]
			if len(tokens) != 6:
				continue
			topic = tokens[0]
			slot = tokens[1]
			if topic not in score_dict:
				score_dict[topic] = {}
			if topic == 'basic':
				score_dict[topic][slot] = tokens[-1]
			else:
				schedule_id = int(tokens[2])
				stat_term = tokens[3]
				value = tokens[-1]
				if schedule_id == 2:
					if value == '-':
						value = 'NULL'
					else:
						value = float(value)
					if slot not in score_dict[topic]:
						score_dict[topic][slot] = {}
					if stat_term not in score_dict[topic][slot]:
						score_dict[topic][slot][stat_term] = value
		input.close()
		return score_dict

	def ensemble(self):
		start_time = time.time()
		out_json = {"sessions":[]}
		out_json['dataset'] = self.dataset
		out_json['wall_time'] = max([log['wall_time'] for log in self.logs])
		for i, session in enumerate(self.logs[0]['sessions']):
			this_session = {"session_id":session["session_id"], "utterances":[]}
			for j, utter in enumerate(session['utterances']):
				sys.stderr.write('%d:%d\n'%(session['session_id'], utter['utter_index']))
				this_utter = {'utter_index': utter['utter_index']}
				if 'frame_prob' in utter:
					topic = self.topic_dic[(session['session_id'],utter['utter_index'])]
					frame_prob_list = []
					for log in self.logs:
						frame_prob_list.append(log['sessions'][i]['utterances'][j]['frame_prob'])
					self._UpdateFrameProb(frame_prob_list, topic)
					self._UpdateFrame()
					self.frame = self.rules.prune_frame_label(topic, self.frame)
					this_utter['frame_prob'] = copy.deepcopy(self.frame_prob)
					this_utter['frame_label'] = copy.deepcopy(self.frame)
				this_session['utterances'].append(this_utter)
			out_json["sessions"].append(this_session)
		end_time = time.time()
		elapsed_time = end_time - start_time
		out_json['wall_time'] += elapsed_time
		return out_json

	def _UpdateFrameProb(self, frame_prob_list, topic):
		out_state = {}
		# prepare
		for bs in frame_prob_list:
			for slot in bs:
				if slot not in out_state:
					out_state[slot] = {'prob': -1, 'values':{}}
				for value in bs[slot]['values']:
					if value not in out_state[slot]['values']:
						out_state[slot]['values'][value] = 0.0
		# ensemble
		for slot in out_state:
			# calculate slot prob, slot prob is the max value
			max_prob = -1
			for bs in frame_prob_list:
				if slot in bs:
					if max_prob < bs[slot]['prob']:
						max_prob = bs[slot]['prob']
			out_state[slot]['prob'] = max_prob

			# calculate value prob
			for value in out_state[slot]['values']:
				value_prob_list = []
				for bs in frame_prob_list:
					if slot not in bs:
						value_prob_list.append(0.0)
					else:
						if value not in bs[slot]['values']:
							value_prob_list.append(0.0)
						else:
							value_prob_list.append(bs[slot]['values'][value])
				if slot_specific and topic in self.weight_dict and slot in self.weight_dict[topic]:
					weight_vector = self.weight_dict[topic][slot]
				else:
					weight_vector = self.default_weight
				for weight, prob in zip(weight_vector, value_prob_list):
					out_state[slot]['values'][value] += weight * prob

		self.frame_prob = out_state

	def _UpdateFrame(self):
		self.frame = {}
		for slot in self.frame_prob:
			if self.frame_prob[slot]['prob'] == -1 or self.frame_prob[slot]['prob'] > self.unified_threshold:
				for value, prob in self.frame_prob[slot]['values'].items():
					if prob >= self.unified_threshold:
						self._AddSlotValue2Frame(slot,value)

	def _AddSlotValue2Frame(self,slot,value):
		if slot not in self.frame:
			self.frame[slot] = []
		if value not in self.frame[slot]:
			self.frame[slot].append(value)









def main(argv):
	parser = argparse.ArgumentParser(description='MSIIP ensemble tracker.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('--LogBaseDir',dest='LogBaseDir',action='store',required=True,help='The base directory that contains the log files')
	parser.add_argument('--config',dest='config',action='store',required=True,help='Config file, indicate log files and weight')
	parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
	parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')
	parser.add_argument('--unified_thres',dest='unified_thres',type=float,action='store',default=0.5,help='output value prob threshold')
	parser.add_argument('--weight_key',dest='weight_key',action='store',default='f1',help='key to calc weight')
	parser.add_argument('--SA',dest='score_averaging',action='store_true',help='use conventional score averaging or not.')
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

	tracker = msiip_ensemble_tracker(tagsets, dataset, args.LogBaseDir, args.config, args.weight_key, args.unified_thres, not args.score_averaging)
	track = tracker.ensemble()

	track_file = open(args.trackfile, "wb")
	json.dump(track, track_file, indent=4)
	track_file.close()

if __name__ =="__main__":
	main(sys.argv)
