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
	def __init__(self, tagsets, dataset, base_dir, config_file, slot_prob_threshold, value_prob_threshold):
		'''
		config_file has many lines
		each line indicates a tracker out json file and its performance
		'''
		input = file(config_file)
		lines = input.readlines()
		input.close()

		self.config = GetConfig()
		self.appLogger = logging.getLogger(self.MY_ID)
		self.rules = DSTC4_rules(tagsets)
		self.slot_prob_threshold = slot_prob_threshold
		self.value_prob_threshold = value_prob_threshold

		self.logs = []
		self.weight_list = []


		self.topic_dic = {}
		for call in dataset:
			session_id = call.log["session_id"]
			for (utter,_) in call:
				utter_index = utter['utter_index']
				self.topic_dic[(session_id, utter_index)] = utter["segment_info"]['topic']

		for line in lines:
			l = line.strip()
			if not l:
				break
			tokens = l.split('\t')
			assert len(tokens) == 2
			filename = os.path.join(base_dir, tokens[0])
			log_input = codecs.open(filename, 'r', 'utf-8')
			self.logs.append(json.load(log_input))
			log_input.close()
			weight = float(tokens[1])
			self.weight_list.append(weight)

			self.appLogger.info('%s\t%.3f' %(filename, weight))

		for i in range(len(self.logs)-1):
			if self.logs[i]['dataset'] != self.logs[i+1]['dataset']:
				raise Exception('Dataset of input logs mismatch!')

		self.dataset = self.logs[0]['dataset']
		self.appLogger.info('dataset: %s' %(self.dataset))
		self.frame_prob = {}
		self.frame = {}

	def ensemble(self):
		out_json = {"sessions":[]}
		out_json['dataset'] = self.dataset
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
					self._UpdateFrameProb(frame_prob_list, self.weight_list)
					self._UpdateFrame()
					self.frame = self.rules.prune_frame_label(topic, self.frame)
					this_utter['frame_prob'] = copy.deepcopy(self.frame_prob)
					this_utter['frame_label'] = copy.deepcopy(self.frame)
				this_session['utterances'].append(this_utter)
			out_json["sessions"].append(this_session)
		return out_json

	def _UpdateFrameProb(self, frame_prob_list, weight_list):
		self.frame_prob = BeliefState.StateEnsemble(frame_prob_list, weight_list)
	
	def _UpdateFrame(self):
		self.frame = {}
		for slot in self.frame_prob:
			if self.frame_prob[slot]['prob'] == -1 or self.frame_prob[slot]['prob'] > self.slot_prob_threshold:
				for value, prob in self.frame_prob[slot]['values'].items():
					if prob >= self.value_prob_threshold:
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
	parser.add_argument('--value_prob',dest='value_prob',type=float,action='store',default=0.8,help='output value prob threshold')
	parser.add_argument('--slot_prob',dest='slot_prob',type=float,action='store',default=0.6,help='output slot prob threshold')
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

	tracker = msiip_ensemble_tracker(tagsets, dataset, args.LogBaseDir, args.config, args.slot_prob, args.value_prob)
	track = tracker.ensemble()
	json.dump(track, track_file, indent=4)
	track_file.close()

if __name__ =="__main__":
	main(sys.argv)
