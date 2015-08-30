#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
The tracker

baseline tracker,
output both frame_prob and frame
'''


import argparse, sys, time, json, os, copy
import logging
from GlobalConfig import *
import math
from fuzzywuzzy import fuzz

from BeliefState import BeliefState
from dstc4_rules import DSTC4_rules
from Utils import *

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import ontology_reader, dataset_walker




class msiip_nsvc_tracker(object):
	MY_ID = 'msiip_baseline'
	def __init__(self, tagsets, ratio_thres = 0.85, \
				bs_mode = 'enhance', bs_alpha = 0.0, \
				unified_thres = 0.5):
		self.tagsets = tagsets
		self.frame = {}
		self.memory = {}
		self.beliefstate = BeliefState(bs_mode, bs_alpha)

		self.ratio_thres = ratio_thres
		self.unified_thres = unified_thres

		self.ratio_thres_factor = math.log(self.unified_thres, self.ratio_thres)

		self.rules = DSTC4_rules(tagsets)
		self.appLogger = logging.getLogger(self.MY_ID)


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

	def _UpdateFrameProb(self, utter):
		topic = utter['segment_info']['topic']
		if utter['segment_info']['target_bio'] == 'B':
			self.frame = {}
			self.beliefstate.reset()
		
		if topic in self.tagsets:
			transcript = utter['transcript'].replace('Singapore', '')
			prob_frame_labels = {}
			for slot in self.tagsets[topic]:
				for value in self.tagsets[topic][slot]:
					ratio = fuzz.partial_ratio(value.lower(), transcript.lower()) * 1.0 / 100
					if ratio > self.ratio_thres:
						if slot not in prob_frame_labels:
							prob_frame_labels[slot] = {'prob':-1, 'values':{}}
						if value not in prob_frame_labels[slot]['values'] or ratio > prob_frame_labels[slot]['values'][value]:
							prob_frame_labels[slot]['values'][value] = ratio
			
			# normalise prob so as 0.5 is the threshold
			self._NormalisProb(prob_frame_labels)
			# update belief state
			self.beliefstate.AddFrame(prob_frame_labels)

	def _NormalisProb(self, prob_frame_labels):
		'''
		based on the threshold, normalise threshold to 0.5
		'''
		for slot in prob_frame_labels:
			for value, ratio in prob_frame_labels[slot]['values'].items():
				prob_frame_labels[slot]['values'][value] = math.pow(ratio, self.ratio_thres_factor)

	def _UpdateFrame(self):
		self.frame = {}
		for slot in self.beliefstate.state:
			for value, ratio in self.beliefstate.state[slot]['values'].items():
				if ratio >= self.unified_thres:
					self._AddSlotValue2Frame(slot,value)

	def _AddSlotValue2Frame(self,slot,value):
		if slot not in self.frame:
			self.frame[slot] = []
		if value not in self.frame[slot]:
			self.frame[slot].append(value)





	def reset(self):
		self.frame = {}
		self.memory = {}
		self.beliefstate.reset()



def main(argv):
	parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
	parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')
	parser.add_argument('--ratio_thres',dest='ratio_thres',type=float,action='store',default=0.85,help='ration threshold')
	parser.add_argument('--BSMode',dest='BSMode',action='store',default='enhance',help='Belief State mode: max, average or enhance')
	parser.add_argument('--BSAlpha',dest='BSAlpha',type=float,action='store',default=0.0,help='Belief State average history alpha')
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

	tracker = msiip_nsvc_tracker(tagsets,
								ratio_thres = args.ratio_thres, 
								bs_mode = args.BSMode, 
								bs_alpha = args.BSAlpha)
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
