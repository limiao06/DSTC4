#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
The tracker

it first use new slot_value_classifier to extract which slot may appear in the sub_segment

then use value_extractor to extract value for slots that appear in the sub_segment
'''


import argparse, sys, time, json, os, copy
import logging
from GlobalConfig import *

from New_Slot_value_classifier import slot_value_classifier
from value_extractor import value_extractor
from BeliefState import BeliefState
from dstc4_rules import DSTC4_rules
from Utils import *

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import ontology_reader, dataset_walker




class msiip_nsvc_tracker(object):
	MY_ID = 'msiip_nsvc'
	def __init__(self, tagsets, model_dir, ratio_thres = 0.8, max_num = 2, slot_prob_thres = 0.5, value_prob_thres = 0.8, mode = 'hr', bs_mode = 'max', bs_alpha = 0.0):
		self.tagsets = tagsets
		self.frame = {}
		self.memory = {}
		self.beliefstate = BeliefState(bs_mode, bs_alpha)

		self.slot_prob_threshold = slot_prob_thres
		self.ratio_thres = ratio_thres
		self.mode = mode

		self.svc = slot_value_classifier()
		self.svc.LoadModel(model_dir)

		self.tuple_extractor = Tuple_Extractor()
		self.rules = DSTC4_rules(tagsets)
		self.appLogger = logging.getLogger(self.MY_ID)

		if not self.svc.is_set:
			self.appLogger.error('Error: Fail to load slot_value_classifier model!')
			raise Exception('Error: Fail to load slot_value_classifier model!')
		self.value_extractor = value_extractor(tagsets, ratio_thres, max_num)


	def addUtter(self, utter):
		self.appLogger.debug('utter_index: %d' % (utter['utter_index']))
		output = {'utter_index': utter['utter_index']}
		topic = utter['segment_info']['topic']
		if topic in self.tagsets:
			self._UpdateFrameProb(utter)
			self._UpdateFrame()
			self.frame = self.rules.prune_frame_label(topic, self.frame)
			output['frame_prob'] = copy.deepcopy(self.frame_prob)
			output['frame_label'] = copy.deepcopy(self.frame)
		return output

	def _UpdateFrameProb(self, utter):
		topic = utter['segment_info']['topic']
		if utter['segment_info']['target_bio'] == 'B':
			self.frame = {}
			self.frame_prob = {}
			
		if topic in self.tagsets:
			transcript = utter['transcript']
			svc_result, result_prob = self.svc.PredictUtter(utter, self.svc.feature.feature_list)
			tuples = []
			probs = []
			for key in svc_result:
				label = svc_result[key]
				prob = result_prob[key][1]
				if label == 1:
					tuples.append(key)
					probs.append(prob)

			self.appLogger.debug('transcript: %s' %(transcript))
			self.appLogger.debug('tuples: %s, probs: %s' %(tuples.__str__(), probs.__str__()))

			# get the prob_frame_labels for the single utterance
			prob_frame_labels = self.tuple_extractor.generate_frame(tuples, probs, self.mode)
			self._AddExtractValue(topic, prob_frame_labels)

			# update belief state
			self.state.AddFrame(prob_frame_labels)


	def _AddExtractValue(self, topic, prob_frame_labels):
		'''
		extract values for non-enumerable slots
		'''
		for slot in prob_frame_labels:
			if slot in self.tagsets[topic]:
				if not self.tuple_extractor.enumerable(slot):
					value_list = self.value_extractor.ExtractValue(topic, slot, transcript)
					for value, ratio in value_list:
						prob_frame_labels[slot]['values'][value] = ratio * 1.0 / 100

	def _UpdateFrame(self):
		self.frame = {}
		for slot in self.state:
			if self.tuple_extractor.enumerable(slot):
				for value, prob in self.frame_prob[slot]['values'].items():
					if prob >= self.slot_prob_threshold:
						self._AddSlotValue2Frame(slot,value)
			else:
				if self.frame_prob[slot]['prob'] > self.slot_prob_threshold:
					for value, ratio in self.frame_prob[slot]['values'].items():
						if ratio >= self.ratio_thres:
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
	parser.add_argument('--model_dir',dest='model_dir',action='store',required=True,metavar='PATH', help='model dir')
	parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
	parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')
	parser.add_argument('--ratio_thres',dest='ratio_thres',type=float,action='store',default=0.8,help='ration threshold')
	
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



	args = parser.parse_args()
	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=False)
	tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()

	track_file = open(args.trackfile, "wb")
	track = {"sessions":[]}
	track["dataset"]  = args.dataset
	start_time = time.time()

	tracker = msiip_nsvc_tracker(tagsets, args.model_dir, ratio_thres=args.ratio_thres)
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
