import argparse, sys, ontology_reader, dataset_walker, time, json
from collections import defaultdict


class sub_segment_extractor(object):
	def __init__(self):
		self.reset()

	def addUtter(self, log_utter, label_utter):
		if log_utter['segment_info']['target_bio'] == 'B':
			self.reset()
			self.state['topic'] = log_utter['segment_info']['topic']
			self.state['frame_label'] = label_utter['frame_label']
			self.state['guide_act'] = log_utter['segment_info']['guide_act']
			self.state['tourist_act'] = log_utter['segment_info']['tourist_act']
			self.state['initiativity'] = log_utter['segment_info']['initiativity']
			self.is_empty = False
		
		if log_utter['segment_info']['target_bio'] != 'O':
			self.state['utter_ids'].append(log_utter['utter_index'])
			self.state['utter_sents'].append('%s: %s' %(log_utter['speaker'],log_utter['transcript']))
			self.state['semantic_tags'].append(label_utter['semantic_tagged'])
			self.state['speech_acts'].append(label_utter['speech_act'])
			assert self.state['frame_label'] == label_utter['frame_label']

	def get_sub_seg_infos():
		if self.is_empty:
			return {}
		else:
			return self.state

	def reset(self):
		self.is_empty = True
		self.state = {}
		self.state['topic'] = ''
		self.state['guide_act'] = ''
		self.state['tourist_act'] = ''
		self.state['initiativity'] = ''
		self.state['utter_ids'] = []
		self.state['utter_sents'] = []
		self.state['semantic_tags'] = []
		self.state['speech_acts'] = []
		self.state['frame_label'] = {}

def main(argv):
	parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
	parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')

	args = parser.parse_args()
	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)
	tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()

	track_file = open(args.trackfile, "wb")
	track = {"sessions":[]}
	track["dataset"]  = args.dataset
	start_time = time.time()

	sub_seg_counter = 0
	topic_slot_counter = {}
	for topic in tagsets:
		topic_slot_counter[topic] = defaultdict(int)

	extractor = sub_segment_extractor()
	for call in dataset:
		this_session = {"session_id":call.log["session_id"], "sub_segments":[]}
		extractor.reset()
		for (log_utter, label_utter) in call:
			sys.stderr.write('%d:%d\n'%(call.log['session_id'], log_utter['utter_index']))
			if log_utter['segment_info']['target_bio'] == 'B':
				if not extractor.is_empty:
					sub_segment = extractor.state
					sub_segment['id'] = sub_seg_counter
					sub_seg_counter += 1
					this_session['sub_segments'].append(sub_segment)
					for slot in sub_segment['frame_label']:
						topic_slot_counter[sub_segment['topic']][slot] += 1

			extractor.addUtter(log_utter,label_utter)
		track["sessions"].append(this_session)
	end_time = time.time()
	elapsed_time = end_time - start_time
	track['wall_time'] = elapsed_time

	json.dump(track, track_file, indent=4)

	track_file.close()

	print json.dumps(topic_slot_counter, indent = 4)

if __name__ =="__main__":
	main(sys.argv)
