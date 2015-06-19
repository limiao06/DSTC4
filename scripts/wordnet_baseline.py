import argparse, sys, ontology_reader, dataset_walker, time, json
from fuzzywuzzy import fuzz
from nltk.corpus import wordnet as wn

def wn_find_synset(word, pos = None):
	w_synset = [word]
	for synset in wn.synsets(word, pos=pos):
		for item in synset.lemma_names:
			if item not in w_synset:
				w_synset.append(item)
	return w_synset


class WN_BaselineTracker(object):
	def __init__(self, tagsets, threshold = 80):
		self.tagsets = tagsets
		self.wn_tagsets = {}
		self.threshold = threshold
		self.frame = {}
		self.memory = {}
		self.init()
		self.reset()
		print 'threshold: %d' %(self.threshold)

	def addUtter(self, utter):
		output = {'utter_index': utter['utter_index']}

		topic = utter['segment_info']['topic']
		transcript = utter['transcript'].replace('Singapore', '')

		if utter['segment_info']['target_bio'] == 'B':
			self.frame = {}
			
		if topic in self.wn_tagsets:
			for slot in self.wn_tagsets[topic]:
				# print 'slot: %s' %(slot)
				for value, value_synsets in self.wn_tagsets[topic][slot].items():
					# print 'value: %s' %(value)
					for item in value_synsets:
						ratio = fuzz.partial_ratio(item, transcript)
						# print 'Compare: %s, %s, score is %d' %(item, transcript, ratio)
						if ratio > self.threshold:
							if slot not in self.frame:
								self.frame[slot] = []
							if value not in self.frame[slot]:
								self.frame[slot].append(value)
							break
			if topic == 'ATTRACTION' and 'PLACE' in self.frame and 'NEIGHBOURHOOD' in self.frame and self.frame['PLACE'] == self.frame['NEIGHBOURHOOD']:
				del self.frame['PLACE']

			output['frame_label'] = self.frame
		return output

	def reset(self):
		self.frame = {}

	def init(self):
		'''
		build wn_tagsets
		'''
		for topic in self.tagsets:
			self.wn_tagsets[topic] = {}
			for slot,values in self.tagsets[topic].items():
				self.wn_tagsets[topic][slot] = {}
				for value in values:
					self.wn_tagsets[topic][slot][value] = wn_find_synset(value)

		return



def main(argv):
	parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
	parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')
	parser.add_argument('-t',dest='threshold',action='store',type=int,default=80,help='Threshold of partial_ratio')

	args = parser.parse_args()
	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=False)
	tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()

	track_file = open(args.trackfile, "wb")
	track = {"sessions":[]}
	track["dataset"]  = args.dataset
	start_time = time.time()

	tracker = WN_BaselineTracker(tagsets, args.threshold)
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
