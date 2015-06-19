import argparse, sys, ontology_reader, dataset_walker, time, json
from fuzzywuzzy import fuzz

class conversation_extractor(object):
	def __init__(self):
		self.reset()

	def addUtter(self, utter):
		print '%d\t%s:\t%s' %(utter['utter_index'], utter['speaker'], utter['transcript'])

	def reset(self):
		self.frame = {}

def main(argv):
	parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')

	args = parser.parse_args()
	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=False)

	start_time = time.time()

	tracker = conversation_extractor()
	for call in dataset:
		this_session = {"session_id":call.log["session_id"], "utterances":[]}
		tracker.reset()
		for (utter,_) in call:
			sys.stderr.write('%d:%d\n'%(call.log['session_id'], utter['utter_index']))
			tracker.addUtter(utter)
	end_time = time.time()
	elapsed_time = end_time - start_time

if __name__ =="__main__":
	main(sys.argv)
