#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
extract sub utterances for three tasks
1. utterances split (utterances -> sub utterances)
2. speech act prediction
3. semantic tag
'''

import argparse, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../scripts/'))

import ontology_reader, dataset_walker, time, json
import re


class sub_utters_extractor(object):
	def __init__(self):
		self.sem_tag_regex = re.compile('<[A-Z \"\-\=]+>(.*?)<\/[A-Z]+>')

	def addUtter(self, log_utter, label_utter):
		# read sub utters
		transcript = log_utter['transcript']
		speech_acts = label_utter['speech_act']
		semantic_tags = label_utter['semantic_tagged']
		if len(speech_acts) != len(semantic_tags):
			print 'Error: sub utters not same in speech_acts and semantic_tags! utter_index: %d' %(log_utter['utter_index'])
			return (None,None,None,None)
		# check if there are some errors in the semantic tags
		sub_utters_list = []
		sub_tag_list = []
		transcript = transcript.strip()
		transcript = transcript.replace('% ','%')
		transcript = transcript.replace('  ',' ')
		for semantic_tag in semantic_tags:

			semantic_tag = semantic_tag.strip()
			semantic_tag = semantic_tag.replace('% ','%')
			semantic_tag = semantic_tag.replace('  ',' ')
			sub_tag_list.append(semantic_tag)

			while self.sem_tag_regex.search(semantic_tag):
				semantic_tag = self.sem_tag_regex.sub('\\1', semantic_tag)
			
			sub_utters_list.append(semantic_tag)
		semantic_tags_string = ' '.join(sub_utters_list)
		if semantic_tags_string.strip() != transcript.strip():
			print 'Warning! semantic_tags_string and transcript mismatch! \ntranscript: %s\nsemantic_tags_string: %s' %(transcript,semantic_tags_string)
			return (None,None,None,None)
		else:
			return (transcript, sub_utters_list, sub_tag_list, speech_acts)



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
	out_json = {'dataset':args.dataset, 'utterances':[]}
	extractor = sub_utters_extractor()
	for call in dataset:
		for (log_utter, label_utter) in call:
			sys.stderr.write('%d:%d\n'%(call.log['session_id'], log_utter['utter_index']))
			print '%d:%d'%(call.log['session_id'], log_utter['utter_index'])
			(transcript, sub_utters_list, sub_tag_list, speech_acts) = extractor.addUtter(log_utter,label_utter)
			if transcript:
				item = {}
				item['transcript'] = transcript
				item['sub_utters_list'] = sub_utters_list
				item['sub_tag_list'] = sub_tag_list
				item['speech_acts'] = speech_acts
				out_json['utterances'].append(item)
	json.dump(out_json, track_file, indent=4)

	#track_file.close()
if __name__ =="__main__":
	main(sys.argv)
