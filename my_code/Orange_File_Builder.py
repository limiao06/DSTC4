#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
input sub_segments file,
for each topic
Extract the coocurrence of semantic_tags and frame_labels
output a orange test_market.tab file format

__basket_foo
basket

Bread Milk
Bread Diapers Beer Eggs
Milk Diapers Beer Cola
Bread Milk Diapers Beer
Bread Milk Diapers Cola
'''

import argparse, sys, time, json, os
import codecs
import re
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__),'../scripts'))
from temp_stat_data import extract_semantic_tags


def SemTagToString(sem_tag, out_slot = False, out_value = False):
	outstr = 'SemTag_'
	outstr += sem_tag['name']

	if out_slot:
		slot_value_list = []
		for slot,value in sem_tag['slots'].items():
			if value != 'NONE':
				slot_value_list.append('%s_%s' %(slot,value)) 
		outstr += '[%s]' %(','.join(slot_value_list))

	if out_value:
		outstr += '(%s)' %('_'.join(sem_tag['value'].split()))

	return outstr

def FrameTagToString(slot,value,out_value = False):
	outstr = 'Frame_%s' %(slot)
	if slot == 'INFO' or out_value:
		outstr += '(%s)' %('_'.join(value.split()))
	return outstr





def main(argv):
	parser = argparse.ArgumentParser(description='Extract the coocurrence of semantic_tags and frame_labels. Build a Orange file format.')
	parser.add_argument('sub_segment_file', help='sub_segment_file')
	parser.add_argument('outDir', help='Output directory.')
	parser.add_argument('-v',dest='value',action='store_true', help='whether output value of the semantic tags.')
	parser.add_argument('-s',dest='slot',action='store_true', help='whether output slot of the semantic tags.')

	args = parser.parse_args()

	if not os.path.exists(args.outDir):
		os.mkdir(args.outDir)

	input = codecs.open(args.sub_segment_file)
	sub_segments = json.load(input)
	input.close()

	topic_dict = {}

	for session in sub_segments['sessions']:
		for segment in session['sub_segments']:
			topic = segment['topic']
			if topic not in topic_dict:
				topic_dict[topic] = []

			# prepare semtags
			sem_tag_set = []
			for utter_tags in segment['semantic_tags']:
				for tag in utter_tags:
					sem_tag_set.extend(extract_semantic_tags(tag))

			sem_tag_set = list(set([SemTagToString(tag, args.slot, args.value) for tag in sem_tag_set]))
			# prepare frame_labels

			frame_label_list = []
			for slot, value_list in segment['frame_label'].items():
				for value in value_list:
					frame_label_list.append(FrameTagToString(slot,value,args.value))

			topic_dict[topic].append((sem_tag_set, frame_label_list))

	for topic, items in topic_dict.items():
		outfile_name = os.path.join(args.outDir, topic)
		outfile_name += '_orange_'
		if args.slot:
			outfile_name += 's'
		if args.value:
			outfile_name += 'v'
		outfile_name += '.tab'

		output = codecs.open(outfile_name, 'w', 'utf-8')
		print >>output, '__basket_foo\nbasket\n'
		for sem_tag_set, frame_label_list in items:
			if not sem_tag_set:
				if not frame_label_list:
					pass
				else:
					print >>output, '%s' %(' '.join(frame_label_list))
			else:
				print >>output, '%s %s' %(' '.join(sem_tag_set), ' '.join(frame_label_list))
		output.close()

if __name__ =="__main__":
	main(sys.argv)

