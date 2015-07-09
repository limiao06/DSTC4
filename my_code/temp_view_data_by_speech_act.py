#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
divide the dialog data based on the speech act
'''

import argparse, sys, os, codecs, json
sys.path.append(os.path.join(os.path.dirname(__file__),'../scripts/'))


def divide_dialog_data_by_speech_act(sub_segments):
	speech_act_dict = {}
	for session in sub_segments['sessions']:
		for sub_seg in session['sub_segments']:
			for speech_acts, semantic_tags in zip(sub_seg['speech_acts'], sub_seg['semantic_tags']):
				for act,tag in zip(speech_acts, semantic_tags):
					act_type = act['act']
					if len(act['attributes']) == 1:
						attr = act['attributes'][0]
						key = '%s:%s' %(act_type.strip(), attr.strip())
						if key not in speech_act_dict:
							speech_act_dict[key] = []
						speech_act_dict[key].append(tag)
	return speech_act_dict




def main(argv):
	parser = argparse.ArgumentParser(description='divide the dialog data based on the speech act.')
	parser.add_argument('sub_segment_file', help='The dataset to analyze')
	parser.add_argument('output', help='output')
	args = parser.parse_args()

	input = codecs.open(args.sub_segment_file, 'r', 'utf-8')
	sub_segments = json.load(input)
	input.close()

	speech_act_dict = divide_dialog_data_by_speech_act(sub_segments)

	output = codecs.open(args.output, 'w', 'utf-8')
	json.dump(speech_act_dict, output, indent=4)
	output.close()

	#track_file.close()
if __name__ =="__main__":
	main(sys.argv)
