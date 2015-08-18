#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
output clean script
clean the output.json so that it can be correct submitted
1. remove the 'frame_prob'



the output file's name is predefined,
if the input file's name is answer.json 
then the output file's name will be answer.cln.json
'''

import argparse, json, os, copy, sys


def main(argv):
	parser = argparse.ArgumentParser(description='output cleaner.')
	parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='The track file to be cleaned.')
	args = parser.parse_args()


	track_file = open(args.trackfile)
	track = json.load(track_file)
	track_file.close()

	filepath, filename = os.path.split(args.trackfile)
	dot_place = filename.rfind('.')
	if dot_place == -1:
		out_name = filename + '.cln'
	else:
		out_name = filename[0:dot_place] + '.cln' + filename[dot_place:]
	outfile = os.path.join(filepath,out_name)

	for session in track['sessions']:
		for utter in session['utterances']:
			if "frame_prob" in utter:
				del utter['frame_prob']


	output = open(outfile, "wb")
	json.dump(track, output, indent=4)
	output.close()

	

if __name__ =="__main__":
	main(sys.argv)
