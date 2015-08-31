#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
temp read train sample json file
'''

import os, sys, codecs, argparse
import json
from Utils import *




def main(argv):
	parser = argparse.ArgumentParser(description='Read alignment dict file.')
	parser.add_argument('alignment_dict', help='alignment_dict ')
	parser.add_argument('output', help='output')
	args = parser.parse_args()

	input = file(args.alignment_dict, 'r')
	alignment_dict = StrKeyDict2TupleKeyDict(json.load(input))
	input.close()

	output = file(args.output, 'w')
	sorted_dict = sorted(alignment_dict.items(), key=lambda x:x[0])
	for key, value in sorted_dict:
		print >>output, "%s----%s" %(key,value)
	output.close()

if __name__ =="__main__":
	main(sys.argv)

