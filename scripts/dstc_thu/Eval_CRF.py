#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
Eval crf output
'''

import argparse, sys, time, os
import codecs
from collections import defaultdict




def eval_crf(crf_out_file):
	input = codecs.open(crf_out_file, 'r', 'utf-8')
	sent_vec = []
	label_tag_dic = defaultdict(int)
	output_tag_dic = defaultdict(int)
	right_tag_dic = defaultdict(int)
	cover_right_tag_dic = defaultdict(int)
	while True:
		line = input.readline()
		if not line:
			break
		line = line.strip()
		if not line:
			# process sent_vec
			label_tags = []
			output_tags = []
			cur_label_tag = None
			cur_output_tag = None
			label_start_pos = -1
			output_start_pos = -1

			for i, tokens in enumerate(sent_vec):
				#print tokens
				label_tag = tokens[-2]
				output_tag = tokens[-1]
				#print label_tag, output_tag
				if label_tag[-1] == 'B':
					if cur_label_tag:
						label_tags.append((cur_label_tag, label_start_pos, i-1))
					cur_label_tag = label_tag[0:-2]
					label_start_pos = i
				elif label_tag[-1] == 'I':
					if label_tag[0:-2] == cur_label_tag:
						pass
					else:
						cur_label_tag = None
						label_start_pos = -1
				else:
					if cur_label_tag:
						label_tags.append((cur_label_tag, label_start_pos, i-1))
					cur_label_tag = None
					label_start_pos = -1

				if output_tag[-1] == 'B':
					if cur_output_tag:
						output_tags.append((cur_output_tag, output_start_pos, i-1))
					cur_output_tag = output_tag[0:-2]
					output_start_pos = i
				elif output_tag[-1] == 'I':
					if output_tag[0:-2] == cur_output_tag:
						pass
					else:
						cur_output_tag = None
						output_start_pos = -1
				else:
					if cur_output_tag:
						output_tags.append((cur_output_tag, output_start_pos, i-1))
					cur_output_tag = None
					output_start_pos = -1

			#print label_tags
			#print output_tags

			for tag in label_tags:
				label_tag_dic[tag[0]] += 1
				label_tag_dic['all'] += 1

			for tag in output_tags:
				output_tag_dic[tag[0]] += 1
				output_tag_dic['all'] += 1

			output_index = 0
			for label_tag in label_tags:
				for output_tag in output_tags:
					if output_tag[0] == label_tag[0]:
						if output_tag[1] == label_tag[1] and output_tag[2] == label_tag[2]:
							right_tag_dic[output_tag[0]] += 1
							right_tag_dic['all'] += 1

							cover_right_tag_dic[output_tag[0]] += 1
							cover_right_tag_dic['all'] += 1
						elif output_tag[1] >= label_tag[1] and output_tag[2] <= label_tag[2]:
							cover_right_tag_dic[output_tag[0]] += 1
							cover_right_tag_dic['all'] += 1
						elif output_tag[1] <= label_tag[1] and output_tag[2] >= label_tag[2]:
							cover_right_tag_dic[output_tag[0]] += 1
							cover_right_tag_dic['all'] += 1
			#a = raw_input('press to continue')
			sent_vec = []
		else:
			sent_vec.append(line.split())
	'''
	print label_tag_dic
	print output_tag_dic
	print right_tag_dic
	print cover_right_tag_dic
	'''

	print 'Key,Precision,Recall,F1,C_Precision,C_Recall,C_F1'
	for key in label_tag_dic:
		print key+',',
		pre = right_tag_dic[key]*1.0 / output_tag_dic[key]
		recall = right_tag_dic[key]*1.0 / label_tag_dic[key]
		print '%.3f(%d/%d),' %(pre, right_tag_dic[key], output_tag_dic[key]) ,
		print '%.3f(%d/%d),' %(recall, right_tag_dic[key], label_tag_dic[key]) ,
		print '%.3f,' %(2*pre*recall/(pre+recall)),

		c_pre = cover_right_tag_dic[key]*1.0 / output_tag_dic[key]
		c_recall = cover_right_tag_dic[key]*1.0 / label_tag_dic[key]
		print '%.3f(%d/%d),' %(cover_right_tag_dic[key]*1.0 / output_tag_dic[key], cover_right_tag_dic[key], output_tag_dic[key]) ,
		print '%.3f(%d/%d),' %(cover_right_tag_dic[key]*1.0 / label_tag_dic[key], cover_right_tag_dic[key], label_tag_dic[key]) ,
		print '%.3f,' %(2*c_pre*c_recall/(c_pre+c_recall))









def main(argv):
	parser = argparse.ArgumentParser(description='Evaluate the crf output file.')
	parser.add_argument('crf_out_file', help='crf_out_file')
	args = parser.parse_args()
	eval_crf(args.crf_out_file)

if __name__ =="__main__":
	main(sys.argv)

