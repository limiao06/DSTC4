#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
temp read train sample json file
'''

import os, sys, codecs, argparse
import json


def show_train_samples(train_sample_json, tuple_name, show_nums = 5):
	if tuple_name not in train_sample_json['train_labels']:
		print 'Error: The input tuple_name is not in train samples!'
		return
	sample_num = sum(train_sample_json['train_labels'][tuple_name])
	print 'There are %d samples for tuple: %s' %(sample_num, tuple_name)
	cur_index = -1
	shown_index = -1
	show_list = []
	while True:
		for index, label in enumerate(train_sample_json['train_labels'][tuple_name]):
			if label == 1:
				cur_index += 1
				if cur_index < shown_index:
					continue
				show_list.append((cur_index, train_sample_json['train_samples'][index]))
				shown_index += 1
				if len(show_list) == show_nums:
					break
		print 'train examples for %s' %(tuple_name)
		for idx, sample in show_list:
			print 'sample %d:' %(idx)
			for token in sample:
				print '['
				for t_str in token:
					print t_str
				print ']'
			print
		if index >= len(train_sample_json['train_labels'][tuple_name]) - 1:
			break
		cmd = raw_input('press any key to continue or press break to break: ')
		if cmd == 'break':
			break
	return




def main(argv):
	parser = argparse.ArgumentParser(description='Read train_sample json file, output train samples of some semantic tuple.')
	parser.add_argument('train_sample_file', help='train sample json file')
	args = parser.parse_args()

	input = file(args.train_sample_file, 'r')
	train_sample_json = json.load(input)
	input.close()

	while True:
		cmd = raw_input('input a tuple name or exit: ')
		if cmd == 'exit':
			break
		else:
			show_train_samples(train_sample_json, cmd)


if __name__ =="__main__":
	main(sys.argv)

