#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
temp read train sample json file
'''

import os, sys, codecs, argparse


def show_train_samples(train_sample_json, tuple_name, show_nums = 5):
	if tuple_name not in train_sample_json['train_labels']:
		print 'Error: The input tuple_name is not in train samples!'
		return
	cur_index = 0
	show_list = []
	while True:
		for index, label in enumerate(train_sample_json['train_labels'][tuple_name]):
			if index < cur_index:
				continue
			if label == 1:
				show_list.append(train_sample_json['train_samples'][index])
				if len(show_list) == show_nums:
					cur_index = index + 1
					break
		print 'train examples for %s' %(tuple_name)
		for sample in show_list:
			print sample.__str__()
		cmd = raw_input('press any key to continue or press break to break')
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
		cmd = raw_input('input a tuple name or exit')
		if cmd == 'exit':
			break
		else:


if __name__ =="__main__":
	main(sys.argv)

