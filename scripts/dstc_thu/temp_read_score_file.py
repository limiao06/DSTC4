#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
temp read score files
'''

import os, sys, codecs, argparse


def read_score_file(input):
	score_dict = {}
	line = input.readline()
	if not line.strip() == 'topic, slot, schedule, stat, N, result':
		raise Exception('Error: the input file is not a dstc score file!')
	while True:
		line = input.readline()
		if not line:
			break
		tokens = [t.strip() for t in line.split(',')]
		if len(tokens) != 6:
			continue
		topic = tokens[0]
		slot = tokens[1]
		if topic not in score_dict:
			score_dict[topic] = {}
		if topic == 'basic':
			score_dict[topic][slot] = tokens[-1]
		else:
			schedule_id = int(tokens[2])
			stat_term = tokens[3]
			value = tokens[-1]
			if value == '-':
				value = 'NULL'
			else:
				value = float(value)

			if slot not in score_dict[topic]:
				score_dict[topic][slot] = {}
			if stat_term not in score_dict[topic][slot]:
				score_dict[topic][slot][stat_term] = [None, None]
			score_dict[topic][slot][stat_term][schedule_id-1] = value

	return score_dict





def main(argv):
	parser = argparse.ArgumentParser(description='Read score file, output a csv file.')
	parser.add_argument('score_file', help='score file')
	parser.add_argument('output', help='output csv file')
	args = parser.parse_args()

	stat_terms = ['acc', 'precision', 'recall', 'f1']

	input = file(args.score_file, 'r')
	score_dict = read_score_file(input)
	input.close()

	output = file(args.output, 'w')

	print >>output, ',,schedule1,,,,schedule2,,,'
	print >>output, 'topic,slot,%s,%s' %(','.join(stat_terms), ','.join(stat_terms))
	for topic in score_dict:
		if topic == 'basic':
			continue
		for slot, stat_terms_dic in score_dict[topic].items():
			print >>output, '%s,%s,' %(topic,slot),
			for i in range(2):
				for key in stat_terms:
					if stat_terms_dic[key][i] == 'NULL':
						print >>output, 'NULL,',
					else:
						print >>output, '%.4f,' %(stat_terms_dic[key][i]) ,
			print >>output
	output.close()

if __name__ =="__main__":
	main(sys.argv)

