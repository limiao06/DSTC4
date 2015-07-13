#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
temp read report files
'''

import os, sys, codecs, argparse


def read_report_file(input):
	last_content = ''
	while True:
		line = input.readline()
		if not line:
			break
		content = line.strip()

		if content.startswith('featured metrics'): # into a metrics
			out_metric = {}
			out_metric['title'] = ';'.join(last_content.split(','))
			while True:
				line = input.readline()
				if not line:
					break
				l = line.strip()
				if not l:
					continue
				if l.startswith('wall_time_per_utterance'): # finish, output
					break
				elif l.startswith('s') and l.find('|')!=-1:
					tokens = l.split('|')
					if len(tokens) == 4:
						key = tokens[0].strip()
						value1 = float(tokens[1].strip())
						value2 = float(tokens[2].strip())
						out_metric[key] = (value1,value2)
				else:
					continue
			yield out_metric
		if content:
			last_content = content




def main(argv):
	parser = argparse.ArgumentParser(description='Read report file, output a csv file.')
	parser.add_argument('report_file', help='report file')
	parser.add_argument('output', help='output csv file')
	args = parser.parse_args()

	keys = ['segment.accuracy', 'slot_value.precision', 'slot_value.recall', 'slot_value.fscore']

	input = file(args.report_file, 'r')
	output = file(args.output, 'w')

	print >>output, ',,schedule1,,,,schedule2,,,'
	print >>output, 'titile,%s,%s' %(','.join(keys), ','.join(keys))
	for metrics in read_report_file(input):
		print >>output, '%s,' %(metrics['title']) ,
		for i in range(2):
			for key in keys:
				print >>output, '%.4f,' %(metrics[key][i]) ,
		print >>output, ''

	output.close()
	input.close()


if __name__ =="__main__":
	main(sys.argv)

