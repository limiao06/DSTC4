#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
Read association rules
output a json format rules file
'''

import argparse, sys, time, json, os, codecs
from collections import defaultdict

def read_association_rule_dir(association_rule_dir):
	if not os.path.isdir(association_rule_dir):
		raise Exception('Error: read_ontology: association_rule_dir %s is not a directory!' %(association_rule_dir))

	association_rules = {}

	files = os.listdir(association_rule_dir)
	for f in files:
		if f.endswith('.tab.rul'):
			f_base_name = f[0:f.find('.')]
			print f_base_name
			tokens = f_base_name.split('_')
			topic = tokens[0]
			rule_type = tokens[2]
			if not rule_type:
				rule_type = 'base'
			if topic not in association_rules:
				association_rules[topic] = {}
			if rule_type not in association_rules[topic]:
				association_rules[topic][rule_type] = {}

			association_rules[topic][rule_type]['rule_file'] = os.path.join(association_rule_dir, f)
			association_rules[topic][rule_type]['rules'] = read_association_rule_file(association_rules[topic][rule_type]['rule_file'])
			
	return association_rules

def read_association_rule_file(rule_file):
	input = codecs.open(rule_file, 'r', 'utf-8')
	lines = input.readlines()
	input.close()

	rules = {}

	for line in lines:
		l = line.strip()
		tokens = l.split(' ')
		if len(tokens) == 7 and tokens[5] == '->':
			#print l
			rule_key = tokens[4]
			if rule_key not in rules:
				rules[rule_key] = {}
			rule_content = tokens[-1]
			assert rule_content not in rules[rule_key]
			rules[rule_key][rule_content] = {}
			rules[rule_key][rule_content]['prob'] = float(tokens[2])
			rules[rule_key][rule_content]['occurrence'] = int(float(tokens[0]))
	return rules


def main(argv):
	parser = argparse.ArgumentParser(description='Read rules generated by orange.')
	parser.add_argument('rule_dir',help='rule_dir')
	parser.add_argument('output',help='output file')
	args = parser.parse_args()

	association_rules = read_association_rule_dir(args.rule_dir)
	output = codecs.open(args.output, 'w', 'utf-8')
	json.dump(association_rules, output, indent=4)
	output.close()



if __name__ =="__main__":
	main(sys.argv)

