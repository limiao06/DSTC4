#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
using orange find association rules
'''

import argparse, sys, time, json, os
import codecs
import Orange

def check_rule(rule):
	flag = False
	str_rule = '%s' %(rule)
	(condition, result) = str_rule.split('->')
	condition_tokens = condition.strip().split()
	for token in condition_tokens:
		if not token.startswith('SemTag'):
			return False
	result_tokens = result.strip().split()
	for token in result_tokens:
		if not token.startswith('Frame'):
			return False

	return True


def main(argv):
	parser = argparse.ArgumentParser(description='using orange find association rules.')
	parser.add_argument('input_dir', help='input_dir')
	parser.add_argument('output_dir', help='Output file.')
	parser.add_argument('-s','-support', dest='support',type=int, action='store', default = 3,
						help='minimum occurance times.')
	args = parser.parse_args()

	max_item_sets_num = 10000000

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	for file_name in os.listdir(args.input_dir):
		if file_name.endswith('.tab'):
			print file_name
			try:
				data = Orange.data.Table(os.path.join(args.input_dir, file_name))
				sample_num = len(data)
				support  = args.support *1.0/ sample_num
				rules = Orange.associate.AssociationRulesSparseInducer(data, support=support, max_item_sets=max_item_sets_num)

				outfile = os.path.join(args.output_dir, file_name)
				outfile += '.rul'
				output = codecs.open(outfile, 'w', 'utf-8')
				for r in sorted(rules, key=lambda x:x.support, reverse=True):
					if check_rule(r):
						print >>output, "%4.1f %4.3f %4.3f  %s" % (int(r.support * sample_num), r.support, r.confidence, r)
				output.close()
			except Exception, e:
				print e



if __name__ =="__main__":
	main(sys.argv)

