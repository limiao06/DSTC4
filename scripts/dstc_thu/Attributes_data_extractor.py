#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
Attributes data extractor

read sub_utters data
extract Attributes train data
'''

import argparse, sys, time, json, os, codecs, logging
import re
from collections import defaultdict

from GlobalConfig import *
from Utils import *



from sub_utters_data_walker import sub_utters_data_walker
from Semantic_Tag_Data_Extractor import SemTagExtractor





def main(argv):
	# 读取配置文件
	InitConfig()
	config = GetConfig()
	config.read([os.path.join(os.path.dirname(__file__),'../config/msiip_simple.cfg')])

	# 设置logging
	log_level_key = config.get('logging','level')
	run_code_name = os.path.basename(sys.argv[0])[0:-3]
	logging.basicConfig(filename = os.path.join(os.path.dirname(__file__), '%s_%s.log' %(run_code_name,time.strftime('%Y-%m-%d',time.localtime(time.time())))), \
    					level = GetLogLevel(log_level_key), 
    					format = '%(asctime)s %(levelname)8s %(lineno)4d %(module)s:%(name)s.%(funcName)s: %(message)s')
	
	parser = argparse.ArgumentParser(description='Extract Semantic Tag Data.')
	parser.add_argument('sub_utters_file', help='sub_utters_file')
	parser.add_argument('output', help='Output json file')

	args = parser.parse_args()

	output = codecs.open(args.output, 'w', 'utf-8')

	walker = sub_utters_data_walker(args.sub_utters_file)
	
	count = 0
	sub_utters_list = []

	interesting_attr = ['HOW_MUCH', 'HOW_TO', 'PREFERENCE', 'WHAT', 'WHEN', 'WHERE', 'WHICH']

	interesting_attr_dic = {}

	for attr in interesting_attr:
		interesting_attr_dic[attr] = []



	for (pre_utter, cul_utter) in walker.ReadUtter():
		for i, sub_tag in enumerate(cul_utter['sub_tag_list']):
			count += 1
			if count % 100 == 0:
				sys.stderr.write('%d\n' %(count))
			(token_list, _, _) = SemTagExtractor._ReadSentTags(sub_tag)
			if token_list:
				sub_utters_list.append(' '.join(token_list))
				
				for attr in cul_utter["speech_acts"][i]["attributes"]:
					if attr in interesting_attr:
						interesting_attr_dic[attr].append(len(sub_utters_list) - 1)

	print 'all', count
	for attr, attr_list in interesting_attr_dic.items():
		print attr, len(attr_list)
			
	out_json = {}
	out_json['sub_utter_data'] = sub_utters_list
	out_json['attr_data_index'] = interesting_attr_dic
	json.dump(out_json, output, indent = 4)
	output.close()

	
	

	#sent = "why we want to go to <AREA FROM-TO=\"TO\" REL=\"NONE\" CAT=\"CITY\">Singapore</AREA> is really %um to really try out the <FOOD CAT=\"MAIN\" FROM-TO=\"NONE\" REL=\"NONE\">food</FOOD>."
	#sent = "what kind of <FOOD CAT=\"MAIN\" FROM-TO=\"NONE\" REL=\"NONE\">food</FOOD> does he like then?"
	#extractor = SemTagExtractor(sent)


if __name__ =="__main__":
	main(sys.argv)

