#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
nsvc boosting method
first use sub_segments file to train a nsvc
then use the nsvc to find alignment in the dataset 
construct new training data sets
'''
import os, sys, codecs, json
from Utils import *
from GlobalConfig import *

from preprocessor import *

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import dataset_walker


def find_stop_words(dataset):
	stop_words_count = {}
	svc = slot_value_classifier()
	for call in dataset:
		for (log_utter, label_utter) in call:
			sys.stderr.write('%d:%d\n'%(call.log['session_id'], log_utter['utter_index']))
			flag = False
			for act in label_utter['speech_act']:
				if "ACK" in act['attributes']:
					flag = True
			if flag:
				sent = log_utter['transcript']
				tokens = svc.tokenizer.tokenize(sent)
				for t in tokens:
					stop_words_count[t] = stop_words_count.get(t, 0) + 1
	return stop_words_count


def main(argv):
	
	# 读取配置文件
	InitConfig()
	config = GetConfig()
	config.read([os.path.join(os.path.dirname(__file__),'../config/msiip_simple.cfg')])

	# 设置logging
	log_level_key = config.get('logging','level')
	run_code_name = os.path.basename(sys.argv[0])[0:-3]
	'''
	logging.basicConfig(filename = os.path.join(os.path.dirname(__file__), '../../output/logs', '%s_%s.log' %(run_code_name,time.strftime('%Y-%m-%d',time.localtime(time.time())))), \
    					level = GetLogLevel(log_level_key), 
    					format = '%(asctime)s %(levelname)8s %(lineno)4d %(module)s:%(name)s.%(funcName)s: %(message)s')
	'''
	

	parser = argparse.ArgumentParser(description='find stop words.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('stop_words_file', help='output stop words file')
	
	args = parser.parse_args()

	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)

	stop_words_count = find_stop_words(dataset)

	output = codecs.open(args.stop_words_file, 'w', 'utf-8')
	sorted_stop_words = sorted(stop_words_count.items(), key=lambda x:x[1], reverse=True)
	for word,count in sorted_stop_words:
		print >>output, '%s\t%d' %(word, count)
	output.close()

if __name__ =="__main__":
	main(sys.argv)

