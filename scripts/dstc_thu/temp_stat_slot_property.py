#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
find slot property: multi-label slot or multi-class slot
'''
import os, sys, codecs, json, argparse


sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import dataset_walker

def find_slot_property(dataset):
	slot_count_dic = {}
	for call in dataset:
		for (log_utter, label_utter) in call:
			if 'frame_label' in label_utter:
				frame_label = label_utter['frame_label']
				for slot in frame_label:
					if slot not in slot_count_dic:
						slot_count_dic[slot] = [0,0]
					slot_count_dic[slot][0] += 1
					if len(frame_label[slot]) > 1:
						slot_count_dic[slot][1] += 1

	return slot_count_dic





def main(argv):
	parser = argparse.ArgumentParser(description='find slot property.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('output', help='output slot property file')
	
	args = parser.parse_args()

	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)

	slot_count_dic = find_slot_property(dataset)

	output = codecs.open(args.output, 'w', 'utf-8')
	sorted_slot_count = sorted(slot_count_dic.items(), key=lambda x:x[1][1], reverse=True)
	for slot,count in sorted_slot_count:
		print >>output, '%s\t%d\t%d' %(slot, count[0], count[1])
	output.close()

if __name__ =="__main__":
	main(sys.argv)

