#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
DSTC4 rules
some rules to prune the frame label
'''

class DSTC4_rules(object):
	'''
	some rules
	'''
	MY_ID = 'DSTC4_rules'
	def __init__(self, tagsets):
		self.tagsets = tagsets

	def prune_frame_label(self, topic, frame_label):
		for slot in frame_label:
			if slot not in self.tagsets[topic]:
				del frame_label[slot]

			remove_list = []
			for i, value in enumerate(frame_label[slot]):
				if value not in self.tagsets[topic][slot]:
					remove_list.append(i)
				frame_label[slot] = [v for i,v in enumerate(frame_label[slot]) if i not in remove_list]

		if topic == 'ATTRACTION' and 'PLACE' in self.frame and 'NEIGHBOURHOOD' in self.frame and self.frame['PLACE'] == self.frame['NEIGHBOURHOOD']:
			del self.frame['PLACE']

		return frame_label
