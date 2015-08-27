import argparse, sys, time, json
from fuzzywuzzy import fuzz

class BaselineTracker(object):
	def __init__(self, tagsets):
		self.tagsets = tagsets
		self.frame = {}
		self.memory = {}

		self.reset()

	def addSubSeg(self, sub_seg):
		self.reset()
		topic = sub_seg['topic']
		for utter in sub_seg['utter_sents']:
			trans = utter[utter.find(' ') +1:]
			self.addTrans(trans, topic)
		return self.frame

	def addTrans(self, trans, topic):
		transcript = trans.replace('Singapore', '')
		if topic in self.tagsets:
			for slot in self.tagsets[topic]:
				for value in self.tagsets[topic][slot]:
					ratio = fuzz.partial_ratio(value.lower(), transcript.lower())
					if ratio > 85:
						if slot not in self.frame:
							self.frame[slot] = []
						if value not in self.frame[slot]:
							self.frame[slot].append(value)
			if topic == 'ATTRACTION' and 'PLACE' in self.frame and 'NEIGHBOURHOOD' in self.frame and self.frame['PLACE'] == self.frame['NEIGHBOURHOOD']:
				del self.frame['PLACE']

	def reset(self):
		self.frame = {}

