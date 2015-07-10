#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
Belief State class
'''

import logging
import codecs


from Utils import *
from GlobalConfig import *

class BeliefState(object):
	MY_ID = 'BeliefState'
	def __init__(self, mode=None, alpha=None):
		'''
		mode = 'max' 取最大值保留
		mode = 'average' 历史和当前加权平均
		mode = 'enhance' 对重复出现的slot_value对进行加强

		self.state is a dict like:
		{
			'info':{
				'prob': -1,
				'values':{
					'PREFERENCE': p1,
					'FOOD': p2,
					...
				}
			}
		}
		'''
		self.config = GetConfig()
        self.appLogger = logging.getLogger(self.MY_ID)

        if not mode:
        	self.appLogger.info('mode is not assigned, use the default config:')
        	self.mode = self.config.get(self.MY_ID,'mode')
        else:
        	self.mode = mode
        self.appLogger.info('mode: %s' %(self.mode))

        if not alpha:
        	self.appLogger.info('alpha is not assigned, use the default config:')
        	self.alpha = self.config.getfloat(self.MY_ID,'alpha')
        else:
        	self.alpha = alpha
        self.appLogger.info('alpha: %.2f' %(self.alpha))

		self.state = {}
		

	def reset(self):
		self.state = {}

	def AddFrame(self, frame):
		'''
		frame is a dict same as self.state
		{
			'info':{
				'prob': -1,
				'values':{
					'PREFERENCE': p1,
					'FOOD': p2,
					...
				}
			}
		}
		'''
		for slot, slot_dict in frame.items():
			self._AddSlot(slot, slot_dict)


	def _AddSlotDict(self, slot, slot_dict):
		# add slot probability:
		self._AddSLot2State(slot, slot_dict['prob'])
		# add slot value probabilities:
		for value, prob in slot_dict['values'].items():
			self._AddSlotValue2State(slot,value,prob)

	def _AddSLot2State(self, slot, prob):
		if slot not in self.state:
			self.state[slot] = {'prob': prob, 'values':{}}
		else:
			if self.state[slot]['prob'] == -1:
					self.state[slot]['prob'] = prob
			elif self.mode == 'max':
				if self.state[slot]['prob'] < prob:
					self.state[slot]['prob'] = prob
			elif self.mode == 'average':
				self.state[slot]['prob'] = self.state[slot]['prob'] * self.alpha + prob * (1-self.alpha)
			elif self.mode == 'enhance':
				self.state[slot]['prob'] = 1 - (1 - self.state[slot]['prob']) * (1-prob)
			else:
				self.appLogger.error('Unknown mode: %s' %(self.mode))


	def _AddSlotValue2State(self, slot, value, prob):
		if value not in self.state[slot]['values']:
			self.state[slot]['values'][value] = prob
		else:
			if self.mode == 'max':
				if self.state[slot]['values'][value] < prob:
					self.state[slot]['values'][value] = prob
			elif self.mode == 'average':
				self.state[slot]['values'][value] = self.state[slot]['values'][value] * self.alpha + prob * (1-self.alpha)
			elif self.mode == 'enhance':
				self.state[slot]['values'][value] = 1 - (1 - self.state[slot]['values'][value]) * (1-prob)
			else:
				self.appLogger.error('Unknown mode: %s' %(self.mode))










