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
		BeliefState class: 
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
			self._AddSlotDict(slot, slot_dict)


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

	@staticmethod
	def StateEnsemble(bs_list, config):
		'''
		bs_list is a list, each item is a dict like state
		config indicate the ensemble method:
			if config is a list of weight, then the ensemble method is average
			elif config is a string 'vote', the ensemble method is vote
		'''
		mode = None
		if isinstance(config, list):
			if len(bs_list) != len(config):
				raise Exception('The ensemble method is average, but the bs vector dose not match the weight vector! %d %d' %(len(bs_list), len(config)))
			sum_weight = sum(config)
			weight_vector = [i*1.0/sum_weight for i in config]
			mode = 'average'
		elif config == 'vote':
			mode = 'vote'
		else:
			raise Exception('Unknown ensemble method: %s' %(config))

		# prepare out_state
		out_state = {}
		for bs in bs_list:
			for slot in bs:
				if slot not in out_state:
					out_state[slot] = {'prob': -1, 'values':{}}
				for value in bs[slot]['values']:
					if value not in out_state[slot]['values']:
						out_state[slot]['values'][value] = 0.0


		# ensemble
		for slot in out_state:
			# calculate slot prob, slot prob is the max value
			max_prob = -1
			for bs in bs_list:
				if slot in bs:
					if max_prob < bs[slot]['prob']:
						max_prob = bs[slot]['prob']
			out_state[slot]['prob'] = max_prob

			# calculate value prob
			for value in out_state[slot]['values']:
				if mode == 'average':
					value_prob_list = []
					for bs in bs_list:
						if slot not in bs:
							value_prob_list.append(0.0)
						else:
							if value not in bs[slot]['values']:
								value_prob_list.append(0.0)
							else:
								value_prob_list.append(bs[slot]['values'][value])
					for weight, prob in zip(weight_vector, value_prob_list):
						out_state[slot]['values'][value] += weight * prob

				if mode == 'vote':
					pass
					# to be completed

		return out_state















