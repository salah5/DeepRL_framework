# -*- coding: utf-8 -*-
from .deep_models.enc2dec.enc2dec import EncDec as seq2seq
import torch.nn as nn

class Environment(nn.Module):

	def __init__(self, 
				name, 
				init_state, 
				init_reward, 
				delayed_reward,
				config):
		super(Environment, self).__init__()
		self.name = name
		self.init_state = init_state
		self.init_reward = init_reward
		self.delayed_reward = delayed_reward

		self.config = config

		self.seq2seq = seq2seq(self.config)

	def assing_reward(self, action, state):
		'''
		it sees an action performed on state and returns a reward for it
		'''
		reward = 1.0

		return reward

	def next_state(self, action, state):

		next_state = None
		
		return next_state

	def get_states(self, inp, out):
		'''
		This function is used in case of delayed reward. 
		it gets a sequence of inputs and returns a sequence of states
		# inp_seq: (nummpy array, list): x_id= bs * max_seq_len, x_len
		# out_seq (nummpy array, list): x_id= bs * max_seq_len, x_len
		'''
			
		decoder_outputs, decoder_hidden, attn = self.seq2seq(
												data_feed= (inp, out), 
							  					mode = self.config.gen_mode)

		return decoder_outputs


	def apply_action(self, action, state):
		'''
			by applying action_t on state_t, what is the next state of the environment
		'''
		
		reward = self.assing_reward(action, state)

		next_state = self.next_state(action, state)


		return next_state, reward
