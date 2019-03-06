# -*- coding: utf-8 -*-

from .agent import Agent

from .environment import Environment 

import numpy as np

import torch

from torch.autograd import Variable

import torch.cuda as cuda

from .deep_models.deep_utils.deep_utils import np2var

from .deep_models.deep_utils.deep_utils import INT, FLOAT, LONG, cast_type

from .deep_models.deep_utils import criterions

import torch.nn as nn

import os

class World(nn.Module):

	def __init__(self, name, config):
		super(World, self).__init__()

		self.name = name

		self.config = config

		self.seed = self.config.seed

		torch.manual_seed(self.seed)

		if torch.cuda.is_available():

			torch.cuda.manual_seed_all(self.seed)

		self.env = Environment(name='mt', init_state=None, init_reward=0.0, delayed_reward=True, config=self.config)

		action_set = config.lang_y.index2word # because we want should get indices and then convert them to word

		self.agent = Agent(name='lang_gen', action_set=action_set, state_size=config.dec_cell_size, config= self.config)

		self.np2var = np2var

		self.use_gpu = config.use_gpu

		if self.use_gpu and torch.cuda.is_available():

			self.agent = self.agent.cuda()

			self.env = self.env.cuda()


		self.pad_index = self.config.lang_y.word2index[self.config.lang_y.pad_word]

		# TODO: define a  function _get_loss which returns a loss function given config
		self.nll_loss = criterions.NLLEntropy(self.pad_index, self.config)
        
		self.cat_kl_loss = criterions.CatKLLoss()
        
		self.cross_ent_loss = criterions.CrossEntropyoss()
        
		self.entropy_loss = criterions.Entropy()
	
		# TODO: get different possible optimisers
		self.optimizer = torch.optim.Adam(self.parameters(), eps=1e-06, lr=config.init_lr)

	def iterate(self, inp_data, update):
		'''
		 This function updates all parameters in the world (including agent and env) one time by backpropagation.
		 inp_data: a list of pairs [(x1,y1),(x2,y2),..., (xn,yn)]
		 inp_data contains only indices of tokens
		 for now we only support delayed_reward= true
		'''
		# convert input_data, output_data to numpy arrays, and decompose the inp_data 

		x_id, x_mask, x_len, x_word, y_id, y_mask, y_len, y_word = list(zip(*inp_data))

		x_id = np.array(x_id, np.int32) #bs * seq_len

		y_id = np.array(y_id, np.int32) #bs * seq_len

		x_id = self.np2var(inputs= x_id, dtype=LONG, use_gpu=self.use_gpu)

		y_id = self.np2var(inputs= y_id, dtype=LONG, use_gpu=self.use_gpu)

		inp = (x_id, x_len)

		out = (y_id, y_len)
		
		# TO DO: set the mode of the world on training

		# re-set optimizer to zero

		self.optimizer.zero_grad()

		# world asks the agent to perform an action

		if self.env.delayed_reward == True:

			states = self.env.get_states(inp, out)

			action_probs, actions = self.agent.policy(states, reward=0) 
			# INFO: action_probs: bs * action_size* (max_dec_len-1)
			# INFO: actions: (bs * max_dec_len-1)

			# state, reward = self.env.apply_action(action, state)

		else:
			raise NotImplementedError('delayed_rewards are only accepted! ')

		# compute the loss function according to the returned reward
		
		# labels are everything except the first item
		labels = y_id[:, 1:].contiguous()
		# bs * (max_dec_len-1)
		
		nll = self.nll_loss(action_probs, labels)
		
		loss = nll
		
		if update:
			
			# compute gradients of  parameters 
			
			loss.backward()

			# update the parameters
			
			self.optimizer.step()

		# # return the loss of this update.
		actions = actions.cpu().numpy()

		labels = labels.cpu().numpy()

		return loss, actions, labels

	def save(self, path):

		torch.save(self.state_dict(), path)




