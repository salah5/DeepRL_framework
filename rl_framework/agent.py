# coding: utf-8
import numpy as np
from .deep_models.output.softmax import SoftMax
import torch
import torch.nn as nn

class Agent(nn.Module):
	def __init__(self, name, action_set, state_size, config):
		super(Agent, self).__init__()
		
		self.name = name

		self.config = config
		
		self.action_set = action_set # a dictionary of possible {id:action}
		
		self.action_size = len(self.action_set)
		
		self.state_size = state_size

		self.softmax = SoftMax(input_size=self.state_size,
			   				   output_size=self.action_size,
			   				   tie_output_embed=False,
			   				   embedding=None)

	def policy(self, states, reward):
		''' 
			given a state or an observation, the agent picks one of the actions in the action_set
			note we should have a policy functoin here to make the decision.
			states: bs * max_dec_len-1 * dec_cell_size
			reward: =0
		'''
		
		batch_size = states.size(0)

		states = states.view(-1, self.state_size)

		predicted_softmax = self.softmax(inp_var=states, batch_size=batch_size) 
		
		action_probs = predicted_softmax.view(batch_size, -1, self.action_size)

		actions = self.determine_actions(action_probs)

		return action_probs, actions

	def determine_actions(self, action_probs):
		'''
			returns an action based on the input mode
			action_probs: bs * max_dec_len-1 * action_size
		'''
		symbols = []
		
		for i in range(0, self.config.max_dec_len-1):

			step_output_slice = action_probs[:, i, :]

			step_output_slice = step_output_slice.squeeze(1)

			if self.config.gen_type == 'greedy':

				symbol = step_output_slice.topk(1)[1]

				symbols.append(symbol)

		symbols = torch.stack(symbols, dim=1)

		symbols = symbols.squeeze(2)
	
		return symbols

        # elif gen_type == 'sample':
        #         symbols = self.gumbel_max(step_output_slice)
        #     elif gen_type == 'beam':
        #         if step == 0:
        #             seq_score = step_output_slice.view(batch_size, -1)
        #             seq_score = seq_score[:, 0:self.output_size]
        #         else:
        #             seq_score = cum_sum + step_output_slice
        #             seq_score = seq_score.view(batch_size, -1)

        #         top_v, top_id = seq_score.topk(beam_size)

        #         back_ptr = top_id.div(self.output_size).view(-1, 1)
        #         symbols = top_id.fmod(self.output_size).view(-1, 1)
        #         cum_sum = top_v.view(-1, 1)
        #         back_pointers.append(back_ptr)
        #     else:
        #         raise ValueError("Unsupported decoding mode")

        #     sequence_symbols.append(symbols)

        #     eos_batches = symbols.data.eq(self.eos_id)
        #     if eos_batches.dim() > 0:
        #         eos_batches = eos_batches.cpu().view(-1).numpy()
        #         update_idx = ((lengths > di) & eos_batches) != 0
        #         lengths[update_idx] = len(sequence_symbols)
        #     return cum_sum, symbols


			        # save the decoded sequence symbols and sequence length

