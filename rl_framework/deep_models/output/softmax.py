# -*- coding: utf-8 -*-
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class SoftMax(nn.Module):

	def __init__(self,
			   input_size,
			   output_size,
			   tie_output_embed=False,
			   embedding=None
				):
		super(SoftMax, self).__init__()

		self.embedding = embedding

		self.input_size = input_size

		self.output_size = output_size
		
		if tie_output_embed:

			self.project = lambda x: x * self.embedding.weight.transpose(0, 1)

		else:
            
			self.project = nn.Linear(self.input_size, self.output_size)
		
		self.function = F.log_softmax

	def forward(self, inp_var, batch_size):

		logits = self.project(inp_var)

		predicted_softmax = self.function(logits, dim=logits.dim()-1)

		return predicted_softmax
