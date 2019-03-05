# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)

import numpy as np

class Dataset_loader(object):
	
	def __init__(self, x_path, y_path, config):

		self.x_path = x_path

		self.y_path = y_path

		self.x_lang = config.lang_x

		self.y_lang = config.lang_y

		self.max_x_len = config.max_seq_len

		self.max_y_len = config.max_dec_len

		self.batch_size = config.batch_size

		self.fix_batch = config.fix_batch

		self.size = None

	def load(self):

		self.x_word = self._load_dataset(self.x_path)

		self.y_word = self._load_dataset(self.y_path)

		assert len(self.x_word)==len(self.y_word)

		self.size = len(self.x_word)

		logger.info('datasets are loaded: %s, %s '%(self.x_path, self.y_path))

		logger.info('dataset size: %d '%(self.size))

		self.x_id, self.x_mask, self.x_len = self._seq_to_id(self.x_word, 
															self.x_lang, 
															self.max_x_len)

		self.y_id, self.y_mask, self.y_len = self._seq_to_id(self.y_word, 
															self.y_lang, 
															self.max_y_len)

		logger.info('converted words to index and padding')

		self.data_samples = list(zip(self.x_id, self.x_mask, self.x_len, self.x_word, 
									 self.y_id, self.y_mask, self.y_len, self.y_word))
		if self.fix_batch:

			self.data_indices = list(np.argsort(self.x_len))[::-1]

		else:

			self.data_indices = range(self.x_len)


	def get_batches(self):
	
		'''
		data_samples: a list of samples [sample1, sample2, sample3,..., sample_n]
		batch_size: int
		'''

		batch_indices = self._get_batch_indices()

		batches = []

		for (start,end) in batch_indices:	

			batch = [self.data_samples[index] for index in self.data_indices[start:end]]

			batches.append(batch)

		return batches

	def _get_batch_indices(self):
    
		num_batches = (self.size + self.batch_size - 1) // self.batch_size  # round up
    
		return [(i * self.batch_size, min(self.size, (i + 1) * self.batch_size)) for i in range(num_batches)]

	def _get_lens(self, k_data):

		return [len(k) for k in k_data] 


	def _seq_to_id(self, k_word, lang, max_len):

		k_id = []
		k_mask = []
		k_len = []

		for k in k_word:
			
			id, mask, [unk_hit, num_hit, total] = lang.sent_to_index(sentence=k, 
																	max_sent_len = max_len, 
																	padding_place='post')

			k_id.append(id)
			
			k_mask.append(mask)

			k_len.append(sum(mask))

		return k_id, k_mask, k_len

	def _load_dataset(self, path):
	    '''
	    path: the path to the preprocessed and saved dataset
	    '''
	    with open(path,'r') as f:
	    
	        content = f.read()

	        ds = [line.strip().split() for line in content.splitlines()]

	    return ds
