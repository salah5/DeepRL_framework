
# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore')

from utils import time

import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO)

logger = logging.getLogger(__name__)

from main.lang import Lang

from main.main import Mai

import params

def run(config):

	logger.info('start.')

	logger.info('params: %s'%config)
	
	engine = Main(config)

	# 1. prepare the datasets, and create a language for encoder and decoder
	engine.data_prepreation()

	# 2. create an instance of the RL world, which needs to define an agent and env
	engine.world_creation(world_name='MIXER')

	# 3. train the agent on whole training set for several epochs, and save the agent
	engine.train()

	# 4. test the model by the model saved in 3.
	# #world.test(max_seq_len)

	# 5. report the results
	# # 	#print(world.report())

	logger.info('end.')

###
##
#
if __name__ == '__main__':

	config , _ = params.get_config()
	
	run(config)
