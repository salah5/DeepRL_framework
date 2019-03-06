# -*- coding: utf-8 -*-

from .lang import Lang

from rl_framework.world import World

from data_processing.dataset_utils import prepare_data

from data_processing.dataset_loader import Dataset_loader

from rl_framework.deep_models.deep_utils.deep_utils import summary

import numpy as np

import logging

logger = logging.getLogger(__name__)

import os

from evaluators.evaluators import bleu


class Main(object):
    ###
    ##
    #
    def __init__(self, config):

        self.config = config

        self.rand = np.random

        self.rand.seed(self.config.seed)

        self.world = None

    ###
    ##
    #
    def train(self):
        ''' 
        train the agent on whole training set for several epochs, and save the agent
        '''

        # 1. load training train_x (input sequence) and train_y (output sequence)
        train_set = Dataset_loader(self.config.train_x_path, 
                                   self.config.train_y_path, 
                                   self.config)

        valid_set = Dataset_loader(self.config.valid_x_path, 
                                   self.config.valid_y_path, 
                                   self.config)

        test_set = Dataset_loader(self.config.test_x_path, 
                                   self.config.test_y_path, 
                                   self.config)

        train_set.load()

        valid_set.load()

        test_set.load()


        # 2. split the training data to some batches

        train_batches = train_set.get_batches() # batch_size is defined in config

        logger.info('number of train batches: %d for batch_size: %d'%(len(train_batches),3))

        if self.config.pilot == True:

            train_batches = train_batches[:2]

            logger.info('shrinking dataset: number of train batches: %d for batch_size: %d'%(len(train_batches),3))

        # 3. go through the all batches, give the batch to world, update the world one for each batch
            
            # 3.1. before that set up some parameters
        patience = self.config.patience  # wait for at least 10 epoch before stop
        
        train_losses = [] # get a history  of losses

        valid_loss_threshold = np.inf # needed for early stopping
            
        best_valid_loss = np.inf

        done_epoch = 0

        self.world.train()

        update_cnt = 0
        
        max_epochs = self.config.max_epoch

        logger.info(summary(self.world, show_weights=False))

        logger.info("**** Training Begins ****")

        for epoch in range(0, max_epochs):

            for batch_id, train_batch in enumerate(train_batches):

                loss, _, _ = self.world.iterate(inp_data = train_batch, update=True)

                train_losses.append((update_cnt,loss))

                if update_cnt % self.config.print_step == 0:

                    logger.info('epoch: %d/%d, update_cnt: %d/%d: loss: %.2f'%(epoch, max_epochs, (update_cnt+1), max_epochs*len(train_batches),loss))

                
                if update_cnt % self.config.ckpt_step == 0:

                    logger.info(' Evaluating Model (current update_cnt:%d)'%update_cnt)
                    
                    logger.info(' \t on Training Set ')
                    
                    train_loss = self.validate(train_set)

                    logger.info('\t\t loss: %.2f'%train_loss)
                    
                    logger.info('\t on Validation Set ')
                    
                    valid_loss = self.validate(valid_set)

                    logger.info("\t\t loss: %.2f"%valid_loss)

                    valid_inputs, valid_outputs, valid_actions, valid_labels = self.generate(valid_set)

                    self.evaluate(inputs=valid_inputs, 
                                outputs= valid_outputs,
                                predictions=valid_actions,
                                 labels=valid_labels , evaluator = self.config.metric,
                                  threshold= self.config.preview_threshold)

                    logger.info('\t on Test Set ')
                    
                    test_loss = self.validate(test_set)

                    test_inputs, test_outputs, test_actions, test_labels = self.generate(test_set)

                    logger.info("\t\t loss: %.2f"%test_loss)

                    self.evaluate(inputs=test_inputs,
                                 outputs = test_outputs, 
                                    predictions= test_actions,
                                     labels=test_labels,
                                      evaluator = self.config.metric)

                    logger.info('Evaluation Done, update early stoping ')
                
                # update early stopping stats
                if valid_loss < best_valid_loss:

                    if valid_loss <= valid_loss_threshold * self.config.improve_threshold:
                        
                        patience = max(patience, done_epoch * self.config.patient_increase)
                        
                        valid_loss_threshold = valid_loss

                        logger.info('Update patience to %d'%(patience))

                    if self.config.save_model:

                        self.world.save(self.config.session_dir+'best_world.mdl')

                        logger.info('Model Saved.')

                    best_valid_loss = valid_loss
                
                if self.config.early_stop and patience <= epoch:

                    logger.info('!!Early stop due to run out of patience!!')

                    logger.info('Best validation loss %.4f' %(best_valid_loss))

                    return

                update_cnt += 1

    ###
    ##
    #
    def evaluate(self, inputs, outputs, predictions, labels, evaluator, threshold = 1.0):
        '''
            given predictions and labels, returns the output of an evaluation metric
            evaluation metric should be implemented in another file.
            threshold: [0,...,1] is used to decide if we should show the inputs or not
        '''
        if (self.config.preview_pred and self.rand.uniform() > threshold):
            
            for sent_inp, sent_pred, sent_label, sent_output in zip(inputs, predictions, labels, outputs): 
                
                if (self.rand.uniform() > threshold):

                    logger.info('\nx: %s,\ny: %s,\nlabel: %s,\npred: %s'%(sent_inp, sent_output ,sent_label, sent_pred))

        if evaluator=='bleu':

            bleu_1, bleu_2, bleu_4 = bleu(predictions=predictions, labels= labels)

            logger.info('\t\t BLEU_1: %.4f, BLEU_2: %.4f, BLEU_4: %.4f'%(bleu_1, bleu_2, bleu_4))

    ###
    ##
    #
    def test(self):
        '''
        This function should be able to return the outputs on a sample dataset that it gets as input
        such as online input
        '''
        raise NotImplementedError
        return


    ###
    ##
    #
    def generate(self, data_set):
        '''
        This function generates a sequence of actions for each sample in data_set
        '''
        self.world.eval()

        batches = data_set.get_batches()

        if self.config.pilot == True:

            batches = batches[:2]

        inputs = []
        
        outputs = []

        generated_actions = []
        
        gold_labels = []
        
        for batch_id, batch in enumerate(batches):

            _ , action_ids, label_ids = self.world.iterate(inp_data = batch, update=False)


            for i in range(0, len(batch)):
                
                i_x = batch[i][3]
                
                i_y = batch[i][7]

                i_action_ids = action_ids[i,:]
        
                i_actions = self.config.lang_y.index_to_sent(i_action_ids, max_sent_len= self.config.max_dec_len-1)
        
                i_label_ids = list(label_ids[i,:])
        
                i_labels = self.config.lang_y.index_to_sent(i_label_ids, max_sent_len= self.config.max_dec_len-1)
                
                inputs.append(i_x)

                outputs.append(i_y)

                generated_actions.append(i_actions)

                gold_labels.append(i_labels)
        
        self.world.train()

        return inputs, outputs, generated_actions, gold_labels

    ###
    ##
    #
    def validate(self, data_set):
        '''
        we user this function to validate the world on any on training, valid, test set
        This function return loss value on the input dataset
        '''            
        self.world.eval()

        batches = data_set.get_batches()

        if self.config.pilot == True:

            batches = batches[:2]

        losses = []
    
        for batch_id, batch in enumerate(batches):

            loss , actions, labels = self.world.iterate(inp_data=batch, update=False)
            
            losses.append(loss)

        valid_loss = np.sum(losses) / float(len(losses))

        self.world.train()

        return valid_loss


    ###
    ##
    #
    def world_creation(self, world_name):
        
        self.world = World(name= world_name, config = self.config)

    ###
    ##
    #
    def data_prepreation(self):

        if self.config.prep_data:

            for i, lang_name in enumerate(self.config.lang_name):

                prepare_data(lang_name =  lang_name,  
                             voc_size  =  self.config.voc_size[i],
                             train_path= self.config.train_path + lang_name, 
                             valid_path= self.config.valid_path + lang_name, 
                             test_path =  self.config.test_path + lang_name,
                             data_dir  =   self.config.data_dir
                             )

        logger.info('datasets are prepared')

         # 1. define domain of input and output tokens 

        lang_input = Lang()

        lang_output = Lang()

        self.config.lang_x  = lang_input.load(self.config.lang_x_pkl)
    
        self.config.lang_y = lang_output.load(self.config.lang_y_pkl)

        logger.info('lang info loaded')

