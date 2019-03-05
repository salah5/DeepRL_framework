
# -*- coding: utf-8 -*-

### import packages
##
#
from data_processing.data_utils import tokenize,rm_extra_tokens

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

import codecs
import pickle

from main.lang import Lang

### process loaded corpus for further processing
##
#
def prepare_data(lang_name, 
                voc_size, 
				train_path, valid_path, test_path,
				data_dir):
    '''
    we generate a lang from train, valid, and test data
    '''
    # get a list of sentences, each tokenized [[t0,t1,...,t_n],...,[t0,t1,...,t_m]
    train_sents = load_data(train_path) 
    valid_sents = load_data(valid_path)
    test_sents = load_data(test_path)
    logger.info('datasets loaded')

    all_tokens = []
    for sent in train_sents+valid_sents+test_sents:
        all_tokens.extend(sent)

    lang = Lang(lang_name)

    lang.build_vocab(all_tokens, voc_size)

    logger.info('voc_size: %d'%voc_size)

    logger.info('20 first words: %s'%[lang.index2word[i] for i in range(0,20)])

    logger.info('nuumber of extracted vocabulary: %d'%len(lang.word2index))

    # save lang
    lang.save(data_dir+'lang.%s.pkl'%lang.name)

    logger.info('language is saved in %s'%(data_dir+'lang.%s.pkl'%lang.name))

    #save train, valid , test in the word format
    with open(data_dir+'train.%s'%lang_name,'w') as f:
        f.write('\n'.join([' '.join(s) for s in train_sents]))

    with open(data_dir+'valid.%s'%lang_name,'w') as f:
        f.write('\n'.join([' '.join(s) for s in valid_sents]))

    with open(data_dir+'test.%s'%lang_name,'w') as f:
        f.write('\n'.join([' '.join(s) for s in test_sents]))

    logger.info('revised train, valid, test for language:%s saved in %s'%(lang_name,data_dir))

    # we then load lang and data to conver sentences to index

###load_original_corpus
##
#

def load_data(file_path):
    '''
    load original dataset, tokenize it, and clean tokens there
    output of is a list of sentences [[t0,t1,t2,...,t9],[t0,t1,t2,...,t9],...,[t0,t1,t2,...,t9]]

    '''
    
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        
        i = 0
        
        sents = []

        for line in input_file:
            
            sent = line.strip()
            
            sent = sent.lower()

            sent = tokenize(sent)

            sent = rm_extra_tokens(sent)

            sents.append(sent)
                 
    return sents



### main is just for test
##
#
if __name__=="__main__":

	prepare_data(lang_name='en',  
                 train_path='./prep/train.de-en.en', 
                 valid_path='./prep/valid.de-en.en', 
                 test_path='./prep/test.de-en.en')
