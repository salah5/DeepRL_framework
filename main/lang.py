# coding: utf-8
import logging

logger = logging.getLogger(__name__)

import dill # we need this package to save the object. It works the same as pickle but works for lambda too
import operator
from collections import defaultdict

from data_processing.data_utils import is_number

class Lang(object):

    def __init__(self, name=None):
        
        self.name = name

      
        self.pad_word = u'<PAD>'
        self.unk_word = u'<UNK>'
        self.num_word = u'<NUM>'
        self.bos_word = u'<BOS>'
        self.eos_word = u'<EOS>'
        self.bod_word = u'<BOD>'
        self.eod_word = u'<EOD>'
        self.bot_word = u'<BOT>'
        self.eot_word = u'<EOT>'


        self.extra_tokens = \
                [
                self.pad_word, 
                self.unk_word, 
                self.num_word,
                self.bos_word,
                self.eos_word,
                self.bod_word,
                self.eod_word,
                self.bot_word,
                self.eot_word
                ]
        
        self.word2index = {}

        self.index2word = {}
  
        for index, word in enumerate(self.extra_tokens):
            self.word2index[word] = index
            self.index2word[index] = word 

        
        self.word2count = defaultdict(lambda: 0)
        
        self.n_words = len(self.index2word) # number of all words
        
        self.embeddings = None


    def build_vocab(self, data, voc_size):
        '''
        data: a list of cleaned tokens; it means you sould clean the data first, but you don't need to replace tokens
        voc_size: the size of vocabulary,e.g. 10,000
        '''
        num_total_words = 0
        
        word2count = {}
        
        for word in data:
            
            num_total_words += 1
        
            if word not in word2count:
            
                word2count[word] = 1
            
            else:
            
                word2count[word] += 1

        if len(word2count) < voc_size:
            
            logger.error(" mismatch in num_voc in data (=%d) and voc_size (=%d)"%(len(word2count),voc_size))
            
        logger.info(" %d total words, %d unique words"%(num_total_words,len(word2count)))
        
        
        sorted_word_freqs = sorted(word2count.items(), 
                            key=operator.itemgetter(1),
                            reverse=True)
                   
        freq_vocabs= [w for w,_ in 
                            sorted_word_freqs[:(voc_size-len(self.word2index))]]
        
        
        for voc in freq_vocabs:
            self.word2index[voc] = self.n_words
            self.index2word[self.n_words] = voc
            self.word2count[voc] =  word2count[voc]
            self.n_words += 1
                             
    def sent_to_index(self, sentence, max_sent_len, padding_place):

        sent_padded= [self.pad_word] * max_sent_len

        max_sent_len = max_sent_len - 2 # because of BOS and EOS   

        trunc = sentence[:max_sent_len]

        trunc = [self.bos_word] + trunc + [self.eos_word]

        
        if padding_place == 'none':

            sent_padded = trunc
        
        if padding_place == 'pre':
                
            sent_padded[-len(trunc):] = trunc
                
        elif padding_place == 'post':
                
            sent_padded[:len(trunc)] = trunc
                    
        output, mask = [], []
        
        num_hit, total, unk_hit = 0.0, 0.0, 0.0
        
        for word in sent_padded:
            
            if is_number(word):
                
                output.append(self.word2index[self.num_word])
                
                num_hit += 1
                
            elif word in self.word2index:
                
                output.append(self.word2index[word])
        
            else:
                
                output.append(self.word2index[self.unk_word])
                
                unk_hit += 1
            

            if word == self.pad_word:
            
                mask.append(0)
            
            else:
            
                mask.append(1)
                
                total += 1

        assert len(output) == len(mask) 

        return output, mask, [int(unk_hit), int(num_hit), int(total)]

    ###
    ##
    #
    def index_to_sent(self, ids, max_sent_len):
        
        sent = []
        
        for word_id in ids:
            
            word = self.index2word[int(word_id)]
            
            sent.append(word)

        return sent

    ###
    ##
    #
    def save(self, path):

        with open(path,'wb') as f:

            dill.dump(self,f)

    def load(self, path):

        with open(path,'rb') as f:

            self = dill.load(f)

        return self

