# -*- coding: utf-8 -*-

import nltk

import string

import re
num_regex = re.compile(u'^[+-]?[0-9]+\.?[0-9]*$')


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)    
    return tokens

def is_number(token):
    return bool(num_regex.match(token))

def get_stopwords(excluding_pronouns = False):
	#snowball stopwords : http://snowball.tartarus.org/algorithms/english/stop.txt
	# 174 stopwrods in snowball
	snowball_stopwords ="i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing would should could ought i'm you're he's she's it's we're they're i've you've we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't cannot couldn't mustn't let's that's who's what's here's there's when's where's why's how's a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very"
	snowball_stopwords =snowball_stopwords.split()
	stopwords = []
	# we do tokenization on stopwords because we tokenized texts before filtering stopwords
	for sw in snowball_stopwords:
	    for t in tokenize(sw):
	        if t not in stopwords:
	            stopwords.append(t)

	if excluding_pronouns == True:
		pronouns = ['i','me','we','us','you','she','her','him','he','it','they','them','myself','ourselves','yourself','yourselves','himself','herself','itself','themselves']
		stopwords = set(stopwords)-set(pronouns)
		stopwords = list(stopwords)	
	return stopwords

def get_punctuations():
	return [t for t in string.punctuation]

def rm_extra_tokens(sent):
	return sent

def get_MAX_LENS(train_x, dev_x, test_x):
    
    if params['padding_level'] == 'document':
        train_lens = [len(e) for e in train_x]
        dev_lens = [len(e) for e in dev_x]
        test_lens = [len(e) for e in test_x]
    
        params['max_train_len'] = np.max(train_lens)
        params['max_dev_len'] = np.max(dev_lens)
        params['max_test_len'] =  np.max(test_lens)

    
        params['max_doc_len'] = \
        np.max([params['max_train_len'],
                params['max_dev_len'],
                params['max_test_len']])
    
    elif params['padding_level'] == 'sentence':
        
        train_lens = [len(e) for e in train_x] # ns
        dev_lens = [len(e) for e in dev_x]# ns
        test_lens = [len(e) for e in test_x]# ns
        
        
    
        params['max_train_len'] = np.max(train_lens)# max ns
        params['max_dev_len'] = np.max(dev_lens)# max ns
        params['max_test_len'] =  np.max(test_lens)# max ns

    
        params['max_doc_len'] = \
        np.max([params['max_train_len'],
                params['max_dev_len'],
                params['max_test_len']]) # max ns in whole corpus
    
        max_sent_len = 0
        
        for essay in train_x + dev_x + test_x:
            
            for sent in essay:
                
                if len(sent) > max_sent_len:
                
                    max_sent_len = len(sent)
       
        params['max_sent_len'] = max_sent_len




