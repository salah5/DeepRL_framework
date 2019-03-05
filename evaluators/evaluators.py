# @Author  : Mohsen Mesgar
# from collections import Counter
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn import metrics
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
# from laed.utils import get_dekenize, get_tokenize
# from scipy.stats import gmean
# import logging
# from laed.dataset.corpora import EOS, BOS
# from collections import defaultdict



def bleu(predictions, labels):

    refs  = labels

    hyps = predictions  

    # compute corpus level scores
    bleu_4 = bleu_score.corpus_bleu(refs, hyps, 
                                            smoothing_function=SmoothingFunction().method1)

    bleu_1 = bleu_score.corpus_bleu(refs, hyps, 
                                            smoothing_function=SmoothingFunction().method1, weights=(1.0, 0.0, 0.0, 0.0))
    
    bleu_2 = bleu_score.corpus_bleu(refs, hyps, 
                                            smoothing_function=SmoothingFunction().method1, weights=(0.5, 0.5, 0.0, 0.0))

    return bleu_1, bleu_2, bleu_4




