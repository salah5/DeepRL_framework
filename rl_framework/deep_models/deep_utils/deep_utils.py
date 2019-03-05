# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
import numpy as np

ref_scores_dtype = 'int32'

INT = 0
LONG = 1
FLOAT = 2


def get_ref_dtype():
	return ref_scores_dtype

###
##
#
def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var
###
##
#
def summary(model, show_weights=True, show_parameters=True):
    """
    Summarizes torch model by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params
        # and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = summary(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        total_params += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ') Total Parameters={}'.format(total_params)
    return tmpstr
###
##
#
import math
def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))


###
##
#
def np2var(inputs, dtype, use_gpu):
    if inputs is None:
        return None
    return cast_type(Variable(torch.from_numpy(inputs)), dtype, use_gpu)
###
##
#
def convert_seq_to_indices(data_x, lang, max_seq_len):
    
    if params['padding_level'] == 'document':
        
        output_x = []
        output_mask = []
        
        unk_hit = 0.
        num_hit =0.
        total = 0.
        
        for seq in data_x:
            out_seq,out_mask,(unk,num,tot) =  lang.indicies_from_sentence(seq ,max_seq_len, params['padding_place']) # does padding and indexing
            output_x.append(out_seq)
            output_mask.append(out_mask)
            unk_hit += unk
            num_hit += num
            total += tot
            
        logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    
    elif params['padding_level'] == 'sentence':

        output_x = []
        output_mask = []
        
        unk_hit = 0.
        num_hit =0.
        total = 0.
        
        for essay in data_x:
            
            essay_output = []
            essay_mask = []
            
            for seq in essay:
                out_seq,out_mask,(unk,num,tot) = \
                    lang.indicies_from_sentence(seq, max_seq_len,params['padding_place']) # do padding and indexing
                
                unk_hit += unk
                num_hit += num
                total += tot
                essay_output.append(out_seq)
                essay_mask.append(out_mask)
                
            output_x.append(essay_output)
            output_mask.append(essay_mask)
            
        logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))

    return output_x, output_mask
