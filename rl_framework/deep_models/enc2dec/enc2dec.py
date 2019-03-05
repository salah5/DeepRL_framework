# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .encoders import EncoderRNN
from .decoders import DecoderRNN

from ..embeddings.embeddings import Embeddings as emb_layer

import numpy as np
import itertools

class EncDec(nn.Module):

    def __init__(self, config):
    
        super(EncDec, self).__init__()

        self.embed_size = config.embed_size 

        self.enc_inp_size = self.embed_size 

        self.go_id = config.lang_y.word2index[config.lang_y.bos_word]
        
        self.eos_id = config.lang_y.word2index[config.lang_y.eos_word]

        self.pad_idx = config.lang_x.word2index[config.lang_x.pad_word]
        
        self.max_seq_len = config.max_seq_len
        
        self.num_layer = config.num_layer
        
        self.dropout = config.dropout
        
        self.enc_cell_size = config.enc_cell_size
        
        self.dec_cell_size = config.dec_cell_size
        
        self.rnn_cell = config.rnn_cell
        
        self.max_dec_len = config.max_dec_len
        
        self.use_attn = config.use_attn
        
        self.beam_size = config.beam_size
        
        #self.utt_type = config.utt_type
        
        self.bi_enc_cell = config.bi_enc_cell
        
        self.attn_type = config.attn_type
        
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size
        
        self.seed = config.seed

        self.vocab_size = config.voc_size
        
        torch.manual_seed(self.seed)

        self.use_gpu = config.use_gpu
        
        if torch.cuda.is_available():
            
            torch.cuda.manual_seed_all(self.seed)


        self.x_embeddings = emb_layer(seq_size = self.max_seq_len,
                     voc_size=self.vocab_size[0],
                     emb_size=self.embed_size,
                     embeddings=None,
                     pad_idx= self.pad_idx)

        self.dec_emb_layer = emb_layer(seq_size = self.max_seq_len, 
                     voc_size = self.vocab_size[1],
                     emb_size = self.embed_size,
                     embeddings = None,
                     pad_idx = self.pad_idx)  


        self.x_encoder = EncoderRNN(self.enc_inp_size, 
                                    self.enc_cell_size,
                                    dropout_p = self.dropout,
                                    rnn_cell = self.rnn_cell,
                                    variable_lengths = config.fix_batch,
                                    bidirection = self.bi_enc_cell)

        self.decoder = DecoderRNN(vocab_size = self.vocab_size[1], 
                                  max_len = self.max_dec_len,
                                  input_size = self.embed_size, 
                                  hidden_size = self.dec_cell_size,
                                  sos_id = self.go_id, 
                                  eos_id = self.eos_id,
                                  n_layers = 1, 
                                  rnn_cell = self.rnn_cell,
                                  input_dropout_p = self.dropout,
                                  dropout_p = self.dropout,
                                  use_attention = self.use_attn,
                                  attn_size = self.enc_cell_size,
                                  attn_mode = self.attn_type,
                                  use_gpu = self.use_gpu,
                                  embedding = self.dec_emb_layer)


    def forward(self, 
                data_feed, 
                mode, 
                gen_type='greedy', 
                sample_n=1, 
                return_latent=False):

        x, y = data_feed
        
        x_id, x_len, y_id, y_len = x[0], x[1], y[0], y[1] # id: bs * max_x_len, _len: list of lengths

        batch_size = len(x_id)

        x_id = x_id.squeeze(1) # bs * max_seq_len 

        # # context encoder
        x_emb = self.x_embeddings(x_id) # bs * max_seq_len * emb_size

        x_enc, x_last = self.x_encoder(input_var=x_emb, 
                                       input_lengths=x_len) 

       
        if type(x_last) is tuple: # x_last is a tuple in case of LSTM, not gru
            
            x_last = x_last[0].transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        
        else:
            
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)
        
        # # INFO: x_enc: bs * max_seq_len * enc_cell_size; 
        # # INFO: x_last : bs * enc_cell_size;

        # if you are going to use the average of all encoder states        
        # # x_last = torch.mean(x_outs, dim=1)

        # # # context language model
        # # qy_logits = self.q_y(x_last).view(-1, self.config.k)
        # # log_qy = F.log_softmax(qy_logits, qy_logits.dim()-1)

        # # # switch that controls the sampling
        # # sample_y, y_ids = self.cat_connector(qy_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        # # sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        # # y_ids = y_ids.view(-1, self.config.y_size)

        # #initial state of decoder
        dec_init_state = x_last.unsqueeze(0) # 1 * bs* enc_cell_size
        # dec_init_state = self.dec_init_connector(sample_y)

        # # define decoder inputs

        # # everything except the last item
        t_id = y_id[:, 0:-1] # bs * (max_dec_len - 1)

        # decode response
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size=  batch_size,
                                                   inputs = t_id, 
                                                   init_state = dec_init_state,
                                                   attn_context=x_enc,
                                                   mode=mode, 
                                                   gen_type=gen_type,
                                                   beam_size=self.beam_size)        


        return dec_outs, dec_last, dec_ctx
        
        # everything except the first item
        # labels = out_utts[:, 1:].contiguous()


   
        # # compute loss or return results
        # if mode == GEN:
        #     return dec_ctx, labels
        # else:
        #     # RNN reconstruction
        #     nll = self.nll_loss(dec_outs, labels)


        #     results = Pack(nll=nll, reg_kl=None, mi=None, bpr=None)


        #     return results
