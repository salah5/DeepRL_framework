import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.init as init
#%%
        
class Embeddings(nn.Module):
    def __init__(self, 
                 seq_size, 
                 voc_size, 
                 emb_size, 
                 embeddings,
                 pad_idx):
        super(Embeddings, self).__init__()
        
        self.seq_size = seq_size
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.pad_idx = pad_idx
        
        
        self.embed = nn.Embedding(self.voc_size, self.emb_size, padding_idx=self.pad_idx) 

        if embeddings != None:

            embeddings = embeddings[:self.voc_size, :self.emb_size]
            
            self.embed.weight = nn.Parameter(embeddings)
        
    def forward(self, x):
        
        emb = self.embed(x)

        return emb