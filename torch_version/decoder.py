# now implement the decoder module

import torch
import torch.nn as nn
from torch.nn import Module, GRU, Linear, LSTM
import torch.nn.functional as F


class Decoder(Module):
    def __init__(self, embedding_size, hidden_size, output_size, num_layers = 1, dropout = 0.1):
        super(Decoder, self).__init__()
        self.gru = GRU(input_size = embedding_size,
                       hidden_size = hidden_size,
                       num_layers = num_layers,
                       dropout = dropout,
                       bidirectional = False)          
        self.output = Linear(in_features = hidden_size,
                               out_features = output_size)
        #  do orthogonal initialization
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

        
    def forward(self, in_token, h_t, embedding):
        rep = embedding(in_token) # (1, batch_size, emb_dim)
        rnn_out, h_new = self.gru(rep, h_t)
        rnn_out = self.output(rnn_out)
        # rnn_out = F.softmax(rnn_out, dim = 1)
        return rnn_out, h_new
        
        
        
