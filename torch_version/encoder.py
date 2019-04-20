import torch
import torch.nn as nn
from torch.nn import Module, GRU, LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size,num_layers = 1, dropout = 0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.is_bi = False
        self.gru = GRU(input_size = embedding_size,
                       hidden_size = hidden_size,
                       num_layers = num_layers,
                       dropout = dropout,
                       bidirectional = self.is_bi)
        #  do initialization
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, in_seq, in_len, embedding):
        rep = embedding(in_seq)
        packed = pack_padded_sequence(rep, in_len)
        outputs, h_T = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs)
        # since outputs are of 2 * hidden dimensione add it
        # print(outputs.shape)
        if(self.is_bi):
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        else:
            outputs = outputs
        # NOTE: the reason to obtain the outputs here is mainly for the attention mechanism to be implemented. The attention is carried out as a dot product between the encode hidden states and the decoder hidden states
        return outputs, h_T
        


