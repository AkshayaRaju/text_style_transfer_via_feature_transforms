 # au autoencoder based on seq2seq, which is used to explore the space of sentence embeddings learned by seq2seq. The ultimate goal is find what style is.

import torch
import numpy as np
from torch.nn import Module, NLLLoss, Embedding
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.optim import Adam
from feeder import AutoFeeder
from encoder import Encoder


from decoder import Decoder
from vocab import BOS_token, PAD_token
from util import Util
from preproc import normalize_str
import random


class SeqTranslator(Module):
    """
    This model will always be teacher-forced
    """
    def __init__(self, vocab, sents, embedding_size, num_layers = 1, clip = 50.0, use_pretrained_embedding = True, embedding_ = None):
        super(SeqTranslator, self).__init__()
        self.vocab = vocab
        self.feeder = AutoFeeder(vocab, sents)
        
        if(not use_pretrained_embedding):
           self.embedding = Embedding(vocab.vocab_size,
                             embedding_dim = embedding_size,
                             padding_idx = PAD_token)
        else:
            print("Use pretrained embedding")
            self.embedding = Embedding.from_pretrained(torch.tensor(embedding_, dtype=torch.float32),
                                                       freeze = True)
            # TODO: set the embedding untrainable
            for param in self.embedding.parameters():
                param.requires_grad = False

        
        

        self.num_layers = num_layers
        self.encoder = Encoder(embedding_size = embedding_size,
                               hidden_size = embedding_size,
                               num_layers = self.num_layers,
                               dropout = 0.1)
        # note we do not need attention because we would like the system get all information fused in the intermediate representation
        self.decoder = Decoder(embedding_size = embedding_size,
                               hidden_size = embedding_size,
                               output_size = vocab.vocab_size,
                               num_layers = self.num_layers,
                               dropout = 0.1)
        self.clip = clip
        self.decoder_lr_ratio = 1.0
        self.teacher_force_ratio = 0.5

    # return the representation
    def forward(self, in_seq, in_len):
        _, h_T = self.encoder(in_seq, in_len, self.embedding)
        return h_T

      # compute the masked nll loss 
    def mask_ce_loss(self, decoder_out, tar_seq, mask):
        return F.cross_entropy(decoder_out, tar_seq, reduction = 'none').masked_select(mask).mean()

    def train_step(self, in_seq, in_len, out_seq, out_mask, out_maxlen,
                   encoder_opt, decoder_opt, device):
        ## define a one-step training logic
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()

        ## move the data onto the specified device
        in_seq = in_seq.to(device)
        in_len = in_len.to(device)
        out_seq = out_seq.to(device)
        out_mask = out_mask.to(device)

        ## do a randomization to decide whether teacher forcing or not
        out_maxlen = out_maxlen
        # generate the probs for subsequent loss computation
        probs = torch.zeros(in_seq.size(1), self.vocab.vocab_size, out_maxlen+1).to(device = device)
        current_token = torch.LongTensor([BOS_token]*in_seq.size(1)).to(device=device)        

        _, h_T = self.encoder(in_seq, in_len, self.embedding)
        h_T = h_T[:self.num_layers]
       
        # is_teacher_force = (np.random.rand() < self.teacher_force_ratio)
        # choose the first word
            
        for i in range(out_maxlen+1):
            # print(current_token)
            # toss a coin at each step 
            is_teacher_force = (np.random.rand() < self.teacher_force_ratio)
            out, h_T = self.decoder(current_token.unsqueeze(0), h_T, self.embedding)
            if(is_teacher_force and i < out_maxlen):
                current_token = out_seq[:, i]
            else:
                current_token = torch.argmax(out, dim = 2)
                current_token = current_token.squeeze()  
            # print(current_token.shape)
            probs[:, :, i] = out
        
        loss = self.mask_ce_loss(probs[:, :, 1:], out_seq, out_mask)

        loss.backward()

        # do gradient clipping
        clip_grad_norm_(self.encoder.parameters(), self.clip)
        clip_grad_norm_(self.decoder.parameters(), self.clip)

        # do descent a.t. the gradient
        self.encoder_opt.step()
        self.decoder_opt.step()

        return loss.item()

    def _parse_sentence(self, seq):
        return self.vocab.parse(seq)

    def greedy_respond(self, in_seq, in_len, out_maxlen, device, parsed = True):
        in_seq = in_seq.to(device)
        in_len = in_len.to(device)
        # do greedy decoding and append to the decoder_out

        _, h_T = self.encoder(in_seq, in_len, self.embedding)
        h_T = h_T[:self.num_layers]
        return self._sentence_decode(h_T, device, out_maxlen, parsed)
        

    # the training subroutine
    def train(self, device, dev_set, batch_size = 4, lr = 0.001, max_epoch = 10):
        # move the models to gpu
        self = self.to(device)
          
        self.encoder_opt = Adam(self.encoder.parameters(), lr = lr)
        self.decoder_opt = Adam(self.decoder.parameters(), lr = lr * self.decoder_lr_ratio)
        # sample_count = 0
        batch_counter = 0
        accumulated_loss = 0.0
        LOG_PERIOD = 100
        cached_epoch_counter = self.feeder.epoch_counter
        while(self.feeder.epoch_counter <= max_epoch):
            batch = self.feeder.next_batch(batch_size)
            batch_loss = self.train_step(*batch, self.encoder_opt, self.decoder_opt, device)
            batch_counter += 1
            accumulated_loss += batch_loss
            if(batch_counter % LOG_PERIOD== 0):
                print("# of BATCH: {} Loss: {:.3f} ratio: {:.4f}".format(batch_counter, accumulated_loss / LOG_PERIOD, self.teacher_force_ratio))
                accumulated_loss = 0.0
            if(cached_epoch_counter != self.feeder.epoch_counter):
                cached_epoch_counter = self.feeder.epoch_counter
                # show some samples
                print("Correct Recovery Ratio: {:.3f}".format(self.recovery_accuracy(device)))
                # self.test_with_samples(device, [random.choice(dev_set) for i in range(4)])
                self.teacher_force_ratio -= 0.05
                

    def recovery_accuracy(self, device, verbose = True):
        self = self.to(device)
        cpu = torch.device("cpu")
        val_size = 100
        val_batch = self.feeder.random_batch(size = val_size)
        input_seq, input_lens, _, _, out_maxlen = val_batch
        input_seq = input_seq.to(device)
        input_lens = input_lens.to(device)
        h_T = self(input_seq, input_lens)
        
        out = self._sentence_decode(h_T.squeeze(0), device, out_maxlen, parsed = False)
        
        # print(input_seq.numpy().T.shape)
        # print(out.shape)
        # print(input_seq.numpy().T == out)
        correct_token_ratio = np.sum(input_seq.to(cpu).numpy().T == out)/float(val_size * out_maxlen)
        # output sample sentences
        if(verbose):
            for i in range(4):
                print("- {} - {}".format(self._parse_sentence(input_seq[:, i].to(cpu).numpy())
                                         ,self._parse_sentence(out[i, :])))
        # print(out)
        return correct_token_ratio
        
        


    # given a batch of sentence, return the embeddings (i.e. the hidden code)
    def _sentence_embedding(self, sents, device):
        self = self.to(device)
        input_seq, input_lens, _, _, _ = self.feeder.build(sents)
        input_seq = input_seq.to(device)
        input_lens = input_lens.to(device)
        cpu = torch.device("cpu")
        h_T = self(input_seq, input_lens)
        return h_T.to(cpu).detach().squeeze(0).numpy()

    def _sentence_decode(self, h_T, device, out_maxlen = 10, parsed = True):
        self = self.to(device)
        cpu = torch.device("cpu")
        # do greedy decoding and append to the decoder_out
        current_token = torch.LongTensor([BOS_token]*h_T.size(0)).to(device=device)
        decoder_out = []
        
        if(h_T.dim() == 2):
            h_T = h_T.unsqueeze(0)
        # h_T = h_T.unsqueeze(0)
        
        for i in range(out_maxlen+1):
            out, h_T = self.decoder(current_token.unsqueeze(0), h_T, self.embedding)
            out = out.squeeze()
            current_token = torch.argmax(out, dim = 1)
            decoder_out.append(current_token.to(cpu).detach().numpy())
        decoder_out = np.array(decoder_out[1:]).T
        if(parsed): decoder_out = [self.vocab.parse(tokens) for tokens in decoder_out]
        return decoder_out
        

    # obtain embeddings of a random batch from the corpora
    def encode(self, device, sents = None):
        if(sents == None):
            sents = self.feeder.random_sents(size = 10)
        else:
            sents = sorted(sents, key = lambda x:len(x), reverse=True) # essential
        return sents, self._sentence_embedding(sents, device)

    def decode(self, features, device):
        h_T = torch.tensor(features, dtype = torch.float32).to(device)
        return self._sentence_decode(h_T, device)
        
        
        
    ## save model
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

        
        
        
    
    
