# to batch data
import itertools
import torch
import numpy as np
from vocab import EOS_token, PAD_token
from util import Util
import random

logger = Util(None)

# auxiliary methods
def sent2id(vocab, sent):
    return [vocab.word2id[x] for x in sent] + [EOS_token]

def safe_sent2id(vocab, sent):
    o = []
    for x in sent:
        if(x not in vocab.word2id):
            o.append(PAD_token)
        else:
            o.append(vocab.word2id[x])
    return o + [EOS_token]

def padding(sents):
    return list(itertools.zip_longest(*sents, fillvalue = PAD_token))


# this feeder is mainly used to train seq2seq model as an autoencoder
class AutoFeeder(object):
    def __init__(self, vocab, sents):
        self.vocab = vocab
        self.sents = sents
        self.counter = 0
        self.epoch_counter = 0

    def random_batch(self, size = 1000):
        batch = self.build(self.random_sents(size))
        return batch

    def random_sents(self, size = 1000):
        sents = [random.choice(self.sents) for i in range(size)]
        return sorted(sents, key = lambda x:len(x), reverse=True)

    def id2sent(self, seq):
        return " ".join([self.vocab.parse(token) for token in seq])
    def safe_sent2id(self, sent):
        return safe_sent2id(self.vocab, sent)

    def next_batch(self, num):
        # @param num: means the current batch size
        batch = None
        if(self.counter + num >= len(self.sents)):
            batch = self.build(self.sents[self.counter:]+self.sents[0:(self.counter + num - len(self.sents))])
            self.counter = self.counter + num - len(self.sents)
            # an epoch finished
            self.epoch_counter += 1
            logger.bar("EPOCH {} FINISHED".format(self.epoch_counter))
        else:
            batch = self.build(self.sents[self.counter: self.counter + num])
            # print(batch) nothing seems wrong with the batch
            self.counter += num
        return batch

    def build(self, raw_batch):
        raw_batch = sorted(raw_batch, key = lambda x:len(x), reverse=True)
        input_batch = raw_batch
        output_batch = raw_batch
        input_seq = [sent2id(self.vocab, sent) for sent in input_batch]
        output_seq = [sent2id(self.vocab, sent) for sent in output_batch]
        
        # input len
        # print([len(seq) for seq in input_seq])
        input_lens = torch.LongTensor([len(seq) for seq in input_seq])
        # padded input seq of size (maxlen, batchsize)   
        input_seq = torch.LongTensor(np.array(padding(input_seq)))
        # max target sequence length, for controlling the generation procedure
        # print(output_seq)
        output_max_len = max([len(seq) for seq in output_seq])
        # print(output_max_len)
        # output sequence padded
        output_seq = np.array(padding(output_seq)).T
        output_mask = torch.ByteTensor((output_seq != PAD_token).astype(np.uint8))
        output_seq = torch.LongTensor(output_seq)
        return input_seq, input_lens, output_seq, output_mask, output_max_len


        
