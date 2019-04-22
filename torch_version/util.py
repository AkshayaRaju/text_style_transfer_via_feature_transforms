import pickle
import os
import csv
import torch
from preproc import normalize_str
from vocab import Vocab
from gensim.models import word2vec
import numpy as np
from functools import reduce
import random
from nltk.translate import bleu_score

class Util(object):
    def __init__(self, path):
        if(path == None):
            self.save_path = None
        else:
            self.save_path = os.path.join(path, "save") 
    
    def dump(self, data, name):
        f = open(os.path.join(self.save_path, name), 'wb')
        pickle.dump(data, f)
        f.close()
        return

    def load(self, name):
        f = open(os.path.join(self.save_path, name), 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def bar(self, line):
        print("{}{}{}".format('='*10, line, '='*10))

UTIL = Util(None)

def _build_vocab_(sents, min_count, name):
    vocab = Vocab(name)
    for s in sents:
        vocab.add_sentence(s)
        
    # filter out infrequent word types
    vocab.trim(min_count)
    return vocab

# build vocabulary
def build_corpus_basic(paths_dict, max_len, min_count, vocab_ = None, has_vocab = False, name = "anonymous", min_len = 3):
    """
    @param paths: a dictionary [(ds_name: path)]
    @param max_len: the maximal length a sentence coule be
    @param min_count: the maximal count to dub a word as rare 
    """
    
    # note we should not use <= here because of the EOS token
    test_len_p = lambda s: len(s) < max_len and len(s) >= min_len
    # load data
    sents_dict = dict()
    UTIL.bar("FIRST CLEAN")
    original_sample_size = 0
    for k in paths_dict:
        f = open(paths_dict[k], 'r', encoding='utf-8')
        sents_dict[k] = [row for row in f]
        original_sample_size += len(sents_dict[k])
        sents_dict[k] = [normalize_str(s) for s in sents_dict[k]]
        sents_dict[k] = [s.split(" ") for s in sents_dict[k]]
        sents_dict[k] = list(filter(test_len_p, sents_dict[k]))
        f.close()
    
        
    sents = list(reduce(lambda x, y: x+y, [s for s in sents_dict.values()]))
    # original_sample_size = len(sents)
    UTIL.bar("BUILD VOCABULARY")
    # construct a common vocabulary
    if(not has_vocab):
        vocab = _build_vocab_(sents, min_count, name)
    else:
        vocab = vocab_

    # re-filter the sents
    del sents
    UTIL.bar("REPLACE RARE WITH UNK")
    for k in sents_dict:
        for i, sent in enumerate(sents_dict[k]):
            for j, token in enumerate(sent):
                if(token not in vocab.word2id):
                    sents_dict[k][i][j] = '<UNK>'

    UTIL.bar("SHUFFLE")
    sents = list(reduce(lambda x, y: x+y, [s for s in sents_dict.values()]))
    random.shuffle(sents)
    
    print("vocabulary size: {}".format(vocab.vocab_size))
    print("{} ratio sentence trimmed.".format(1.0 - float(len(sents))/original_sample_size))
    print("Original: {} Current: {}".format(original_sample_size, len(sents)))
    
    return vocab, sents, sents_dict




def load_embeddings(vocab, path, embedding_size):
    dim_e = embedding_size
    embeddings = np.zeros(shape = (vocab.vocab_size, dim_e+4), dtype = np.float32)
    #add embs for go eos unk pad
    embeddings[0,dim_e]=1
    embeddings[1,dim_e+1]=1
    embeddings[2,dim_e+2]=1
    embeddings[3, dim_e+3]=1
    
    dim_e += 4
    ## open the file
    f = open(path, 'r')
    vec_dict = dict()
    reader = csv.reader(f, delimiter = ' ')

    HIT = 0
    for row in reader:
        vec_dict[row[0]] = np.array([float(x) for x in row[1:]]+[0,0,0,0])
    for i in range(vocab.vocab_size):
        if(vocab.id2word[i] in vec_dict):
            embeddings[i, :] = vec_dict[vocab.id2word[i]]
            HIT += 1
    print("HIT RATIO:{}".format(float(HIT)/vocab.vocab_size))
    return embeddings, dim_e

def train_embeddings(sents, embedding_size, path, min_count = 3):
    model = word2vec.Word2Vec(iter = 5, size = embedding_size, min_count = min_count, workers = 10)
    model.build_vocab(sents)
    print(model.corpus_count)
    model.train(sents, total_examples=model.corpus_count,epochs=model.epochs)
    model.wv.save_word2vec_format(path)






def write_lines(path, lines):
    f = open(path, 'w+')
    for line in lines:
        f.write(line + '\n')
    f.close()



if __name__ == "__main__":
    # let us test on the yelp data
    path = "data/yelp/sentiment.trainx.0"
    _, sents = build_vocab_basic(path, 10, 3, "yelp")
    for i in range(4):
        print(sents[i])
