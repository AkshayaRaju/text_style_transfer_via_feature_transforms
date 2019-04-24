# implement the echo routine as a seq2seq model

import numpy as np
from numpy.linalg import eig
import torch
import os
import sys
from functools import reduce
import random
from torch.nn import Module
from torch.nn.functional import l1_loss, mse_loss
from torch.optim import SGD, Adam

# PATH = "/home/mlsnrs/data/pxd/text_style_transfer_via_feature_transforms/data/yelp/"
# PATH = "/home/morino/code/text_style_transfer_via_feature_transforms/data/fake_yelp/"

PREFIX = "/home/morino/code/"
NAME = "YELP"

PATH = PREFIX + "text_style_transfer_via_feature_transforms/data/{}/".format(NAME.lower())


VOCAB_BUILD = False
MAX_LEN = 15
MIN_COUNT = 3
MAX_EPOCH = 10
EMBEDDING_SIZE = 500
EMBEDDING_PATH = "save/emb_{}.txt".format(EMBEDDING_SIZE)
BATCH_SIZE = 200
USE_PRETRAINED = False
SAVE_PATH = "save/{}_model_{}".format(NAME, EMBEDDING_SIZE)
TRAIN = True
DEVICE = torch.device('cuda:0')

if(NAME == "SHAKE"):
    DATASET = {
        "POS": os.path.join(PATH, "train.original.nltktok"),
        "NEG": os.path.join(PATH, "train.modern.nltktok")
    }
    TEST = {
        "TEST_POS" :  os.path.join(PATH, "test.original.nltktok"),
        "TEST_NEG" :  os.path.join(PATH, "test.modern.nltktok")
    }
elif(NAME == "YELP"):
    DATASET = {
        "POS": os.path.join(PATH, "sentiment.train.1"),
        "NEG": os.path.join(PATH, "sentiment.train.0")
    }
    TEST = {
        "TEST_POS" :  os.path.join(PATH, "sentiment.test.1"),
        "TEST_NEG" :  os.path.join(PATH, "sentiment.test.0")
    }


from util import build_corpus_basic, Util, load_embeddings, train_embeddings, write_lines
# from feeder import AutoFeeder
from seq_auto import SeqTranslator


UTIL = Util(PATH)




    

## CORE TRANSFER FUNCTION
def _transfer(vec1,vec2):#seq_num * dim_h
    m1=np.mean(vec1,axis=0)
    m2=np.mean(vec2,axis=0)
    vec1=vec1-m1
    vec2=vec2-m2

    
    vec1=np.transpose(vec1)#dim_h * seq_num 

    n = vec1.shape[1]
    covar1=np.dot(vec1,vec1.T)/(n-1)
    print(covar1.shape)
    vec2=np.transpose(vec2)
    covar2=np.dot(vec2,vec2.T)/(n-1)
    #print(covar2)
    thresh = 0.0001

    evals1,evecs1 = eig(covar1)
    eig_s = evals1
    eig_s = np.array(list(filter(lambda x:x > thresh, evals1)))
    eig_s = np.power(eig_s, -1/2)

    
    evecs1 = evecs1[:,:len(eig_s)]


    # print(eig_s.shape)
    # print(evecs1.shape)
    # print(vec1.shape)

    evals2,evecs2 = eig(covar2)
    eig_t = evals2
    eig_t = np.array(list(filter(lambda x:x > thresh, evals2)))
    eig_t = np.power(eig_t, 1/2)
    
    evecs2 = evecs2[:,:len(eig_t)]
    
    
    # print(evals2)
    # evals2=np.diag(np.power(np.abs(evals2),1/2))
    fc = evecs1 @ np.diag(eig_s) @ evecs1.T @ vec1 # dim_h * seq_num
    # print(fc.shape)
    fcs = evecs2 @ np.diag(eig_t) @ evecs2.T @ fc # dim_h * seq_num
    
    # fc=np.dot(np.dot(np.dot(evecs1,evals1),evecs1.T),vec1)
    # fcs=np.dot(np.dot(np.dot(evecs2,evals2),evecs2.T),fc)
    return fcs.T+m2


class Mat(Module):
    def __init__(self, channel_num, sample_num):
        super(Mat, self).__init__()

        self.channel_num = channel_num
        self.sample_num = sample_num
        self.A = torch.nn.Parameter(torch.tensor(np.random.randn(self.channel_num, self.channel_num)*0.1, dtype = torch.float32))

    # do embedding level style transfer with the learned matrix
    def forward(self, x):
        return self.A.matmul(x)
    
    def loss(self, F, G):
        transfered_gram = (self.A.matmul(F).matmul(F.transpose(0, 1)).matmul(self.A.transpose(0, 1)))/(self.sample_num - 1)
        original_gram = G.matmul(G.transpose(0,1))/(self.sample_num - 1)
        var_loss = mse_loss(transfered_gram, original_gram, reduction = "mean")
        content_loss = l1_loss(self.A.matmul(F), F, reduction = 'mean')
        # print(content_loss)
        # print(var_loss)
        la = 0.005
        return var_loss + la * content_loss


# a learning way for style transfer 
def learning_transfer(content_sents, style_sents, translator, device = DEVICE):
    content_sents, vec1 = translator.encode(device, content_sents)
    _, vec2 = translator.encode(device, style_sents)
    n = vec1.shape[1]
    m1 = np.mean(vec1, axis = 0)
    m2 = np.mean(vec2, axis = 0)
    F = (vec1 - m1).T
    G = (vec2 - m2).T
    F = torch.tensor(F).to(device)
    G = torch.tensor(G).to(device)
    # to minimize ||AFA.T - G||
    mat = Mat(F.shape[0], F.shape[1]).to(device)
    solver = Adam(mat.parameters(), lr = 0.01)
    tol = 0.0001
    for i in range(1000):
        solver.zero_grad()
        loss = mat.loss(F, G)
        loss.backward()
        solver.step()
        if(loss.item() < tol):
            break
        print("Gram Loss: {}".format(loss.item()))
    cpu = torch.device('cpu')
    F_prime = mat(F).to(cpu).detach().numpy().T + m2
    out = translator.decode(F_prime, device)
    for i, sent in enumerate(out):
        print("Sample A.{}. {} -> {}".format(i+1, " ".join(content_sents[i]), out[i])) # simply output
    return mat
    
    
    
    

# user interface
def interpolation(sent1, sent2, emb1, emb2, translator, device = DEVICE):
    # do linear interpolation
    print("SENT_x: {} SENT_y {}".format(sent1, sent2))

    interp_embs = []
    
    for t in np.linspace(0, 1.0, 20):
        interp_embs.append(emb1 * (1-t) + emb2 * t)
    interp_embs = np.array(interp_embs)
    out = translator.decode(interp_embs, device)
    for i, t in enumerate(np.linspace(0, 1.0, 20)):
        print("node {:.3f} sentence: {}".format(t, out[i]))
        


        
def transfer(content_sents, style_sents, translator, device = DEVICE, has_cipher = False, cipher = None, reverse = True):
    content_sents, content_embs = translator.encode(device, content_sents)
    _, style_embs = translator.encode(device, style_sents)
    # do transfer
    content_embs = _transfer(content_embs, style_embs)
    out = translator.decode(content_embs, device)
    # accuracy = 0.0
    correct = 0
    total = 0
    decipher_result = [] # temp
    
    for i, sent in enumerate(out):
        acc = 0
        sent = sent.split(' ')
        if(not has_cipher):
            print("Sample B.{}. {} -> {}".format(i+1, " ".join(content_sents[i]), out[i])) # simply output
        else:
            for j in range(min(len(content_sents[i]), len(sent))):
                if(not reverse):
                    if('_' in sent[j] and sent[j].split('_')[-1] == content_sents[i][j]):
                        acc += 1
                        correct += 1
                else:
                    if('_' in content_sents[i][j] and sent[j] == content_sents[i][j].split('_')[-1]):
                        acc += 1
                        correct += 1                    
            total += 1
            
    if has_cipher: print("Ratio of Correct Token: {}".format(float(correct) / total))
    return out

            
            
            
        

def cipher_gen(sents, vocab):
    # generate ciphered sents and the corresponding reference dictionary
    SKIP = 3 # also with the UNK label
    lst = list(range(SKIP, vocab.vocab_size))
    random.shuffle(lst)
    cipher = dict()
    decipher = dict()
    # cipher["<UNK>"] = "<UNK>"
    for i, j in enumerate(lst):
        cipher[vocab.id2word[SKIP + i]] = vocab.id2word[j]
        decipher[vocab.id2word[j]] = vocab.id2word[SKIP + i]
    # do the ciphering
    ciphered_sents = [['{}_{}'.format(cipher[token], token) for token in sent] for sent in sents]
    return ciphered_sents, cipher, decipher





# generate the decipher task data
def decipher_main():
    UTIL.bar("BUILD DATASET")
    PREFIX = 'decipher/'
    DATASET = {
         "POS": os.path.join(PATH, "sentiment.train.1")
    }
    vocab, sents, sents_dict = build_corpus_basic(DATASET,
                                     max_len = MAX_LEN,
                                     min_count = MIN_COUNT,
                                     name = DATASET)
    ciphered_sents, cipher, decipher = cipher_gen(sents, vocab)
    print("SAMPLE CIHPER:{} -> {}".format(sents[0], ciphered_sents[0]))
    # UTIL.bar("TRAIN EMBEDDING")
    # train_embeddings(sents, EMBEDDING_SIZE, EMBEDDING_PATH, MIN_COUNT)
    UTIL.dump(vocab, PREFIX +"vocab.p")
    UTIL.dump(decipher, PREFIX + "decipher.p")
    UTIL.dump(cipher, PREFIX + "cipher.p")
    f = open(os.path.join(PATH, "sentiment.train.r"), 'w+')
    for line in ciphered_sents:
        f.write(" ".join(line) + '\n')
    f.close()

    


def main():
    global EMBEDDING_SIZE
    if VOCAB_BUILD:
        UTIL.bar("BUILD DATASET")
        vocab, sents, sents_dict = build_corpus_basic(DATASET,
                                         max_len = MAX_LEN,
                                         min_count = MIN_COUNT,
                                         name = DATASET)
        UTIL.bar("TRAIN EMBEDDING")
        train_embeddings(sents, EMBEDDING_SIZE, EMBEDDING_PATH, MIN_COUNT, iter_ = 200)
        UTIL.dump(vocab, "vocab.p")
        UTIL.dump(sents_dict, "sents_dict.p")
        # sys.exit(0)
    else:
        UTIL.bar("LOAD DATASET")
        vocab, sents_dict = UTIL.load("vocab.p"), UTIL.load("sents_dict.p")

        # recover sents from sents_dict
        sents = list(reduce(lambda x, y: x+y, [s for s in sents_dict.values()]))
        # add the word embedding loading code
        UTIL.bar("LOAD EMBEDDING")

    print("TRAIN POS SIZE:{}".format(len(sents_dict["POS"])))
    print("TRAIN NEG SIZE:{}".format(len(sents_dict["NEG"])))
    # cipher = UTIL.load("decipher/cipher.p")
    cipher = None
    if(USE_PRETRAINED):
        embeddings, EMBEDDING_SIZE = load_embeddings(vocab, EMBEDDING_PATH, EMBEDDING_SIZE)

    # load dev set
    _, test_sents, test_dict = build_corpus_basic(TEST,
                                      max_len = MAX_LEN,
                                      min_count = MIN_COUNT,
                                      vocab_ = vocab,
                                      has_vocab = True,
                                         name = TEST)

    translator = SeqTranslator(vocab, sents, EMBEDDING_SIZE,
            num_layers = 1,
            use_pretrained_embedding = USE_PRETRAINED,
                    embedding_ = embeddings if USE_PRETRAINED else None)

    
    print("DATASET SIZE: {}".format(len(sents)))
    
    
    print("TEST POS SIZE: {}".format(len(test_dict["TEST_POS"])))
    print("TEST NEG SIZE: {}".format(len(test_dict["TEST_NEG"])))

    try:
        translator.load(SAVE_PATH)
    except RuntimeError:
        UTIL.bar("LOAD MODEL FAIL. START FROM SCRATCH")
    except FileNotFoundError:
        UTIL.bar("NO SUCH MODEL. CREATE")
    
    print(translator.recovery_accuracy(device = DEVICE))

    if(TRAIN):
        translator.train(device = DEVICE,
                 dev_set = test_sents,
                 batch_size = BATCH_SIZE,
                         max_epoch = MAX_EPOCH)

    cpu = torch.device('cpu')

    sents, embs = translator.encode(device = DEVICE)
    print(embs.shape)
    
    translator.save(SAVE_PATH)
    # out = translator.decode(embs[:, :], device)
    # interpolation(sents[0], sents[1], embs[0, :], embs[1, :], translator, device = DEVICE)

    SAMPLE_SIZE = 1000
    pos_sent = [random.choice(test_dict['TEST_POS']) for i in range(SAMPLE_SIZE)]
    neg_sent = [random.choice(test_dict['TEST_NEG']) for i in range(SAMPLE_SIZE)]
    
    deciphered = transfer(pos_sent, neg_sent, translator, DEVICE, has_cipher = False, cipher = cipher, reverse = True)
    learning_transfer(pos_sent, neg_sent, translator, DEVICE)
    neg_sent = [" ".join(seq) for seq in neg_sent]

    # write_lines(os.path.join(PATH, 'hypothesis.txt'), deciphered)
    # write_lines(os.path.join(PATH, 'reference.txt'), neg_sent)

    # # calculate the bleu value on the fly
    # from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    # chencherry = SmoothingFunction()

    # hypothesis_tokens = [line.split(' ') for line in deciphered]
    # reference_tokens = [line.split(' ') for line in neg_sent]
    # print(len(reference_tokens))
    # print(len(hypothesis_tokens))
    # bleu = corpus_bleu([[line.split(' ') for line in neg_sent[:1000]] for i in range(SAMPLE_SIZE)], hypothesis_tokens, smoothing_function = chencherry.method4)
    
    # print("BLEU: {}".format(bleu * 100))
    
    

if __name__ == '__main__':
    # decipher_main()
    main()
