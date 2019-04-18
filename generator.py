## to wrap the E-D model into a module called SeqGenerator

import numpy as np
import tensorflow as tf
## use rnn module for implementing a lstm soft decoder
import tensorflow.contrib.rnn as rnn


from keras.layers import Input, LSTM, Dense, Embedding

#from util import *

from options import FLAGS

## helper functions

def softmax_word(dropout_rate, proj_W, proj_b, embedding, gamma):
    def loop_func(output):
        #output = tf.nn.dropout(output, keep_prob= 1-dropout_rate)
        logits = tf.matmul(output, proj_W) + proj_b  #batch_size * vocab_size
        #prob = tf.nn.softmax(tf.nn.relu(logits) / gamma)
        prob = tf.nn.softmax(logits / gamma)         #batch_size * vocab_size
        # use the soft combination of embeddings   
        inp = tf.matmul(prob, embedding)            #batch_size * dim_e
        return inp, prob, logits
    return loop_func

def softmax_word_sample(dropout_rate, proj_W, proj_b, embedding):
    def loop_func(output):
        #output = tf.nn.dropout(output, keep_prob= 1-dropout_rate)
        logits = tf.matmul(output, proj_W) + proj_b  #batch_size * vocab_size
        prob = tf.nn.softmax(logits)  #batch_size * vocab_size
        #sample = tf.reshape(tf.random.categorical(tf.log(logits),1),[-1])
        sample = tf.argmax(prob,axis=-1)
        nxt_inp = tf.nn.embedding_lookup(embedding,sample) #batch_size * dim_e
        return nxt_inp,sample,prob,logits
    return loop_func


def rnn_decode(h, inp, length, cell, softmax):
    h_seq, probs_seq, logits_seq = [], [], []
    for t in range(length+1):#with the eos token
        h_seq.append(tf.expand_dims(h, 1))
        output, h = cell(inp, h)
        inp, probs, logits = softmax(output)
        probs_seq.append(tf.expand_dims(probs, 1))
        logits_seq.append(tf.expand_dims(logits, 1))
    return tf.concat(h_seq, 1), tf.concat(probs_seq, 1), tf.concat(logits_seq, 1)

#以prob的概率使用来自采样的上一个样本，当检测生成效果时，prob=1
def mle_decode(h,seq_inp,length,cell,softmax,schedule_prob):
    prob_seq,sample_seq,logit_seq=[],[],[]
    inp_sam = seq_inp[:,0,:]
    
    #sam_pro=tf.constant([prob]*FLAGS.batch_size)
    for t in range(length+1):
        inp=tf.where(tf.random_uniform([FLAGS.batch_size])<schedule_prob,inp_sam,seq_inp[:,t,:])
        #inp=seq_inp[:,t,:]
        output,h = cell(inp,h)
        inp_sam,sample,prob,logit=softmax(output)
        prob_seq.append(tf.expand_dims(prob,1))
        sample_seq.append(tf.expand_dims(sample,1))
        logit_seq.append(tf.expand_dims(logit,1))
       
    return tf.concat(prob_seq,1),tf.concat(sample_seq,1),tf.concat(logit_seq,1)

#prob=1
def mle_decode1(h,inp_sam,length,cell,softmax):
    prob_seq,sample_seq,logit_seq=[],[],[]
    
    #sam_pro=tf.constant([prob]*FLAGS.batch_size)
    for t in range(length+1):
        inp=inp_sam        
        output,h = cell(inp,h)
        inp_sam,sample,prob,logit=softmax(output)
        prob_seq.append(tf.expand_dims(prob,1))
        sample_seq.append(tf.expand_dims(sample,1))
        logit_seq.append(tf.expand_dims(logit,1))
       
    return tf.concat(prob_seq,1),tf.concat(sample_seq,1),tf.concat(logit_seq,1)



class SeqTranslator(object):
    def __init__(self,
                vocab_size,
                dim_h = FLAGS.dim_h,
                dim_e = FLAGS.dim_e,
                style_num = FLAGS.style_num,
                maxlen = FLAGS.maxlen,
                dropout_rate = FLAGS.dropout_rate,
                gumbel_gamma = FLAGS.gumbel_gamma,
                word_embeddings = None,
                inputs = None,
                variables = None,
                modules = None):


        self.dim_h = dim_h
        self.dim_e = dim_e
        ## only use one shared vocabulary
        self.vocab_size = vocab_size
        self.maxlen = maxlen

        self.dropout_rate = dropout_rate
        self.gumbel_gamma = gumbel_gamma
        self.style_num = style_num
        self.word_embeddings = word_embeddings
    


        ## if not serve as a sub-module
        if(inputs == None):
            self.inputs = self.__create_input_port__()
        else:
            self.inputs = inputs  #otherwise, the supermodule should create the inputs

        if(variables == None):
            self.variables = self.__create_variables__()
        else:
            self.variables = variables

        if(modules == None):
            self.modules = self.__create_modules__()
        else:
            self.modules = modules	

        self.batch_size = self.inputs["src_seq"][0]

       
        src_embedding = tf.matmul(self.inputs["src_seq"], tf.tile(tf.expand_dims(self.variables["embedding_table"], 0), [tf.shape(self.inputs["src_seq"])[0], 1, 1]))
        initial_state = tf.zeros(shape=[tf.shape(self.inputs["src_seq"])[0], self.dim_h])
  
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(self.modules["encoder_cell"], src_embedding, initial_state=initial_state, dtype = tf.float32) 
        decoder_state = encoder_state

        #define the graph of the pretrain step
        softmax = softmax_word_sample(self.dropout_rate, self.variables["proj_W"], self.variables["proj_b"], self.variables["embedding_table"])
        self.decoder_inps_mle=tf.nn.embedding_lookup(self.variables["embedding_table"], self.inputs["shifted_src_seq"]) # batch_size * seq_len * d_emb
        self.prob_mle, self.sample ,self.logit_mle= mle_decode(decoder_state, self.decoder_inps_mle,self.maxlen, 
            self.modules["decoder_cell"], softmax,self.inputs["schedule_prob"])


        self.data=tf.placeholder(dtype = tf.int32, shape = (None, None))
        src_embedding = tf.nn.embedding_lookup(self.variables["embedding_table"],self.data)
        initial_state = tf.zeros(shape=[tf.shape(self.data)[0], self.dim_h])
  
        _,self.encoder_state = tf.nn.dynamic_rnn(self.modules["encoder_cell"], src_embedding, initial_state=initial_state, dtype = tf.float32) 

        self.hidden_state_in=tf.placeholder(dtype = tf.float32, shape = (None, None))
        inp=tf.nn.embedding_lookup(self.variables["embedding_table"],tf.zeros(shape=[tf.shape(self.hidden_state_in)[0]],dtype=tf.int32))
        _, self.sample1 ,_ = mle_decode1(self.hidden_state_in,inp,self.maxlen,self.modules["decoder_cell"],softmax) 
                


 
    ## wrap the creation of the input ports
    def __create_input_port__(self):
        return {
            "src_seq" : tf.placeholder(dtype = tf.float32, shape = (None, None)),
            "tgt_style": tf.placeholder(dtype = tf.int32, shape = (None, ))
        }

    ## wrap the creation of the parameters (not including the built-in keras components')
    def __create_variables__(self):
        #init = tf.random_normal((self.vocab_size, self.dim_e))
        return {

            "embedding_table" : tf.get_variable("embedding_table" ,[self.vocab_size, self.dim_e]) if self.word_embeddings is None else tf.get_variable("embedding_table" ,
             initializer = self.word_embeddings,trainable=False),#固定embedding
            "style_embedding_table" : tf.get_variable("style_embedding_table", initializer = np.eye(self.style_num, dtype = np.float32), trainable = False),
            "proj_W" : tf.get_variable("proj_W", [self.dim_h, self.vocab_size]),
            "proj_b" : tf.get_variable("proj_b", [self.vocab_size])
        }

    ## wrap the creation of the modules
    def __create_modules__(self):
        return {
            "encoder_cell" : rnn.GRUCell(self.dim_h),
            "decoder_cell" : rnn.GRUCell(self.dim_h)
        }


    def get_trainable_weights(self):
        #weights = list([self.variables["embedding_table"], self.variables["proj_W"], self.variables["proj_b"]])     
        weights = list([self.variables["proj_W"], self.variables["proj_b"]])
        for module in self.modules.values():
            weights += module.trainable_weights           
        return weights


