## used to wrap the whole model
import tensorflow as tf
import numpy as np
from dataloader import *
from util import *
#from tensorflow.keras.layers import GRUCell
import tensorflow.contrib.rnn as rnn
import sys
import random



def softmax_word_sample(dropout_rate, output_layer, embedding):
    def loop_func(output):
        #output = tf.nn.dropout(output, keep_prob= 1-dropout_rate)
        logits = output_layer(output) #batch_size * vocab_size
        prob = tf.nn.softmax(logits)  #batch_size * vocab_size
        #sample = tf.reshape(tf.random.categorical(tf.log(logits),1),[-1])
        sample = tf.argmax(prob,axis=-1)
        nxt_inp = tf.nn.embedding_lookup(embedding,sample) #batch_size * dim_e
        return nxt_inp,sample,prob,logits
    return loop_func



#以prob的概率使用来自采样的上一个样本，当检测生成效果时，prob=1
def mle_decode(h,seq_inp,length,cell,softmax,schedule_prob):
    prob_seq,sample_seq,logit_seq=[],[],[]
    inp_sam = seq_inp[:,0,:]
    
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
    
    for t in range(length+1):
        inp=inp_sam        
        output,h = cell(inp,h)
        inp_sam,sample,prob,logit=softmax(output)
        prob_seq.append(tf.expand_dims(prob,1))
        sample_seq.append(tf.expand_dims(sample,1))
        logit_seq.append(tf.expand_dims(logit,1))
       
    return tf.concat(prob_seq,1),tf.concat(sample_seq,1),tf.concat(logit_seq,1)

def id2sentens(data,sent):
    sent = [[data.id2word[i] for i in sent]]
    sent = strip_eos(strip_pad(sent))
    return ' '.join(sent[0])+'\n'

class Autoencoder(object):
    def __init__(self, sess, dataloader, FLAGS, word_embeddings=None):
        self.sess = sess
        self.vocab_size = FLAGS.vocab_size
        self.dataloader = dataloader
        self.batch_size = self.dataloader.batch_size
        self.dim_h = FLAGS.dim_h
        self.dim_e = FLAGS.dim_e
        self.maxlen = FLAGS.maxlen
        self.dropout_rate = FLAGS.dropout_rate

        self.inputs = self.__create_input_port__()
        self.modules = self.__create_modules__()

        self.embedding_table= tf.get_variable("embedding_table", initializer = word_embeddings)#固定embedding

        src_embedding = tf.nn.embedding_lookup(self.embedding_table,self.inputs["src_seq"])
        initial_state = tf.zeros(shape=[self.batch_size, self.dim_h])
  
        encoder_outputs, self.hidden_state = tf.nn.dynamic_rnn(self.modules["encoder_cell"], src_embedding, initial_state=initial_state, dtype = tf.float32) 

        covar=tf.matmul(tf.reshape(self.hidden_state,[-1,self.dim_h,1]),tf.reshape(self.hidden_state,[-1,1,self.dim_h]))
        covar=tf.reshape(covar,[self.batch_size,self.dim_h*self.dim_h])
        self.class_logit_covari_1=tf.reshape(self.modules["classifier_layer_covari_1"](covar),[-1])
        self.class_logit_covari_2=tf.reshape(self.modules["classifier_layer_covari_2"](covar),[-1])

              
        #define the graph of the decoder
        softmax = softmax_word_sample(self.dropout_rate, self.modules["output_layer"], self.embedding_table)
        self.decoder_inps_mle=tf.nn.embedding_lookup(self.embedding_table, self.inputs["shifted_src_seq"]) # batch_size * seq_len * d_emb
        self.prob_mle, self.sample ,self.logit_mle= mle_decode(self.hidden_state, self.decoder_inps_mle,self.maxlen, 
            self.modules["decoder_cell"], softmax,self.inputs["schedule_prob"])

        #define the graph to transfer the seqs(using trained parameters)
        self.data=tf.placeholder(dtype = tf.int32, shape = (None, None))
        src_embedding = tf.nn.embedding_lookup(self.embedding_table,self.data)
        initial_state = tf.zeros(shape=[tf.shape(self.data)[0], self.dim_h])
  
        _,self.hidden_state_out = tf.nn.dynamic_rnn(self.modules["encoder_cell"], src_embedding, initial_state=initial_state, dtype = tf.float32) 

        self.hidden_state_in=tf.placeholder(dtype = tf.float32, shape = (None, None))
        inp2=tf.nn.embedding_lookup(self.embedding_table,tf.zeros(shape=[tf.shape(self.hidden_state_in)[0]],dtype=tf.int32))
        _, self.sample1 ,_ = mle_decode1(self.hidden_state_in,inp2,self.maxlen,self.modules["decoder_cell"],softmax) 


        self.mle_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputs["src_seq"],logits=self.logit_mle))
       
        self.classifier_loss_covari_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputs["src_style"],logits=self.class_logit_covari_1))
        self.classifier_loss_covari_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputs["src_style"],logits=self.class_logit_covari_2))
        self.labels=tf.one_hot(self.inputs["label"],depth=2)
        self.classifier_loss = self.labels[0]*self.classifier_loss_covari_1+self.labels[1]*self.classifier_loss_covari_2
    
       
       
        ## hyperparameter
        learning_rate = 0.0005

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.trainable_weights = self.get_trainable_weights()
        self.min_mle = self.optimizer.minimize(self.mle_loss+0.1*self.classifier_loss,var_list=self.trainable_weights)

      
        tf.summary.scalar("mle_loss",self.mle_loss)
        tf.summary.scalar("classifier_loss",self.classifier_loss)

        tf.summary.scalar("classifier_loss_covari_1",self.classifier_loss_covari_1)
        tf.summary.scalar("classifier_loss_covari_2",self.classifier_loss_covari_2)
     



        ## wrap the creation of the modules
    def __create_modules__(self):
        return {
    
            "encoder_cell" : rnn.GRUCell(self.dim_h,kernel_initializer=tf.orthogonal_initializer()),
            "decoder_cell" : rnn.GRUCell(self.dim_h,kernel_initializer=tf.orthogonal_initializer()),
            "classifier_layer_covari_1" : tf.layers.Dense(1),
            "classifier_layer_covari_2" : tf.layers.Dense(1),
            "output_layer" : tf.layers.Dense(self.vocab_size)

        }

    def __create_input_port__(self):
        return {
            "src_seq" : tf.placeholder(dtype = tf.int32, shape = (None, None)),
            "src_style" : tf.placeholder(dtype = tf.float32, shape = (None)),
            "shifted_src_seq" : tf.placeholder(dtype = tf.int32, shape = (None, None)),
            "schedule_prob":tf.placeholder(dtype=tf.float32,shape=(None)),
            "label":tf.placeholder(dtype=tf.int32)         

        }


    def get_trainable_weights(self):
        #weights = list([self.embedding_table])     
        weights = []
        for module in self.modules.values():
            weights += module.trainable_weights           
        return weights


    def autoencoder(self,batch,schedule_prob):
        feed_dict = self.create_feed_dict(batch)
        feed_dict[self.inputs["schedule_prob"]]=schedule_prob
        src = batch["src_seq"]
        recon = self.sess.run(self.sample,feed_dict=feed_dict)

        src = [[self.dataloader.id2word[i] for i in sent] for sent in src]
        src = strip_eos(strip_pad(src))

        recon = [[self.dataloader.id2word[i] for i in sent] for sent in recon]
        recon = strip_eos(strip_pad(recon))
        
        return src,recon



    def save_data(self,dataloader,file_name1,file_name2):
           
        neg_data=[]
        pos_data=[]
        post_data=[]
        pres_data=[]
        for batch in dataloader.batches:
            feed_dict=self.create_feed_dict(batch)
            if batch['label'] ==0:

                logits=self.sess.run(self.class_logit_covari_1,feed_dict=feed_dict)
                for i in range(self.batch_size):
                    if batch['src_style'][i]==0 and logits[i]<-23:#-50:
                        neg_data.append(batch['src_seq'][i])       
                    if batch['src_style'][i]==1 and logits[i]>30:#100:            
                        pos_data.append(batch['src_seq'][i])
            elif batch['label'] ==1:
                
                logits=self.sess.run(self.class_logit_covari_2,feed_dict=feed_dict)
                for i in range(self.batch_size):
                    if batch['src_style'][i]==0 and logits[i]<-23:#-50:
                        post_data.append(batch['src_seq'][i])       
                    if batch['src_style'][i]==1 and logits[i]>30:#100:            
                        pres_data.append(batch['src_seq'][i])
              
                    
        print("neg_data length is {}".format(len(neg_data)))
        print("pos_data length is {}".format(len(pos_data)))
        print("post_data length is {}".format(len(post_data)))
        print("pres_data length is {}".format(len(pres_data)))
       
        random.shuffle(pos_data)
        #pos_data=pos_data[:10000]
        random.shuffle(neg_data)
        #neg_data=neg_data[:10000]
        random.shuffle(post_data)
        random.shuffle(pres_data)

        f0=open(file_name1+'.0','w')
        f1=open(file_name1+'.1','w')
        for sent in neg_data:
            f0.write(id2sentens(dataloader,sent))
        for sent in pos_data:
            f1.write(id2sentens(dataloader,sent))
        
        f0.close()
        f1.close()

        f0=open(file_name2+'.0','w')
        f1=open(file_name2+'.1','w')
        for sent in post_data:
            f0.write(id2sentens(dataloader,sent))
        for sent in pres_data:
            f1.write(id2sentens(dataloader,sent))
        
        f0.close()
        f1.close()
        print("save the datas with high confidence")

    def create_feed_dict(self, batch):
        return {
            self.inputs["src_seq"] : batch["src_seq"],#one_hot(batch["src_seq"], self.vocab_size),
            self.inputs["src_style"] : batch["src_style"],  
            self.inputs["shifted_src_seq"] :  batch["shifted_src_seq"],
            self.inputs["label"] :batch["label"]
        }


if __name__ == '__main__':
    with tf.Session() as sess:

        FLAGS.vocab_size =8664
        yelp_data = yelp_dataloader(FLAGS.train_path,FLAGS.maxlen,FLAGS.minlen,FLAGS.batch_size)
        batch = yelp_data.next_batch()
        model = Autoencoder(sess,yelp_data,FLAGS)
        sess.run(tf.global_variables_initializer())
        
        feed_dict = model.create_feed_dict(batch)
        l_rec, _ = sess.run([model.rec_loss, model.min_recon], feed_dict = feed_dict)

        l_d, _ = sess.run([model.L_D, model.min_D], feed_dict=feed_dict)
        l_g, _ = sess.run([model.L_G, model.min_G], feed_dict=feed_dict)
