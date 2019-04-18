## used to wrap the whole model
import tensorflow as tf
import numpy as np
from generator import SeqTranslator
from dataloader import *
from util import *
import sys

class EmotionGAN(object):
    def __init__(self, sess, data, FLAGS, word_embeddings=None):
        self.sess = sess
        self.FLAGS = FLAGS
        self.vocab_size = self.FLAGS.vocab_size
        self.data = data
        self.batch_size = self.data.batch_size

        self.inputs = self.__create_input_port__()

        self.G_MLE = SeqTranslator(vocab_size=self.vocab_size, inputs={
                                        "src_seq":self.inputs["src_seq"],
                                        "tgt_style":self.inputs["src_style"],
                                        "shifted_src_seq":self.inputs["shifted_src_seq"],
                                        "schedule_prob":self.inputs["schedule_prob"]},
                                        word_embeddings=word_embeddings)
        self.prob_mle = self.G_MLE.prob_mle #batch_size * seq_len * vorcab_size
        self.sample = self.G_MLE.sample    #batch_size * seq_len 
        self.logit_mle = self.G_MLE.logit_mle #batch_size * seq_len * vocab_size

        print(self.G_MLE.get_trainable_weights())
        print('\n')
         
     
        self.mle_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.inputs["src_seq"],logits=self.logit_mle))

        ## hyperparameter
        lambda_cls = 1
        lambda_rec = 1
        learning_rate = 0.01
        grad_clip = 30.0

        ## do some pretraining
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)

     
        self.theta_GMLE = self.G_MLE.get_trainable_weights()
       # print(self.theta_GMLE)

        self.min_mle = self.optimizer.minimize(self.mle_loss,var_list=self.theta_GMLE)

  

        tf.summary.scalar("mle_loss",self.mle_loss)



        '''
        tf.summary.scalar("L_discriminator", self.L_D)
        tf.summary.scalar("L_generator", self.L_G)
        tf.summary.scalar("L_cls", self.cls_loss_src)
        tf.summary.scalar("L_id", self.rec_loss)
        '''


    def translate(self,batch):
        feed_dict = self.create_feed_dict(batch)

        src = batch["src_seq"]
        prime, recon, probs,b1,b = self.sess.run([self.x_prime_hard_seq,self.x_recon_hard_seq,self.x_prime,
            self.G_MLE.variables["proj_b"],self.G.variables["proj_b"]],feed_dict=feed_dict)

        src = [[self.data.id2word[i] for i in sent] for sent in src]
        src = strip_eos(strip_pad(src))

        prime = [[self.data.id2word[i] for i in sent] for sent in prime]
        #prime = [[str(i) for i in sent] for sent in prime]
        prime = strip_eos(strip_pad(prime))

        recon = [[self.data.id2word[i] for i in sent] for sent in recon]
        recon = strip_eos(strip_pad(recon))

        return src, prime, recon, probs,b1,b

    def autoencoder(self,batch,schedule_prob):
        feed_dict = self.create_feed_dict(batch)
        feed_dict[self.inputs["schedule_prob"]]=schedule_prob
        src = batch["src_seq"]
        recon = self.sess.run(self.sample,feed_dict=feed_dict)

        src = [[self.data.id2word[i] for i in sent] for sent in src]
        src = strip_eos(strip_pad(src))

        recon = [[self.data.id2word[i] for i in sent] for sent in recon]
        recon = strip_eos(strip_pad(recon))
        

        return src,recon




    def create_feed_dict(self, batch):
        #print(one_hot(batch["src_seq"], self.vocab_size).shape)
        return {
            self.inputs["src_seq"] : one_hot(batch["src_seq"], self.vocab_size),
            self.inputs["tgt_style"] : batch["tgt_style"],
            self.inputs["src_style"] : batch["src_style"],
    
            self.inputs["batch_len"] : batch["len"],
            self.inputs["shifted_src_seq"] :  batch["shifted_src_seq"]
        }



    def __create_input_port__(self):
        flags = self.FLAGS
        return {
            "src_seq" : tf.placeholder(dtype = tf.float32, shape = (None, None, self.vocab_size)),
    
            "src_style" : tf.placeholder(dtype = tf.int32, shape = (None, )),
            "tgt_style": tf.placeholder(dtype = tf.int32, shape = (None, )),
            "batch_len" : tf.placeholder(dtype = tf.int32),
            "shifted_src_seq" : tf.placeholder(dtype = tf.int32, shape = (None, None)),

            "schedule_prob":tf.placeholder(dtype=tf.float32,shape=(None))

        }

if __name__ == '__main__':
    with tf.Session() as sess:

        FLAGS.vocab_size =8664
        yelp_data = yelp_dataloader(FLAGS.train_path,FLAGS.maxlen,FLAGS.minlen,FLAGS.batch_size)
        batch = yelp_data.next_batch()
        model = EmotionGAN(sess,yelp_data,FLAGS)
        sess.run(tf.global_variables_initializer())
        
        feed_dict = model.create_feed_dict(batch)
        l_rec, _ = sess.run([model.rec_loss, model.min_recon], feed_dict = feed_dict)

        l_d, _ = sess.run([model.L_D, model.min_D], feed_dict=feed_dict)
        l_g, _ = sess.run([model.L_G, model.min_G], feed_dict=feed_dict)
