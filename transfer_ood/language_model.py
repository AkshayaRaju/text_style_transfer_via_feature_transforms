import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
from options import FLAGS
from eval_dataloader import eval_dataloader
from cnn_classifier import load_embeddings



class language_model(object):
    def __init__(self,vocab_size,embeddings, 
        dim_e = FLAGS.eval_dim_e,
        dim_h = FLAGS.eval_dim_h):
 
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.inp_seqs = tf.placeholder(tf.int32, [None, None],name='input_sequences')    #batch_size * max_len
        self.labels = tf.placeholder(tf.int32, [None, None],name='labels')
        self.weights = tf.placeholder(tf.float32, [None, None],name='weights')

        learning_rate=0.0005

        self.embeddings = tf.get_variable('embedding_tabel',initializer=embeddings,trainable=True)
        #embedding_tabel=tf.get_variable("embedding_tabel",[vocab_size,dim_e],dtype=tf.float32)

        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab_size])
            proj_b = tf.get_variable('b', [vocab_size])

        inputs = tf.nn.embedding_lookup(self.embeddings, self.inp_seqs)  #batch_size * max_len * dim_e
        cell = tf.nn.rnn_cell.GRUCell(dim_h)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=self.keep_prob)

        outputs, _ = tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32, scope='language_model') #batch_size * max_len * dim_h
        outputs = tf.nn.dropout(outputs, self.keep_prob)
        outputs = tf.reshape(outputs, [-1, dim_h])
        self.logits = tf.matmul(outputs, proj_W) + proj_b

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.labels,[-1]),logits=self.logits)
        loss = loss*tf.reshape(self.weights,[-1])
        self.total_loss=tf.reduce_sum(loss)
        self.lm_loss = self.total_loss/tf.cast(tf.shape(self.inp_seqs)[0],tf.float32)
   

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.lm_loss)

        self.saver = tf.train.Saver()


def evaluate(sess, model, batches):
    tot_loss, n_words = 0, 0

    for batch in batches:
        tot_loss += sess.run(model.total_loss,
            feed_dict={model.inp_seqs:batch['seqs_shift'],
                        model.labels:batch['seqs'],
                        model.weights:batch['weights'],
                        model.keep_prob:FLAGS.lm_keep_prob})
        n_words += np.sum(batch['weights'])

    return np.exp(tot_loss / n_words)

def evaluate_perplexity(embeddings=None,save_path='save/yelp/language_model.ckpt',train=False,test=True):
    data = eval_dataloader(FLAGS.train_path,FLAGS.maxlen,FLAGS.minlen,FLAGS.batch_size)
    if embeddings is None:        
        embeddings=load_embeddings(data, FLAGS.eval_embedding_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = language_model(data.vocab_size,embeddings)
        if train:
            print('Creating model with fresh parameters.')
            sess.run(tf.global_variables_initializer())            
        else:
            print('Loading model')
            model.saver.restore(sess, save_path)

        train_batch = data.batch
        dev_batch,_=data.load_datas(FLAGS.dev_path)      
        test_batch,_=data.load_datas(FLAGS.test_path)
        random.shuffle(train_batch)
        if train:
            
            best_dev = float('inf')
            epoch=10

            for i in range(epoch): 
                for batch in train_batch:
                    sess.run([model.optimizer],
                        feed_dict={model.inp_seqs:batch['seqs_shift'],
                        model.labels:batch['seqs'],
                        model.weights:batch['weights'],
                        model.keep_prob:FLAGS.lm_keep_prob})
                
                ppl = evaluate(sess, model, dev_batch)
                print('dev perplexity %.2f' % ppl)
                if ppl < best_dev:
                    best_dev = ppl
                    print('Saving model...')
                    model.saver.save(sess, save_path)
        if test:
            ppl = evaluate(sess, model, test_batch)
            print('test perplexity %.2f' % ppl)

if __name__ == '__main__':
    FLAGS.test_path="outputs/yelp/result_500"
    evaluate_perplexity(None,train=False)     
