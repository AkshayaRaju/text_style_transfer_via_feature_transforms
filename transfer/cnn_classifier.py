import os
os.environ["CUDA_VISIBLE_DEVICES"]='1' 
import sys
import random
from datetime import datetime
import numpy as np
import csv
import tensorflow as tf
from options import FLAGS
from eval_dataloader import eval_dataloader
from util import *

csv.field_size_limit(sys.maxsize)

def load_embeddings(data, vector_file):
    dim_e = FLAGS.eval_dim_e
    embeddings = np.zeros(shape = (data.vocab_size, dim_e+4), dtype = np.float32)
    #add embs for go eos unk pad
    embeddings[0,dim_e]=1
    embeddings[1,dim_e+1]=1
    embeddings[2,dim_e+2]=1
    embeddings[3,dim_e+3]=1
    FLAGS.dim_e += 4
    ## open the file
    f = open(vector_file, 'r')
    vec_dict = dict()
    reader = csv.reader(f, delimiter = ' ')

   
    for row in reader:
        vec_dict[row[0]] = np.array([float(x) for x in row[1:]]+[0,0,0,0])
    for i in range(data.vocab_size):
        if(data.id2word[i] in vec_dict):
            embeddings[i, :] = vec_dict[data.id2word[i]]
    return embeddings


def evaluate(sess,model, batches,labels):
    probs = []
    for batch in batches:
        p = sess.run(model.probs,
            feed_dict={model.inp_seq: batch['seqs'],
                       model.keep_prob: 1})
        probs += p.tolist()
    labels_hat = [p > 0.5 for p in probs]
    same = [p == q for p, q in zip(labels, labels_hat)]
    return 100.0 * sum(same) / len(labels)


def id2sentens(data,sent):
    sent = [[data.id2word[i] for i in sent]]
    sent = strip_eos(strip_pad(sent))
    return ' '.join(sent[0])+'\n'
 
def save_data(sess,model,train_batches,data,file_name):
    f0=open(file_name+'.0','w')
    f1=open(file_name+'.1','w')

    for batch in train_batches:
        probs = sess.run(model.probs,
            feed_dict={model.inp_seq:batch['seqs'],
                       model.keep_prob:1})
        for i in range(len(batch['seqs'])):
            if batch['labels'][i]==1 and probs[i]>0.99999:
                f1.write(id2sentens(data,batch['seqs'][i]))
            if batch['labels'][i]==0 and probs[i]<0.00001:
                f0.write(id2sentens(data,batch['seqs'][i]))

    f0.close()
    f1.close()



def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)


def cnn(inp, filter_sizes, n_filters, dropout):
    dim = inp.get_shape().as_list()[-1]
    inp = tf.expand_dims(inp, -1)

    with tf.variable_scope('cnn') as vs:

        outputs = []
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                W = tf.get_variable('W', [size, dim, 1, n_filters])
                b = tf.get_variable('b', [n_filters])
                conv = tf.nn.conv2d(inp, W,
                    strides=[1, 1, 1, 1], padding='VALID')
                h = leaky_relu(conv + b)
                # max pooling over time
                pooled = tf.reduce_max(h, reduction_indices=1)
                pooled = tf.reshape(pooled, [-1, n_filters])
                outputs.append(pooled)
        outputs = tf.concat(outputs, 1)
        outputs = tf.nn.dropout(outputs, dropout)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [n_filters*len(filter_sizes), 1])
            b = tf.get_variable('b', [1])
            logits = tf.reshape(tf.matmul(outputs, W) + b, [-1])

    return logits

class cnn_classifier:
    def __init__(self,vocab_size,embeddings,
        dim_e=FLAGS.eval_dim_e,
        filter_sizes=[int(i) for i in FLAGS.filter_sizes.split(',')],
        output_channel=FLAGS.output_channel):
        
        self.inp_seq=tf.placeholder(tf.int32,[None,None])
        self.labels=tf.placeholder(tf.float32,[None])
        self.keep_prob=tf.placeholder(tf.float32)

        self.embeddings=tf.get_variable("embedding_tabel",initializer = embeddings,trainable=True)
        #embedding_tabel=tf.get_variable("embedding_tabel",[vocab_size,dim_e],dtype=tf.float32)
        inp_emb=tf.nn.embedding_lookup(self.embeddings,self.inp_seq)
        self.logits=cnn(inp_emb, filter_sizes, output_channel, self.keep_prob)
        self.probs=tf.sigmoid(self.logits)

        learning_rate=0.0005
        self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,logits=self.logits))
        self.optimizer=tf.train.AdamOptimizer(learning_rate)
        self.min_cross_entropy=self.optimizer.minimize(self.loss)

        tf.summary.scalar("loss",self.loss)
        self.merged = tf.summary.merge_all()
        self.saver=tf.train.Saver()




def calculate_acc(embeddings=None,save_path='save/classifier_yelp.ckpt',train=False,test=True):

    data = eval_dataloader(FLAGS.train_path,FLAGS.maxlen,FLAGS.minlen,FLAGS.batch_size)
    if embeddings is None:        
        embeddings=load_embeddings(data, FLAGS.eval_embedding_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = cnn_classifier(data.vocab_size,embeddings)
        if train:
            print('Creating model with fresh parameters.')
            sess.run(tf.global_variables_initializer())            
        else:
            print('Loading model')
            model.saver.restore(sess, save_path)
            
        v = tf.trainable_variables()#tf.get_collection(tf.GraphKeys.TRAINALBEL_VARIABLES)
        #print(v)
        #sys.exit()

        print("test_path:{}".format(FLAGS.test_path))
        train_batch=data.batch
        dev_batch,dev_labels=data.load_datas(FLAGS.dev_path)      
        test_batch,test_labels=data.load_datas(FLAGS.test_path)
        pos_batch,pos_labels=data.load_datas_pos(FLAGS.test_path)
        neg_batch,neg_labels=data.load_datas_neg(FLAGS.test_path)

        if train:
            #writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train'+datetime.now().strftime("%Y%m%d_%H%M%S"), sess.graph)
            best_dev = float('-inf')
            ite=1
            for epoch in range(10):
                print("--------------------------eopch {}---------------------------".format(epoch))

                for i in range(len(train_batch)):
                    #_,loss,summary=sess.run([model.min_cross_entropy,model.loss,model.merged],
                    _,loss=sess.run([model.min_cross_entropy,model.loss],
                        feed_dict={model.inp_seq : train_batch[i]['seqs'],
                        model.labels : train_batch[i]['labels'],
                        model.keep_prob: FLAGS.classifier_keep_prob})
                    #writer.add_summary(summary,ite)
                    ite=ite+1

                acc=evaluate(sess,model,dev_batch,dev_labels)
                print('dev accuracy is {}'.format(acc))

                if(acc>best_dev):
                    best_dev=acc
                    print('saving model...')
                    model.saver.save(sess,save_path)
            #save_data(sess,model,train_batch,data,'data/yelp/high_confidence')
            
        if test:
            acc=evaluate(sess,model,test_batch,test_labels)
            print('test accuracy is {}'.format(acc))
            acc_pos=evaluate(sess,model,pos_batch,pos_labels)
            acc_neg=evaluate(sess,model,neg_batch,neg_labels)
            print('pos acc is {}\nneg acc is{}'.format(acc_pos,acc_neg))
            return acc
           

if __name__ == '__main__':
    classifier_path="save/yelp/classifier_300.ckpt"
    #FLAGS.test_path="/home/mlsnrs/data/bns/baseline1/tmp/sentiment.dev.epoch20"
    FLAGS.test_path="/home/mlsnrs/data/bns/baseline3/trainsamples/dev_transfer"
    calculate_acc(None,save_path=classifier_path,train=False)
    






