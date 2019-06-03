import os
os.environ["CUDA_VISIBLE_DEVICES"]='1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'

from cnn_classifier import calculate_acc
from eval_bleu import bleu_eval_pair
from keras import backend as K
K.clear_session()
import tensorflow as tf
from dataloader import *
from options import FLAGS
from model import Autoencoder
from transfer_util import *
from datetime import datetime
import csv
import sys
import numpy as np
import math
from util import *
csv.field_size_limit(sys.maxsize)
np.set_printoptions(threshold=np.inf)


MLE_pretrain = False
dataset='yelp'
neg_logit=-65
pos_logit=165
#path for saving model
MLE_path = "save/"+dataset+"/MLE_30_unsupervised.ckpt"
classifier_path="save/"+dataset+"/classifier_300.ckpt"
#save the transferred sentences
FLAGS.result_path="outputs/"+dataset+"/result_300_"+str(neg_logit)+"_"+str(pos_logit)
#save the sentences with high confidence to calculate style matrix
FLAGS.transfer_path="outputs/"+dataset+"/high_confidence_300"
FLAGS.used_train_path="outputs/"+dataset+"/used_train_300_"+str(neg_logit)+"_"+str(pos_logit)
FLAGS.test_path=FLAGS.result_path



def load_embeddings(data, vector_file):
    dim_e = FLAGS.dim_e
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


def create_model(sess, flags = FLAGS, pretrained = False, version = "pretrained", data = None, word_embeddings = None):
    model = Autoencoder(sess,data,FLAGS,word_embeddings) #Model(sess, FLAGS, vocab, word_embeddings)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train'+datetime.now().strftime("%Y%m%d_%H%M%S"), sess.graph)
    if(pretrained):
        print("Loading Model from {}".format(FLAGS.ckpt_path.format(version)))
        model.saver.restore(sess, FLAGS.ckpt_path.format(version))
    else:
        print('Creating model with fresh parameters.')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
    return model, merged, writer


if __name__ == '__main__':
    ####################  data load  ####################################
    #with K.get_session():
    yelp_data = yelp_dataloader(FLAGS.train_path,FLAGS.maxlen,FLAGS.minlen,FLAGS.batch_size)
    FLAGS.vocab_size = yelp_data.vocab_size
    word_embeddings = load_embeddings(yelp_data, FLAGS.embedding_path)
    print("data preparation finished")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) 
    model, merged, writer = create_model(sess, FLAGS, pretrained = False, data = yelp_data, word_embeddings = word_embeddings)
   
    var_list=model.get_trainable_weights()
    saver=tf.train.Saver(var_list=var_list)

    if MLE_pretrain:
        saver.restore(sess,MLE_path)
        
        
    else:
        ite =1
        prob=0
        #test MLE
        for e in range(25):           
            yelp_data.reset()
            prob=0.05*e
            print("epoch: {},schedule prob is {}".format(e,prob))
            schedule_prob=FLAGS.batch_size*[prob]
            for c in range(1,1+yelp_data.batch_num):
                batch = yelp_data.next_batch()
                feed_dict = model.create_feed_dict(batch)
                feed_dict[model.inputs["schedule_prob"]]=schedule_prob
                summary,mle = sess.run([merged,model.min_mle],feed_dict = feed_dict)          
                writer.add_summary(summary,ite)
                
                ite+=1
                if(c % 200 == 0):
                    #yelp_data.reset()
                    sample_batch = yelp_data.random_batch()
                    ost, ae = model.autoencoder(sample_batch,schedule_prob)
                    for i in range(min(4, len(ost))):
                        print("Sample {}. {} \n -> {} \n".format(i, ' '.join(ost[i]),  ' '.join(ae[i])))
        saver.save(sess,MLE_path)
           

    #yelp
    
    model.save_data(yelp_data,FLAGS.transfer_path,neg_logit,pos_logit)
    #trans_vec1,trans_vec2=transfer_vecs(sess,model,yelp_data)
    trans_vec1,trans_vec2,data_totrans1,data_totrans2=transfer_vecs_via_datas(sess,model,yelp_data,FLAGS.transfer_path+'.0',FLAGS.transfer_path+'.1')
    save_and_print(sess,model,yelp_data,trans_vec1,trans_vec2,data_totrans1,data_totrans2)
    
    acc=calculate_acc(None,save_path=classifier_path,train=False)
    bleu=bleu_eval_pair()*100
    
    g_score=math.sqrt(acc*bleu)
    mean=(acc+bleu)/2
    print("g_score:{}".format(g_score))   
    print("mean:{}".format(mean))          
   
    print("down")
   
