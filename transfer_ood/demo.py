import os
os.environ["CUDA_VISIBLE_DEVICES"]='0' 
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
from util import *
csv.field_size_limit(sys.maxsize)
np.set_printoptions(threshold=np.inf)


MLE_pretrain = True

#path for saving model
MLE_path = "save/MLE_300.ckpt"


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
    dataset=['yelp','amazon']
    #对原来amazon的截断情况
    #trunc = [70000,-1]#对每个数据集取数据的长度，-1则不截断
    #对amazon1的截断情况
    trunc=[60000,60000]
    dataloader = dataloader(dataset,trunc)


    FLAGS.vocab_size = dataloader.vocab_size
    word_embeddings = load_embeddings(dataloader, FLAGS.embedding_path)
    print("data preparation finished")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) 
    model, merged, writer = create_model(sess, FLAGS, pretrained = False, data = dataloader, word_embeddings = word_embeddings)
   
    var_list=model.get_trainable_weights()
    saver=tf.train.Saver(var_list=var_list)
    #saver=tf.train.Saver()

    if MLE_pretrain:
        saver.restore(sess,MLE_path)
    
        
    else:
        ite =1
        prob=0
        #test MLE
        for e in range(30):           
            dataloader.reset()
            prob=0.05*e
            print("epoch: {},schedule prob is {}".format(e,prob))
            schedule_prob=FLAGS.batch_size*[prob]
            for c in range(1,1+dataloader.batch_num):
                batch = dataloader.next_batch()
                feed_dict = model.create_feed_dict(batch)
                feed_dict[model.inputs["schedule_prob"]]=schedule_prob
                labels=sess.run(model.labels,feed_dict = feed_dict)
                #print("label {}".format(labels))
                summary,mle = sess.run([merged,model.min_mle],feed_dict = feed_dict)          
                writer.add_summary(summary,ite)
                
                ite+=1
                if(c % 200 == 0):
                    #dataloader.reset()
                    sample_batch = dataloader.random_batch()
                    ost, ae = model.autoencoder(sample_batch,schedule_prob)
                    for i in range(min(4, len(ost))):
                        print("Sample {}. {} \n -> {} \n".format(i, ' '.join(ost[i]),  ' '.join(ae[i])))
        saver.save(sess,MLE_path)


    transfer_path_sentiment="outputs/sentiment_high_confidence_300"
    transfer_path_tense="outputs/tense_high_confidence_300"
    
    model.save_data(dataloader,transfer_path_sentiment,transfer_path_tense)
    print("save finished")
    
    #yelp on tense
    classifier_path="save/classifier_tense.ckpt"#save the transferred sentences
    FLAGS.result_path="outputs/"+dataset[0]+"/result_300" #save the sentences with high confidence to calculate style matrix
    FLAGS.used_train_path="outputs/"+dataset[0]+"/used_train_300"
    FLAGS.test_path=FLAGS.result_path
    data_tobe_transferd=dataloader.data_yelp[:100000]
    print(np.shape(data_tobe_transferd))
    trans_vec1,trans_vec2=transfer_vecs_via_datas(sess,model,dataloader,data_tobe_transferd,transfer_path_tense+'.0',transfer_path_tense+'.1')
    save_and_print(sess,model,dataloader,data_tobe_transferd,trans_vec1,trans_vec2)
    FLAGS.train_path=FLAGS.train_path+dataset[1]+'/train'
    FLAGS.dev_path=FLAGS.dev_path+dataset[1]+'/dev'
    calculate_acc(None,save_path=classifier_path,train=True)
    bleu_eval_pair()
    

    '''
    #amazon on sentiment
    classifier_path="save/classifier_sentiment.ckpt"
    FLAGS.result_path="outputs/"+dataset[1]+"/result_300"
    FLAGS.used_train_path="outputs/"+dataset[1]+"/used_train_300"
    FLAGS.test_path=FLAGS.result_path
    data_tobe_transferd=dataloader.data_amazon[:100000]
    trans_vec1,trans_vec2=transfer_vecs_via_datas(sess,model,dataloader,data_tobe_transferd,transfer_path_sentiment+'.0',transfer_path_sentiment+'.1')
    save_and_print(sess,model,dataloader,data_tobe_transferd,trans_vec1,trans_vec2)
    FLAGS.train_path=FLAGS.train_path+dataset[0]+'/train'
    FLAGS.dev_path=FLAGS.dev_path+dataset[0]+'/dev'
    calculate_acc(None,save_path=classifier_path,train=True)
    bleu_eval_pair()
    '''



    
    



        
   
    print("down")
    sys.exit(0)
