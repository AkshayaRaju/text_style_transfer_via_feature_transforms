import os
os.environ["CUDA_VISIBLE_DEVICES"]='0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='0'
from keras import backend as K
K.clear_session()
import tensorflow as tf
from dataloader import *
from options import FLAGS
from model import EmotionGAN
from datetime import datetime
import csv
import sys
import numpy as np
from numpy.linalg import eig
from util import *

np.set_printoptions(threshold=np.inf)

MLE_pretrain = False

def transfer(vec1,vec2):#seq_num * dim_h

    m1=np.mean(vec1,axis=0)
    m2=np.mean(vec2,axis=0)
    vec1=vec1-m1
    vec2=vec2-m2

    
    vec1=np.transpose(vec1)#dim_h * seq_num 

    area = vec1.shape[1] - 1
    covar1=np.dot(vec1,vec1.T)/area
    print(covar1.shape)
    vec2=np.transpose(vec2)
    covar2=np.dot(vec2,vec2.T)/area
    #print(covar2)
    print(np.linalg.cond(covar1))

    evals1,evecs1 = eig(covar1)
<<<<<<< Updated upstream
    # print(evals1)
    evals1 = np.diag(np.power(evals1,-1/2))
=======
    eig_s = evals1
    eig_s = np.array(list(filter(lambda x:x > 0.00001, evals1)))
    eig_s = np.power(eig_s, -1/2)

    
    evecs1 = evecs1[:,:len(eig_s)]
    #print(evals1)
    # evals1 = np.diag(np.power(np.abs(evals1),-1/2))
>>>>>>> Stashed changes

    print(eig_s.shape)
    print(evecs1.shape)
    print(vec1.shape)

    evals2,evecs2 = eig(covar2)
<<<<<<< Updated upstream
    # print(evals2)
    evals2=np.diag(np.power(evals2,1/2))

    fc=np.dot(np.dot(np.dot(evecs1,evals1),evecs1.T),vec1)
    fcs=np.dot(np.dot(np.dot(evecs2,evals2),evecs2.T),fc)
=======
    eig_t = evals2
    eig_t = np.array(list(filter(lambda x:x > 0.00001, evals2)))
    eig_t = np.power(eig_t, 1/2)
    
    evecs2 = evecs2[:,:len(eig_t)]
    
    
    #print(evals2)
    # evals2=np.diag(np.power(np.abs(evals2),1/2))
    fc = evecs1 @ np.diag(eig_s) @ evecs1.T @ vec1 # dim_h * seq_num
    # print(fc.shape)
    fcs = evecs2 @ np.diag(eig_t) @ evecs2.T @ fc # dim_h * seq_num
    
    # fc=np.dot(np.dot(np.dot(evecs1,evals1),evecs1.T),vec1)
    # fcs=np.dot(np.dot(np.dot(evecs2,evals2),evecs2.T),fc)
>>>>>>> Stashed changes

    return fcs.T+m2


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
    model = EmotionGAN(sess,data,FLAGS,word_embeddings) #Model(sess, FLAGS, vocab, word_embeddings)
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
    sess = tf.Session(config=config) 
    model, merged, writer = create_model(sess, FLAGS, pretrained = False, data = yelp_data, word_embeddings = word_embeddings)
    #saver = tf.train.Saver()
    #var_list=model.G_MLE.get_trainable_weights()
    var_list=model.G_MLE.get_trainable_weights()

    saver=tf.train.Saver(var_list=var_list)

    if MLE_pretrain:
        saver.restore(sess,"save/MLE_5_100.ckpt")
        # print(model.G_MLE.get_trainable_weights()[1])
        for i in range(len(model.G_MLE.get_trainable_weights())):
            pass
            # print(model.G_MLE.get_trainable_weights()[i])
            #print(sess.run(model.G_MLE.get_trainable_weights()[i]))
        # print(sess.run(model.G_MLE.get_trainable_weights()[1]))
    else:
        ite =1
        prob=0
        #test MLE
        for e in range(2):           
            yelp_data.reset()
            prob=0.2*e
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
        save_path = saver.save(sess,"save/MLE_5_100.ckpt")
        for i in range(len(model.G_MLE.get_trainable_weights())):
            print(model.G_MLE.get_trainable_weights()[i])
            #print(sess.run(model.G_MLE.get_trainable_weights()[i]))
        print(sess.run(model.G_MLE.get_trainable_weights()[1]))

    
    vec1=sess.run(model.G_MLE.encoder_state,feed_dict={model.G_MLE.data:yelp_data.data1}) #seq_num * dim_h
    vec2=sess.run(model.G_MLE.encoder_state,feed_dict={model.G_MLE.data:yelp_data.data2})

    trans_vec1=transfer(vec1,vec2).tolist()
    trans_vec1 = trans_vec1[:1000]
    trans_seq=sess.run(model.G_MLE.sample1,feed_dict={model.G_MLE.hidden_state_in:vec1[:10]})
    data1 = [[yelp_data.id2word[i] for i in sent] for sent in yelp_data.data1]
    data1 = strip_eos(strip_pad(data1))
    trans_seq = [[yelp_data.id2word[i] for i in sent] for sent in trans_seq]
    trans_seq = strip_eos(strip_pad(trans_seq))

    for i in range(len(trans_seq)):
        print("Sample {}. {} \n -> {} \n".format(i, ' '.join(data1[i]),  ' '.join(trans_seq[i])))
        
   
    print("down")
    sys.exit(0)
