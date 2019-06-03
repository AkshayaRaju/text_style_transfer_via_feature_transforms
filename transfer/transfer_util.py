import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS,Isomap,LocallyLinearEmbedding
from options import FLAGS
from util import *
pca = PCA(n_components=2)
mds=MDS(n_components=2)

isomap=Isomap(n_components=2)
lle=LocallyLinearEmbedding(n_components=2)


def encode(sequens,model,sess,length=-1):
    if length == -1:
        length=len(sequens)
    vecs=[]
    s=0
    while(s<length):
        t=min(s+10000,length)
        vecs.extend(sess.run(model.hidden_state_out,feed_dict={model.data:sequens[s:t]}))
        s=t   
    return vecs

def decode(vecs,model,sess,length=-1):
    if length == -1:
        length=len(vecs)
    sequens=[]

    s=0
    while(s<length):
        t=min(s+10000,length)
        sequens.extend(sess.run(model.sample1,feed_dict={model.hidden_state_in:vecs[s:t]}))
        s=t

    return sequens

def visualization(vec1,vec2,transfered,covar1,covar2,save_path):
    # do visualization
    print("visualization path is {}".format(save_path))
    fig,ax = plt.subplots()
    
    ax.imshow(covar1, cmap=plt.get_cmap('autumn'), interpolation='nearest',origin='upper')
    #ax.imshow(covar1, cmap=plt.cm.gray, interpolation='nearest',origin='upper')
    plt.savefig(save_path+'0.png')
    plt.cla()
    fig,ax = plt.subplots()
    #ax.imshow(covar2, cmap=plt.cm.gray, interpolation='nearest',origin='upper')
    ax.imshow(covar2, cmap=plt.get_cmap('autumn'), interpolation='nearest',origin='upper')
    plt.savefig(save_path+'1.png')
    plt.cla()

    num=300
    full = np.concatenate([vec1[:num] , vec2[:num], transfered[:num]], axis = 0)
    full = pca.fit_transform(full)

    cur = 0
    l_vec1 = full[:num]
    l_vec2 = full[num:2*num]
    l_transfer = full[2*num:3*num]

    print(l_vec1.shape)
    plt.scatter(l_vec1[:,0], l_vec1[:, 1], c = 'b')
    plt.scatter(l_vec2[:, 0], l_vec2[:, 1], c= 'y')
    plt.scatter(l_transfer[:, 0], l_transfer[:, 1],c= 'g')
    plt.savefig(save_path+'_pca.png')

    

#直接从vec1 transfer到vec2所在domain
def transfer(vec1,vec2,path):#seq_num * dim_h

    m1=np.mean(vec1,axis=0)
    m2=np.mean(vec2,axis=0)
    vec1=vec1-m1
    vec2=vec2-m2

    vec1=np.transpose(vec1)#dim_h * seq_num 
    covar1=np.dot(vec1,vec1.T)/(np.shape(vec1)[1]-1)
    vec2=np.transpose(vec2)
    covar2=np.dot(vec2,vec2.T)/(np.shape(vec2)[1]-1)
  
    evals1,evecs1 = eig(covar1)
    evals1 = np.diag(np.power(np.abs(evals1),-1/2))

    evals2,evecs2 = eig(covar2)
    evals2=np.diag(np.power(np.abs(evals2),1/2))

    fc=np.dot(np.dot(np.dot(evecs1,evals1),evecs1.T),vec1)
    fcs=np.dot(np.dot(np.dot(evecs2,evals2),evecs2.T),fc)
    transfered=fcs.T+m2
       
    
    return fcs.T+m2

#储存vec1，vec2均值和得到的svd分解矩阵和，对于新来的vec进行transfer
def get_transfer_function(vec1,vec2):
    m1=np.mean(vec1,axis=0)
    m2=np.mean(vec2,axis=0)
    vec1=vec1-m1
    vec2=vec2-m2

    vec1=np.transpose(vec1)#dim_h * seq_num 
    covar1=np.dot(vec1,vec1.T)/(np.shape(vec1)[1]-1)
    vec2=np.transpose(vec2)
    covar2=np.dot(vec2,vec2.T)/(np.shape(vec2)[1]-1)

    evals1,evecs1 = eig(covar1)
    evals2,evecs2 = eig(covar2)


    def transfer_from1_to2(vec):
        vec=vec-m1
        vec=np.transpose(vec)
        evals11=np.diag(np.power(np.abs(evals1),-1/2))
        evals22=np.diag(np.power(np.abs(evals2),1/2))
        fc=np.dot(np.dot(np.dot(evecs1,evals11),evecs1.T),vec)
        fcs=np.dot(np.dot(np.dot(evecs2,evals22),evecs2.T),fc)
        transfered=fcs.T+m2
        
        return fcs.T+m2 

    def transfer_from2_to1(vec):
        vec=vec-m2
        vec=np.transpose(vec)
        evals11=np.diag(np.power(np.abs(evals1),1/2))
        evals22=np.diag(np.power(np.abs(evals2),-1/2))
        fc=np.dot(np.dot(np.dot(evecs2,evals22),evecs2.T),vec)
        fcs=np.dot(np.dot(np.dot(evecs1,evals11),evecs1.T),fc)
        transfered=fcs.T+m1
        
        return fcs.T+m1

    return transfer_from1_to2,transfer_from2_to1


#直接利用yelp data自身构建转移矩阵，transfer yelp data的向量
def transfer_vecs(sess,model,data):
    vec1=encode(data.data1,model,sess)#seq_num * dim_h
    vec2=encode(data.data2,model,sess)
  
    trans_vec1=transfer(vec1,vec2,'_0to1')
    trans_vec2=transfer(vec2,vec1,'_1to0')

    return trans_vec1,trans_vec2,data.data1,data.data2


def transfer_vecs_via_datas(sess,model,data,file_name0,file_name1):
    data1=data.load_data(file_name0)
    data2=data.load_data(file_name1)
    
    vec1=encode(data1,model,sess)
    vec2=encode(data2,model,sess)

    data_totrans1=data.load_data(FLAGS.dev_path+'.0')
    data_totrans2=data.load_data(FLAGS.dev_path+'.1')  

    vec1_totrans=encode(data_totrans1,model,sess)
    vec2_totrans=encode(data_totrans2,model,sess)

    
    transfer_from1_to2,transfer_from2_to1=get_transfer_function(vec1,vec2)
    trans_vec1=transfer_from1_to2(vec1_totrans)
    trans_vec2=transfer_from2_to1(vec2_totrans)

    return trans_vec1,trans_vec2,data_totrans1,data_totrans2




def save_and_print(sess,model,data,trans_vec1,trans_vec2,data_totrans1,data_totrans2):
    trans_vec1=trans_vec1.tolist()
    trans_vec2=trans_vec2.tolist()

    trans_seq1=decode(trans_vec1,model,sess)
    trans_seq2=decode(trans_vec2,model,sess)
   
    data1 = strip_eos(strip_pad(data1))
    data2 = [[data.id2word[i] for i in sent] for sent in data_totrans2]
    data2 = strip_eos(strip_pad(data2))

    trans_seq1 = [[data.id2word[i] for i in sent] for sent in trans_seq1]
    trans_seq1 = strip_repeat(strip_eos(strip_pad(trans_seq1)))
    trans_seq2 = [[data.id2word[i] for i in sent] for sent in trans_seq2]
    trans_seq2 = strip_repeat(strip_eos(strip_pad(trans_seq2)))


    for i in range(200):
        print("Sample {}. {} \n-> {} ".format(i, ' '.join(data1[i]),' '.join(trans_seq1[i])))

    
    with open(FLAGS.used_train_path+'.0','w') as f:
        for i in range(len(data1)):
            f.write(' '.join(data1[i])+'\n')
    with open(FLAGS.used_train_path+'.1','w') as f:
        for i in range(len(data2)):
            f.write(' '.join(data2[i])+'\n')
    
    with open(FLAGS.result_path+'.1','w') as f:
        for i in range(len(trans_seq1)):
            f.write(' '.join(trans_seq1[i])+'\n')
    with open(FLAGS.result_path+'.0','w') as f:
        for i in range(len(trans_seq2)):
            f.write(' '.join(trans_seq2[i])+'\n')