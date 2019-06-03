import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from options import FLAGS
from util import *
pca = PCA(n_components=2)
 
def visualization(vec1,vec2,transfered,covar1,covar2,save_path):
    # do visualization
    print("visualization path is {}".format(save_path))
    fig,ax = plt.subplots()
    ax.imshow(covar1, cmap=plt.cm.gray, interpolation='nearest',origin='upper')
    plt.savefig(save_path+'0.png')
    plt.cla()
    fig,ax = plt.subplots()
    ax.imshow(covar2, cmap=plt.cm.gray, interpolation='nearest',origin='upper')
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
'''
def eigenvector_visualization(sess,model,data,vec):#seq_num * dim_h
    m=np.mean(vec,axis=0)
    vec=vec-m
    vec=np.transpose(vec)
    covar=np.dot(vec,vec.T)/(np.shape(vec)[1]-1)
  

    evals,evecs=eig(covar)
    evals=np.diag(evals)

    #print("evals:{}".format(evals.tolist()))

    evecs=evecs.T.tolist()
    seqs=sess.run(model.sample1,feed_dict={model.hidden_state_in:evecs})
    seqs = [[data.id2word[i] for i in sent] for sent in seqs]
    seqs = strip_eos(strip_pad(seqs))
    
    for i in range(10):
        print(seqs[i])
        '''

def eigenvector_visualization(sess,model,data,evecs):
    evecs=evecs.tolist()
    seqs=sess.run(model.sample1,feed_dict={model.hidden_state_in:evecs})
    seqs = [[data.id2word[i] for i in sent] for sent in seqs]
    seqs = strip_eos(strip_pad(seqs))
    
    for i in range(10):
        print(seqs[i])

def get_eigenvectors(sess,model,data,file_name):
    new_data=data.load_data(file_name)
    vec=sess.run(model.hidden_state_out,feed_dict={model.data:new_data})
    m=np.mean(vec,axis=0)
    vec=vec-m
    vec=np.transpose(vec)
    covar=np.dot(vec,vec.T)/(np.shape(vec)[1]-1)
    evals,evecs=eig(covar)

    idx=np.argsort(evals)
    evals=evals[idx]
    evals=np.diag(evals)
    evecs=evecs[:,idx]
    evecs=evecs.T

    return evecs
    
def new_yelp_visualization(sess,model,data,file_name):
    evecs1=get_eigenvectors(sess,model,data,file_name+'.1')
    evecs2=get_eigenvectors(sess,model,data,file_name+'.2')
    evecs3=get_eigenvectors(sess,model,data,file_name+'.3')
    evecs4=get_eigenvectors(sess,model,data,file_name+'.4')

    eigenvector_visualization(sess,model,data,evecs1)
    eigenvector_visualization(sess,model,data,evecs2)
    eigenvector_visualization(sess,model,data,evecs3)
    eigenvector_visualization(sess,model,data,evecs4)
    num=2

    full = np.concatenate([evecs1[:num] , evecs1[:num], evecs3[:num], evecs4[:num]], axis = 0)
    print(np.shape(full))
    full = pca.fit_transform(full)
    print(full.tolist())

    A=[0,0]
    color=['r','b','g','y']
    fig=plt.figure()
    ax=fig.add_subplot(111)

    for i in range(4):
        for j in range(num):
            ax.arrow(A[0], A[1], full[i*num+j,0], full[i*num+j,1],
                     length_includes_head=True,# 增加的长度包含箭头部分
                     head_width=0.05, head_length=0.1, ec=color[i])
    

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    #plt.show()
    plt.savefig(FLAGS.result_path+'_new_yelp_visualization.png')
    plt.cla()

    

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
    #print(covar2)

    evals1,evecs1 = eig(covar1)
    #print(evals1)
    evals1 = np.diag(np.power(np.abs(evals1),-1/2))

    evals2,evecs2 = eig(covar2)
    #print(evals2)
    evals2=np.diag(np.power(np.abs(evals2),1/2))

    fc=np.dot(np.dot(np.dot(evecs1,evals1),evecs1.T),vec1)
    fcs=np.dot(np.dot(np.dot(evecs2,evals2),evecs2.T),fc)
    transfered=fcs.T+m2
    visualization(vec1.T+m1,vec2.T+m2,transfered,covar1,covar2,FLAGS.result_path+path)   
    
    return fc.T,fcs.T+m2

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
        visualization(vec1.T+m1,vec2.T+m2,transfered,covar1,covar2,FLAGS.result_path+"_0to1")
 
        return fc.T,fcs.T+m2 

    def transfer_from2_to1(vec):
        vec=vec-m2
        vec=np.transpose(vec)
        evals11=np.diag(np.power(np.abs(evals1),1/2))
        evals22=np.diag(np.power(np.abs(evals2),-1/2))
        fc=np.dot(np.dot(np.dot(evecs2,evals22),evecs2.T),vec)
        fcs=np.dot(np.dot(np.dot(evecs1,evals11),evecs1.T),fc)
        transfered=fcs.T+m1
        visualization(vec2.T+m2,vec1.T+m1,transfered,covar2,covar1,FLAGS.result_path+"_1to0")
        return fc.T,fcs.T+m1

    return transfer_from1_to2,transfer_from2_to1

#直接利用yelp data自身构建转移矩阵，transfer yelp data的向量
def transfer_vecs(sess,model,data):
    s=0
    vec1=[]
    vec2=[]
    while(s<len(data.data1)):
        t=min(s+50000,len(data.data1))
        vec1.extend(sess.run(model.hidden_state_out,feed_dict={model.data:data.data1[s:t]}))
        vec2.extend(sess.run(model.hidden_state_out,feed_dict={model.data:data.data2[s:t]}))
        s=t
    
    #vec1=sess.run(model.hidden_state_out,feed_dict={model.data:data.data1[:150000]}) #seq_num * dim_h
    #vec2=sess.run(model.hidden_state_out,feed_dict={model.data:data.data2[:150000]})
    white_vec1,trans_vec1=transfer(vec1,vec2,'_0to1')
    white_vec2,trans_vec2=transfer(vec2,vec1,'_1to0')

    return white_vec1,trans_vec1,white_vec2,trans_vec2


def transfer_vecs_via_datas(sess,model,data,data_tobe_transferd,file_name0,file_name1):
    data1=data.load_data(file_name0)
    #print('length of data1 is {}'.format(len(data1)))
    data2=data.load_data(file_name1)
    #print('length of data2 is {}'.format(len(data2)))
    s=0
    vec1=[]
    while(s<len(data1)):
        t=min(s+50000,len(data1))
        vec1.extend(sess.run(model.hidden_state_out,feed_dict={model.data:data1[s:t]}))
        s=t
    s=0
    vec2=[]
    while(s<len(data2)):
        t=min(s+50000,len(data2))
        vec2.extend(sess.run(model.hidden_state_out,feed_dict={model.data:data2[s:t]}))
        s=t


    length=len(data_tobe_transferd)

    s=0
    vec_totrans=[]
    while(s<length):
        t=min(s+50000,length)
        vec_totrans.extend(sess.run(model.hidden_state_out,feed_dict={model.data:data_tobe_transferd[s:t]}))
        s=t
    '''
    print("eigenvector visualization for vec1")
    eigenvector_visualization(sess,model,data,vec1)
    print("eigenvector visualization for vec2")
    eigenvector_visualization(sess,model,data,vec2)
    '''
    transfer_from1_to2,transfer_from2_to1=get_transfer_function(vec1,vec2)
    white_vec1,trans_vec1=transfer_from1_to2(vec_totrans)
    white_vec2,trans_vec2=transfer_from2_to1(vec_totrans)

    return trans_vec1,trans_vec2




def save_and_print(sess,model,data,data_tobe_transferd,trans_vec1,trans_vec2):
    trans_vec1=trans_vec1.tolist()
    trans_vec2=trans_vec2.tolist()


    trans_seq1=[]
    trans_seq2=[]
    s=0
    while(s<100000):
        t=min(s+5000,len(trans_vec1))
        trans_seq1.extend(sess.run(model.sample1,feed_dict={model.hidden_state_in:trans_vec1[s:t]}))
        trans_seq2.extend(sess.run(model.sample1,feed_dict={model.hidden_state_in:trans_vec2[s:t]}))
        s=t

    #trans_seq1=sess.run(model.sample1,feed_dict={model.hidden_state_in:trans_vec1})
    #trans_seq2=sess.run(model.sample1,feed_dict={model.hidden_state_in:trans_vec2})
    #trans_seq,outputs=sess.run([model.sample1,model.sample_beam_search],feed_dict={model.hidden_state_in:trans_vec1})
    data_tobe_transferd = [[data.id2word[i] for i in sent] for sent in data_tobe_transferd]
    data_tobe_transferd = strip_eos(strip_pad(data_tobe_transferd))

    trans_seq1 = [[data.id2word[i] for i in sent] for sent in trans_seq1]
    trans_seq1 = strip_eos(strip_pad(trans_seq1))
    trans_seq2 = [[data.id2word[i] for i in sent] for sent in trans_seq2]
    trans_seq2 = strip_eos(strip_pad(trans_seq2))


    '''
    outputs = [[[data.id2word[i] for i in s]for s in sent] for sent in outputs]
    outputs = strip_eos(strip_pad(outputs))
    '''

    for i in range(200):
        print("Sample {}.\n {} \n-> {} \n -> {}".format(
            i, ' '.join(data_tobe_transferd[i]),' '.join(trans_seq1[i]),' '.join(trans_seq2[i])))
        #for o in outputs[i]:
        #    print("{}\n".format(' '.join(o)))
    
    with open(FLAGS.used_train_path,'w') as f:
        for i in range(len(data_tobe_transferd)):
            f.write(' '.join(data_tobe_transferd[i])+'\n')

    
    with open(FLAGS.result_path+'.1','w') as f:
        for i in range(len(trans_seq1)):
            f.write(' '.join(trans_seq1[i])+'\n')
    with open(FLAGS.result_path+'.0','w') as f:
        for i in range(len(trans_seq2)):
            f.write(' '.join(trans_seq2[i])+'\n')