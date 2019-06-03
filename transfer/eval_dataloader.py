
from collections import Counter
from options import FLAGS
import numpy as np
from gensim.models import word2vec
import csv
import sys
import random


def load_sentence(path, maxlen=1000, minlen=5, max_size=-1):
	sents = []
	with open(path) as f:
		for line in f:
			if len(sents) == max_size:
				break
			toks = line.split()
			if(len(toks) <= maxlen and len(toks) >= minlen):
				sents.append(toks)
	return sents


def get_batch(seqs, labels, word2id, batch_size,noisy=False):
	pad = word2id['<pad>']
	go = word2id['<go>']
	eos = word2id['<eos>']
	unk = word2id['<unk>']

	x_eos = []
	go_x = []
	maxlen = max([len(sent) for sent in seqs])

	weights=[]
	## do the padding
	for sent in seqs:
		sent_id = [word2id[w] if w in word2id else unk for w in sent]
		l = len(sent)
		weights.append([1.0]*l+[0.0]*(maxlen-l+1))
		padding = [pad] * (maxlen - l)
		new_sent = sent_id + [eos] + padding
		x_eos.append(new_sent)
		go_x.append([go]+new_sent[:-1])

	batch=[]
	s=0
	while s<len(seqs):
		t=min(len(seqs),s+batch_size)
		batch.append({'seqs':x_eos[s:t],'labels':labels[s:t],
			'weights':weights[s:t],'seqs_shift':go_x[s:t]})
		s=t
	
	return batch



#以训练集的内容建立词表
class eval_dataloader(object):
	def __init__(self,train_path,maxlen,minlen,batch_size):
		self.batch_size = batch_size
		self.minlen=minlen
		self.maxlen=maxlen
		
		datas1 = load_sentence(train_path + '.0', maxlen = maxlen, minlen = minlen)
		datas2 = load_sentence(train_path + '.1', maxlen = maxlen, minlen = minlen)
		if len(datas1) <len(datas2):
			datas2 = datas2[:len(datas1)] 
		else:
			datas1 = datas1[:len(datas2)]
		print('#sents of training file 0: {}'.format(len(datas1)))
		print('#sents of training file 1: {}'.format(len(datas2)))
		self.word2id = {'<pad>':0, '<go>':1, '<eos>':2, '<unk>':3}
		self.id2word = ['<pad>','<go>','<eos>','<unk>']
		
		datas = datas1 + datas2
		labels = [0]*len(datas1)+[1]*len(datas2)
		minconut = 3
		#pretrain_emb(datas,minconut)

		words = [word for sent in datas for word in sent]
		cnt = sorted(Counter(words).items(), key=lambda obj: obj[0])
		
		cout = 0
		for word in cnt:
			if word[1] >= minconut and word[0]!='<pad>' and word[0]!='<go>' and word[0]!='<eos>' and word[0]!='<unk>':
				self.word2id[word[0]] = len(self.word2id)
				self.id2word.append(word[0])
			else:
				cout += 1

		
		self.vocab_size = len(self.word2id)
		print("vocabulary size is {}".format(self.vocab_size))

		print("unknow num:{}".format(cout))

		
		#mix the data of different labels
		index=list(range(len(datas)))
		random.shuffle(index)
		datas=np.array(datas)
		datas=datas[index]
		labels=np.array(labels)
		labels=labels[index]

		# sort the batch in sentence length, which will make the training more smooth
		z = sorted(zip(datas,labels),key=lambda i:len(i[0]))
		self.datas,self.labels=zip(*z)
		
		#for i in range(20):
		#	print(datas[i])
		#	print(labels[i])
		

		self.batch=get_batch(self.datas,self.labels,self.word2id,self.batch_size)


	#利用训练集的词表，加载其他测试集、验证集	
	def load_datas(self,datapath):
		datas1 = load_sentence(datapath + '.0', maxlen = self.maxlen, minlen = self.minlen)
		datas2 = load_sentence(datapath + '.1', maxlen = self.maxlen, minlen = self.minlen)
		datas=datas1+datas2
		labels=[0]*len(datas1)+[1]*len(datas2)
		
		return get_batch(datas,labels,self.word2id,self.batch_size),labels
		
	
	def load_datas_pos(self,datapath):
		datas = load_sentence(datapath + '.1', maxlen = self.maxlen, minlen = self.minlen)		
		labels=[1]*len(datas)
		return get_batch(datas,labels,self.word2id,self.batch_size),labels

	def load_datas_neg(self,datapath):
		datas = load_sentence(datapath + '.0', maxlen = self.maxlen, minlen = self.minlen)		
		labels=[0]*len(datas)
		return get_batch(datas,labels,self.word2id,self.batch_size),labels

	

if __name__ == '__main__':
	data = classifier_dataloader(FLAGS.train_path,1000,FLAGS.minlen,FLAGS.batch_size)
	dev_batch,dev_labels=data.load_datas(FLAGS.dev_path)   

	'''
	batch = data.next_batch()
	a = [[data.id2word[i] for i in id_seq] for id_seq in batch["src_seq"]]
	print(batch["src_seq"][0])
	print(a[0])
	'''








