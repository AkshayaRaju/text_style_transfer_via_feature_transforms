from collections import Counter
from options import FLAGS
import numpy as np
from gensim.models import word2vec
import csv



def load_sentence(path, maxlen=20, minlen=5, max_size=-1):
	sents = []
	with open(path) as f:
		for line in f:
			if len(sents) == max_size:
				break
			toks = line.split()
			if(len(toks) <= maxlen and len(toks) >= minlen):
				sents.append(toks)
	return sents

#预训练词向量模型
def pretrain_emb(data,minconut):
	
	#先使用外部数据集text8，增量的加入yelp数据集
	'''
	sentences = word2vec.Text8Corpus('data/text8')
	model = word2vec.Word2Vec(sentences,min_count=10,size=FLAGS.dim_e)
	#model.save('data/text8_50.model')
	model.build_vocab(data,update=True)
	model.train(data,total_examples=model.corpus_count,epochs=model.epochs)
	'''
	

	#直接使用yelp数据集

	model = word2vec.Word2Vec(iter=1)
	#model.save('data/text8_50.model')
	model.build_vocab(data)
	model.train(data,total_examples=model.corpus_count,epochs=model.epochs)
	
	

	#model = word2vec.Word2Vec.load('data/text8_50.model')	
	model.wv.save_word2vec_format('data/embs_100')



def make_up(_x,n):
	x =[]
	for i in range(n):
		x.append(_x[i % len(_x)])
	return x

def get_batch(seq, src_syl,tar_syl, word2id, maxlen, noisy=False):#x:sequences 2batch_size   y: style 2batch_size
	pad = word2id['<pad>']
	go = word2id['<go>']
	eos = word2id['<eos>']
	unk = word2id['<unk>']

	x_eos = []
	x_shift = []
	#maxlen = max([len(sent) for sent in x])


	## do the padding
	for i  in range(len(seq)):
		sent_id = [word2id[w] if w in word2id else unk for w in seq[i]]
		l = len(seq[i])
		padding = [pad] * (maxlen - l)
		new_sent = sent_id + [eos] + padding
		x_eos.append(new_sent)
		x_shift.append([go]+new_sent[:-1])



	src_style = np.array(src_syl)
	src_seq = np.array(x_eos)
	tgt_style = np.array(tar_syl)
	seq_shift = np.array(x_shift)


	return {
		"src_seq" : src_seq, #batch_size * maxlen 
		"shifted_src_seq":seq_shift,
		"src_style" : src_style,
		"tgt_style" : tgt_style,
		"len" : maxlen + 1, ## with the eos

	}
def get_data_id(datas,word2id,maxlen):
	pad = word2id['<pad>']
	eos = word2id['<eos>']
	unk = word2id['<unk>']
	d=[]
	for data in datas:
		sent_id = [word2id[i] if i in word2id else unk for i in data]
		l = len(data)
		padding = [pad] * (maxlen - l)
		new_sent = sent_id + [eos] + padding
		d.append(new_sent)
	return d

class yelp_dataloader(object):
	def __init__(self,train_path,maxlen,minlen,batch_size):
		self.batch_size = batch_size
		self.maxlen = maxlen

		datas1 = load_sentence(train_path + '4w.0', maxlen = maxlen, minlen = minlen)
		datas2 = load_sentence(train_path + '4w.1', maxlen = maxlen, minlen = minlen)
		'''
		f=open(train_path + '4w.1','w')
		for i in range(40000):
			f.write(" ".join(datas2[i]))
			f.write('\n')
			'''
		print('#sents of training file 0: {}'.format(len(datas1)))
		print('#sents of training file 1: {}'.format(len(datas2)))
		self.word2id = {'<go>':0,'<pad>':1,'<eos>':2, '<unk>':3}
		self.id2word = ['<go>','<pad>','<eos>','<unk>']
		

		datas = datas1 + datas2
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

		self.data1=get_data_id(datas1,self.word2id,self.maxlen)
		self.data2=get_data_id(datas2,self.word2id,self.maxlen)

		if len(datas1) <len(datas2):
			datas1 = make_up(datas1,len(datas2))
		else:
			datas2 = make_up(datas2,len(datas1))


		# sort the batch in sentence length, which will make the training more smooth
		datas1 = sorted(datas1,key=lambda i:len(i))
		datas2 = sorted(datas2,key=lambda i:len(i))

		self.batches = []
		n = len(datas1)
		s = 0
		t = s + int(batch_size/2)
		while t < n:
			self.batches.append(get_batch(datas1[s:t] + datas2[s:t],
			    [0]*(t-s) + [1]*(t-s), [1]*(t-s) + [0]*(t-s),self.word2id, self.maxlen))
			s = t
			t = s + int(batch_size/2)

		'''
		self.batches = []
		n = len(datas1)
		s = 0
		t = s + batch_size
		while t < n:
			self.batches.append(get_batch(datas1[s:t], [0]*(t-s), [1]*(t-s),self.word2id, self.maxlen))
			s = t
			t = s + batch_size
		n = len(datas2)
		s = 0
		t = s+batch_size
		while t<n:
			self.batches.append(get_batch(datas2[s:t], [1]*(t-s), [0]*(t-s),self.word2id, self.maxlen))
			s = t
			t = s + batch_size
		'''


		self.batch_num = len(self.batches)
		self.pointer = -1


	def next_batch(self):
		self.pointer = (self.pointer + 1) % self.batch_num
		return self.batches[self.pointer]

	def reset(self):
		self.pointer = -1
	
	def random_batch(self):
		point_ran = np.random.randint(low = 0, high = self.batch_num-1)
		return self.batches[point_ran]


if __name__ == '__main__':
	data = yelp_dataloader(FLAGS.train_path,FLAGS.maxlen,FLAGS.minlen,FLAGS.batch_size)

	'''
	batch = data.next_batch()
	a = [[data.id2word[i] for i in id_seq] for id_seq in batch["src_seq"]]
	print(batch["src_seq"][0])
	print(a[0])
	'''















