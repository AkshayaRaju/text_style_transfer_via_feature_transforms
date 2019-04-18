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
	
	'''
	sentences = word2vec.Text8Corpus('data/text8')
	model = word2vec.Word2Vec(sentences,min_count=10,size=FLAGS.dim_e)
	model.save('data/text8_50.model')
	model.build_vocab(datas,update=True)
	model.train(datas,total_examples=model.corpus_count,epochs=model.epochs)
	'''
	model = word2vec.Word2Vec.load('data/text8_50.model')
	
	model.wv.save_word2vec_format('data/embs_50')



def make_up(_x,n):
	x =[]
	for i in range(n):
		x.append(_x[i % len(_x)])
	return x

def get_batch(x, y, word2id, maxlen, noisy=False):#x:sequences 2batch_size   y: style 2batch_size
	pad = word2id['<pad>']
	go = word2id['<go>']
	eos = word2id['<eos>']
	unk = word2id['<unk>']

	x_eos = []
	x_shifted = []
	#maxlen = max([len(sent) for sent in x])


	## do the padding
	for sent in x:
		sent_id = [word2id[w] if w in word2id else unk for w in sent]
		l = len(sent)
		padding = [pad] * (maxlen - l)
		new_sent = sent_id + [eos] + padding
		x_eos.append(sent_id + [eos] + padding)
		x_shifted.append([eos] + new_sent[:-1])



	## exchange the source and target
	if(np.random.rand() <= 0.5):
		x_eos.reverse()
		x_shifted.reverse()
		y.reverse()

	half = len(y) // 2
	seqs = np.array(x_eos)
	shifted_seqs = np.array(x_shifted)
	y = np.array(y)
	src_seq = seqs[0:half, :]
	shifted_src_seq = shifted_seqs[0:half, :]
	tgt_seq = seqs[half:, :]
	tgt_style = y[half:]
	src_style = y[0:half]


	return {
		"src_seq" : src_seq, #batch_size * maxlen 
		"tgt_seq" : tgt_seq,
		"src_style" : src_style,
		"tgt_style" : tgt_style,
		"len" : maxlen + 1, ## with the eos
		"shifted_src_seq" : np.concatenate((shifted_src_seq, shifted_src_seq), axis = 0)
	}
	'''
	return {
		"src_seq" : np.concatenate((src_seq, src_seq), axis = 0), #batch_size * maxlen 
		"tgt_seq" : np.concatenate((tgt_seq, src_seq), axis = 0),
		"src_style" : np.concatenate((src_style, src_style), axis = 0),
		"tgt_style" : np.concatenate((tgt_style, src_style), axis = 0),
		"len" : maxlen + 1, ## with the eos
		"shifted_src_seq" : np.concatenate((shifted_src_seq, shifted_src_seq), axis = 0)
	}
	'''

class yelp_dataloader(object):
	def __init__(self,train_path,maxlen,minlen,batch_size):
		self.batch_size = batch_size
		self.maxlen = maxlen

		datas1 = load_sentence(train_path + '.0', maxlen = maxlen, minlen = minlen)
		datas2 = load_sentence(train_path + '.1', maxlen = maxlen, minlen = minlen)
		print('#sents of training file 0: {}'.format(len(datas1)))
		print('#sents of training file 1: {}'.format(len(datas2)))
		self.word2id = {'<pad>':0, '<go>':1, '<eos>':2, '<unk>':3}
		self.id2word = ['<pad>','<go>','<eos>','<unk>']
		

		datas = datas1 + datas2
		minconut = 3
		#pretrain_emb(datas,minconut)

		words = [word for sent in datas for word in sent]
		cnt = Counter(words)
		
		cout = 0
		for word in cnt:
			if cnt[word] >= minconut and word!='<pad>' and word!='<go>' and word!='<eos>' and word!='<unk>':
				self.word2id[word] = len(self.word2id)
				self.id2word.append(word)
			else:
				cout += 1

		
		self.vocab_size = len(self.word2id)
		print("vocabulary size is {}".format(self.vocab_size))

		print("unknow num:{}".format(cout))

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
		t = s + batch_size
		while t < n:
			self.batches.append(get_batch(datas1[s:t] + datas2[s:t],
			    [0]*(t-s) + [1]*(t-s), self.word2id, self.maxlen))
			s = t
			t = s + batch_size
		self.batch_num = len(self.batches)
		self.pointer = -1

	def next_batch(self):
		self.pointer = (self.pointer + 1) % self.batch_num
		return self.batches[self.pointer]

	def reset(self):
		self.pointer = -1
	
	def random_batch(self):
		self.pointer = np.random.randint(low = 0, high = self.batch_num-1)
		return self.batches[self.pointer]


if __name__ == '__main__':
	data = yelp_dataloader(FLAGS.train_path,FLAGS.maxlen,FLAGS.minlen,FLAGS.batch_size)

	'''
	batch = data.next_batch()
	a = [[data.id2word[i] for i in id_seq] for id_seq in batch["src_seq"]]
	print(batch["src_seq"][0])
	print(a[0])
	'''








