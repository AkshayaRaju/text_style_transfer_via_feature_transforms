import nltk
import random
import _pickle as cPickle
import multiprocessing
from multiprocessing import Pool
from options import FLAGS

# SAMPLES = 200
#REFSIZE = 5000

    
def run_f(ele):
    reference, fn, weight = ele
    reference=[reference]
    BLEUscore_f = nltk.translate.bleu_score.sentence_bleu(reference, fn, weight,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)  
    return BLEUscore_f


def bleu_eval(generate_file=None,reference_file=None,SIZE=3000):

    if generate_file is None:
        generate_file=FLAGS.result_path+'.1'
        print("hypothesis path is {}".format(generate_file))
    if reference_file is None:
        reference_file=FLAGS.train_path+'.1'
        print("reference path is {}".format(reference_file))
    hypothesis_list = [] #keep 4000
 
    #################################################
    ## output generated sentences
    hypothesis_list=[]
    f=open(generate_file,'r')
    for line in f:
        if(len(hypothesis_list)<SIZE):
            hypothesis_list.append(line.split())
    f.close()

    f=open(reference_file,'r')
    reference_list = [] #keep 4000
    for line in f:
        if(len(reference_list)<10000):
            reference_list.append(line.split())
    f.close()

    random.shuffle(hypothesis_list)


    for ngram in range(1, 4):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(multiprocessing.cpu_count() - 4)
        bleu = pool.map(run_f, [(reference_list, hypothesis_list[i], weight) for i in range(SIZE)])
     
        pool.close()
        pool.join()

        print(len(weight), '-gram BLEU(b) score : ', 1.0 * sum(bleu) / len(bleu))

def bleu_eval_pair(generate_file=None,reference_file=None,SIZE=5000):
    if generate_file is None:
        generate_file=FLAGS.result_path
        print("hypothesis path is {}".format(generate_file))
    if reference_file is None:
        reference_file=FLAGS.used_train_path
        print("reference path is {}".format(reference_file))
 
    #################################################
    ## output generated sentences
    hypothesis_list=[]
    f=open(generate_file+'.0','r')
    for line in f:
        #if(len(hypothesis_list)<SIZE/2):
        hypothesis_list.append(line.split())
    f.close()
    f=open(generate_file+'.1','r')
    for line in f:
        #if(len(hypothesis_list)<SIZE):
        hypothesis_list.append(line.split())
    f.close()
    print("bleu hypothesis length is {}".format(len(hypothesis_list)))
 

    reference_list = [] 
    f=open(reference_file+'.1','r')
    for line in f:
        #if(len(reference_list)<SIZE/2):
        reference_list.append(line.split())
    f.close()
    f=open(reference_file+'.0','r')
    for line in f:
        #if(len(reference_list)<SIZE):
        reference_list.append(line.split())
    f.close()
    print("bleu reference length is {}".format(len(reference_list)))

    bleu=-1
    for ngram in range(1, 5):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(multiprocessing.cpu_count() - 4)
        bleu = pool.map(run_f, [(reference_list[i], hypothesis_list[i], weight) for i in range(SIZE)])
     
        pool.close()
        pool.join()
        
        print(len(weight), '-gram BLEU(b) score : ', 1.0 * sum(bleu) / len(bleu))
    return sum(bleu) / len(bleu)

if __name__ == '__main__':

    #bleu_eval_pair('/home/mlsnrs/data/bns/baseline1/tmp/sentiment.dev.epoch20','/home/mlsnrs/data/hrz/transfer4/data/yelp/dev')
    bleu_eval_pair('/home/mlsnrs/data/bns/baseline3/trainsamples/dev_transfer','/home/mlsnrs/data/bns/baseline3/trainsamples/dev_origin')
   

