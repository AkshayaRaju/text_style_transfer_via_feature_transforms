import nltk
import os
import random
import multiprocessing
from multiprocessing import Pool

def run_f(ele):
    reference, fn, weight = ele
    BLEUscore_f = nltk.translate.bleu_score.sentence_bleu(reference, fn, weight,
                                                          smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)  
    return BLEUscore_f


def bleu_eval(reference, hypothesis_list):
    SAMPLES = len(hypothesis_list)
    
    for ngram in range(2, 6):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(multiprocessing.cpu_count() - 4)
        bleu_irl = pool.map(run_f, [(reference, hypothesis_list[i], weight) for i in range(SAMPLES)])
        pool.close()
        pool.join()

        print(len(weight), '-gram BLEU(f) score : ', 1.0 * sum(bleu_irl) / len(bleu_irl))


def read_file(path):
    f = open(path, 'r')
    return [line for line in f]


if __name__ == '__main__':
    PREFIX = "/home/morino/code/text_style_transfer_via_feature_transforms/data/fake_yelp"
    reference = read_file(os.path.join(PREFIX, 'reference.txt'))
    hypothesis = read_file(os.path.join(PREFIX, 'hypothesis.txt'))
    random.shuffle(hypothesis)
    bleu_eval(reference, hypothesis[:5000])
