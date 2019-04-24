# to build the vocabulary

PAD_token = 0
BOS_token = 1
EOS_token = 2
UNK_token = 3

class Vocab(object):
    def __init__(self, name):
        self.name = name
        self.word2id = dict()
        self.id2word = dict()
        self.vocab_size = 4
        self.word_counter = dict()
        self.trimmed = False
        self.word2id = {"<PAD>":PAD_token, "<BOS>": BOS_token, "<EOS>": EOS_token, '<UNK>': UNK_token}
        self.id2word = {PAD_token: "<PAD>", BOS_token:"<BOS>", EOS_token:"<EOS>", UNK_token: '<UNK>'}
        self.vocab_size = 4
  

    # given a list of int tokens, parse out the underlying sentence until 
    def parse(self, tokens):
        out = []
        for t in tokens:
            if t == EOS_token:
                break
            elif t == PAD_token:
                continue
            else:
                out.append(self.id2word[t])
        return " ".join(out)
        

    def add_sentence(self, seq):
        for word in seq:
            self.add_word(word)
        return

    def add_word(self, word):
        if not (word in self.word2id):
            self.word2id[word] = self.vocab_size
            self.id2word[self.vocab_size] = word
            self.word_counter[word] = 1
            self.vocab_size += 1            
        else:
            self.word_counter[word] += 1

    def trim(self, min_count):
        if self.trimmed: return

        previous_vocab_size = self.vocab_size
        # begin trimming, assume it would only be invoked when counter has been constructed
        keep_words = [word for word in self.word_counter if self.word_counter[word] >= min_count]

        self.word2id = {"<PAD>":PAD_token, "<BOS>": BOS_token, "<EOS>": EOS_token, '<UNK>': UNK_token}
        self.id2word = {PAD_token: "<PAD>", BOS_token:"<BOS>", EOS_token:"<EOS>", UNK_token: '<UNK>'}
        self.vocab_size = 4
        
        for word in keep_words:
            self.word2id[word] = self.vocab_size
            self.id2word[self.vocab_size] = word
            self.vocab_size += 1
        print("{} ratio types trimmed.".format(1.0 - float(self.vocab_size) / previous_vocab_size))
        
