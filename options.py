import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


#PREFIX = "/Users/huangruozi/Desktop/transfer/transfer1/"
# PREFIX = "/home/mlsnrs/data/hrz/transfer2/"
# PREFIX = "/home/morino/code/text_style_transfer_via_feature_transforms/"
PREFIX = '/home/mlsnrs/data/pxd/text_style_transfer_via_feature_transforms/'

flags.DEFINE_string('train_path',PREFIX + 'data/yelp/sentiment.train', "path for training dataset")
flags.DEFINE_integer('minlen', 5, "minimum size of a legal sent")
flags.DEFINE_integer('maxlen', 10, "maximal size of a legal sent")
flags.DEFINE_string('vocab_path', PREFIX + 'data/yelp/yelp.vocab', "path for vocabulary storage")
flags.DEFINE_string('embedding_path',PREFIX + 'data/embs_500', "path for embedding")
flags.DEFINE_integer('batch_size',200, "batch size")
flags.DEFINE_integer('vocab_size', 0, "size of vocabulary")
flags.DEFINE_integer('dim_h', 500, "dimension of encoder hidden state")
flags.DEFINE_integer('dim_e', 500, "dimension of embedding")
flags.DEFINE_integer('style_num', 2, "number of style")
flags.DEFINE_float('dropout_rate', 0.1, "rate of drop out")
flags.DEFINE_float("gumbel_gamma", 1, "gamma defined in gumbel softmax") ## may need to let the gumbel gamma to adopt the annealing strategy
flags.DEFINE_string('summaries_dir', PREFIX + 'data/yelp', 'path for summary')
flags.DEFINE_integer("pretrain_epochs", 1, "max epoch number")
flags.DEFINE_integer("max_epochs", 32, "max epoch number")
flags.DEFINE_float("classifier_keep_prob",1,"keep_prob of classifier ")


flags.DEFINE_integer('dim_c_h', 16, "dimension of classifier hidden state")

flags.DEFINE_bool('dev', True, "toggle for development mode")
flags.DEFINE_bool('train', True, "toggle for train mode")
flags.DEFINE_bool('test', False, "toggle for test mode")
flags.DEFINE_string('dev_path', PREFIX + 'transfer/data/yelp_syn/sentiment.dev.new', "path for development dataset")

flags.DEFINE_string('test_path', PREFIX + 'transfer/data/yelp_syn/sentiment.test', "path for test dataset")

flags.DEFINE_string('ckpt_path', PREFIX + 'transfer/data/yelp_syn/drive-download-20180827T030617Z-001/model_e_{}.ckpt', 'path template for checkpoints')
flags.DEFINE_string('cipher_path', PREFIX + 'transfer/data/yelp_syn/cipher.pickle', 'path for ciphers')
flags.DEFINE_string('vector_file_path', PREFIX + 'transfer/data/yelp_syn/vectors_syn_50.txt', 'pretrained embedding path')
## this is for the synthetic case
