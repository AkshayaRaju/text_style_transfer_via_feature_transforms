import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


PREFIX = ""
dataset='yelp'

flags.DEFINE_string('train_path',PREFIX + 'data/'+dataset+'/train', "path for training dataset")
flags.DEFINE_string('dev_path',PREFIX + 'data/'+dataset+'/dev', "path for validation dataset")
flags.DEFINE_string('test_path',PREFIX + 'data/'+dataset+'/test', "path for test dataset")
flags.DEFINE_string('used_train_path',PREFIX,"save the used training data")
flags.DEFINE_integer('minlen', 3, "minimum size of a legal sent")
flags.DEFINE_integer('maxlen', 15, "maximal size of a legal sent")

flags.DEFINE_string('embedding_path',PREFIX + 'data/'+dataset+'/embs_30_out', "path for embedding")
flags.DEFINE_integer('dim_h', 30, "dimension of encoder hidden state")
flags.DEFINE_integer('dim_e', 30, "dimension of embedding")
flags.DEFINE_string('eval_embedding_path',PREFIX + 'data/'+dataset+'/embs_300_out', "path for embedding of cnn classifier")
flags.DEFINE_integer('eval_dim_e', 300, "dimension of embedding")
flags.DEFINE_integer('eval_dim_h', 300, "dimension of encoder hidden state")
flags.DEFINE_string('result_path',PREFIX,'save the transferred sentences')#在demo中赋值
flags.DEFINE_string('transfer_path',PREFIX,'save the sentences with high confidence to calculate style matrix')
flags.DEFINE_integer('batch_size',200, "batch size")
flags.DEFINE_integer('vocab_size', 0, "size of vocabulary")

flags.DEFINE_integer('style_num', 2, "number of style")
flags.DEFINE_float('dropout_rate', 0.1, "rate of drop out")
flags.DEFINE_float("gumbel_gamma", 1, "gamma defined in gumbel softmax") ## may need to let the gumbel gamma to adopt the annealing strategy
flags.DEFINE_string('summaries_dir', PREFIX + 'tensorboard/'+dataset, 'path for summary')
flags.DEFINE_integer("pretrain_epochs", 1, "max epoch number")
flags.DEFINE_integer("max_epochs", 32, "max epoch number")
flags.DEFINE_float("classifier_keep_prob",0.9,"keep_prob of classifier ")
flags.DEFINE_string("filter_sizes","1,2,3,4,5","filter_sizes of the cnn classifier")
flags.DEFINE_integer("output_channel",100,"number of output channel")
flags.DEFINE_float("lm_keep_prob",0.9,"keep_prob of language model")



