#-*- coding:utf-8 -*-

import tensorflow as tf
import csv

MAX_LEN_sentence = 40
MAX_NUM_sentence = 30

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
  """Vocabulary class for mapping between words and ids (integers)"""

  def __init__(self, vocab_file=r'/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/vocab.txt', max_size=50000):
    """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

    Args:
      vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
      max_size: integer. The maximum size of the resulting Vocabulary."""
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          print 'Warning: incorrectly formatted line in vocabulary file: %s\n' % line
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count)
          break

    print "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1])

  def word2id(self, word):
    """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    """Returns the word (string) corresponding to an id (integer)."""
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    """Returns the total size of the vocabulary"""
    return self._count

  def write_metadata(self, fpath):
    """Writes metadata file for Tensorboard word embedding visualizer as described here:
      https://www.tensorflow.org/get_started/embedding_viz

    Args:
      fpath: place to write the metadata file
    """
    print "Writing word embedding metadata file to %s..." % (fpath)
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in xrange(self.size()):
        writer.writerow({"word": self._id_to_word[i]})


#卷积过程
def conv2d(x,w,strides=[1,2,2,1]):
    return tf.nn.conv2d(x,w,
                        strides,padding='SAME')
def depthwise_conv2d(x,w,strides=[1,2,2,1]):
    return tf.nn.depthwise_conv2d(x,w,
                                  strides,padding='SAME')
#池化过程
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                        strides=[1,2,2,1],padding='SAME')

def onehot(y):
    '''转化为one-hot 编码'''
    size1 = tf.size(y)
    y = tf.expand_dims(y, 1)
    indices = tf.expand_dims(tf.range(0, size1, 1), 1)
    concated = tf.concat([indices, y], 1)
    y = tf.sparse_to_dense(concated, tf.stack([size1, 10]), 1.0, 0.0)
    return y

class myModel():
    def __init__(self, looP=200000, batch_Size=32, model_Save_path='mynet/save_net.ckpt',
                 Vocab_size = 50000,embedding_Dim=20):
        self.loop = looP
        self.batch_size = batch_Size
        self.model_save_path = model_Save_path
        self.vocab_size = Vocab_size
        self.embedding_dim = embedding_Dim
        # dropoup参数
        self.keep_prob = tf.placeholder("float")
        self.x_placeholder = tf.placeholder(tf.int32, shape=[self.batch_size,MAX_NUM_sentence,MAX_LEN_sentence])
        self.dense_w = {
            "w_conv1": tf.Variable(tf.truncated_normal([7, 7, 20, 32], stddev=0.13), name="w_conv1"),
            "b_conv1": tf.Variable(tf.constant(0.13, shape=[32]), name="b_conv1"),
            "w_conv2": tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="w_conv2"),
            "b_conv2": tf.Variable(tf.constant(0.13, shape=[64]), name="b_conv2"),
            "w_conv3": tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.13), name="w_conv3"),
            "b_conv3": tf.Variable(tf.constant(0.13, shape=[128]), name="b_conv3"),
            "w_fc1": tf.Variable(tf.truncated_normal([MAX_NUM_sentence * MAX_LEN_sentence * 128, 128], stddev=0.13), name="w_fc1"),
            "b_fc1": tf.Variable(tf.constant(0.13, shape=[128]), name="b_fc1"),
            "w_fc2": tf.Variable(tf.truncated_normal([128, 128], stddev=0.13), name="w_fc2"),
            "b_fc2": tf.Variable(tf.constant(0.13, shape=[128]), name="b_fc2"),
            "w_fc3": tf.Variable(tf.truncated_normal([128, MAX_NUM_sentence*3], stddev=0.13), name="w_fc3"),
            "b_fc3": tf.Variable(tf.constant(0.13, shape=[MAX_NUM_sentence*3]), name="b_fc3")
        }
        self.deepwise_w_conv11 = tf.Variable(tf.truncated_normal([7, 7, 20, 2], stddev=0.13), name="deepwise_w_conv11")
        self.deepwise_b_conv11 = tf.Variable(tf.truncated_normal([20*2], stddev=0.13), name="deepwise_b_conv11")
        self.deepwise_w_conv12 = tf.Variable(tf.truncated_normal([1, 1, 20*2,32], stddev=0.13), name="deepwise_w_conv12")
        self.deepwise_b_conv12 = tf.Variable(tf.truncated_normal([32], stddev=0.13), name="deepwise_b_conv12")

    def gennet(self):
        embedding = tf.Variable(tf.truncated_normal(shape=[self.vocab_size, self.embedding_dim], stddev=0.1),name='embedding')
        input_x_embedded = tf.nn.embedding_lookup(embedding, self.x_placeholder)
        # print "mark___zsp"
        # print input_x_embedded
        # self.mark = input_x_embedded
        # input_x_embedded = tf.bitcast(input_x,tf.float32)
        # input_x_embedded = tf.reshape(input_x_embedded,[self.batch_size,30,40,1])

        # 第一卷积层
        # h_conv1 = tf.nn.relu(conv2d(input_x_embedded, self.dense_w["w_conv1"], [1, 1, 1, 1]) + self.dense_w["b_conv1"])


        #deepwise conv
        h_conv11 = tf.nn.relu(depthwise_conv2d(input_x_embedded, self.deepwise_w_conv11, [1, 1, 1, 1]) + self.deepwise_b_conv11)
        h_conv12 = tf.nn.relu(conv2d(h_conv11, self.deepwise_w_conv12, [1, 1, 1, 1]) + self.deepwise_b_conv12)
        h_conv1 = h_conv12


        # 第二卷积层
        h_conv2 = tf.nn.relu(conv2d(h_conv1, self.dense_w["w_conv2"], [1, 1, 1, 1]) + self.dense_w["b_conv2"])

        # 第三卷积层
        h_conv3 = tf.nn.relu(conv2d(h_conv2, self.dense_w["w_conv3"], [1, 1, 1, 1]) + self.dense_w["b_conv3"])

        # 第一全连接层
        # h_conv3_flat = tf.reshape(h_conv3,[-1,24*24*(int(128/self.ratio)+1)])
        h_conv3_flat = tf.reshape(h_conv3, [-1, MAX_NUM_sentence * MAX_LEN_sentence * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.dense_w["w_fc1"]) + self.dense_w["b_fc1"])

        # 第二全连接层
        # h_fc2 = tf.nn.softmax(tf.matmul(h_fc1, self.dense_w["w_fc2"]) + self.dense_w["b_fc2"])
        h_fc2 = tf.matmul(h_fc1, self.dense_w["w_fc2"]) + self.dense_w["b_fc2"]

        # 防止过拟合的dropout
        # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        h_fc2_drop =h_fc2
        # 第三全连接层
        # y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, self.dense_w["w_fc3"]) + self.dense_w["b_fc3"])
        y_conv = tf.matmul(h_fc2_drop, self.dense_w["w_fc3"]) + self.dense_w["b_fc3"]
        
        return y_conv
