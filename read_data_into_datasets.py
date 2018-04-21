# -*- coding: utf-8 -*-

import os
import  numpy as np
import re
import cnn_model
import tensorflow as tf

vocab = cnn_model.Vocab()
MAX_LEN_sentence = 40
MAX_NUM_sentence = 30

def genRecord(img_data, label):
    #img_bytes = img_data[0]
    data_feature = img_data
    data_label = label

    """
    proto结构
    message Features {
        map<string, Feature> feature = 1;
    };
    message Feature {
      oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
      }
    };
    """
    features = tf.train.Features(feature={
        'data_feature_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_feature])),
        "data_label_bytes": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_label]))
    })
    return features

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader(name=None,options=None)
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={'data_feature_bytes' : tf.FixedLenFeature([], tf.string),
                                                 'data_label_bytes': tf.FixedLenFeature([], tf.string),
                                       }
                                       )

    raw_bytes1 = tf.decode_raw(features['data_feature_bytes'], tf.int32)
    data_feature = tf.reshape(raw_bytes1, [MAX_NUM_sentence,MAX_LEN_sentence])
    raw_bytes2 = tf.decode_raw(features['data_label_bytes'], tf.float32)
    data_label = tf.reshape(raw_bytes2, [MAX_NUM_sentence*3])
    #print("mark11111",img)

    return data_feature, data_label


def main(f_path = r'/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/training'):
    datasets_file_path = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn' + '/train.data'
    opts = None  # tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path=datasets_file_path, options=opts)
    for root, dirs, files in os.walk(f_path):
        D = 0
        for file in files:
            D +=1
            data_feature = np.zeros([MAX_NUM_sentence,MAX_LEN_sentence],dtype=np.int32)
            data_label = np.zeros([MAX_NUM_sentence,3],dtype = np.float32)
            data_sentence_index =0
            file_path = os.path.join(root, file)
            #print file_path
            with open(file_path, 'r') as f:
                # part_mark表示当前读到了文档的部分，
                # 0：url of the original article;，1：sentences in the article and their labels，
                # 2：extractable highlights，3：named entity mapping.
                part_mark = 0
                for line in f:
                    line = line.lower()  # 转换为小写
                    line = re.sub('[,\.\'!:"-]', "", line)#去掉标点符号
                    if line == '\n':
                        part_mark = part_mark + 1
                        #print "############################################"
                        continue
                    if part_mark == 0:
                        continue
                    # 遇到标志性空行
                    if part_mark == 1:
                        words_num=0
                        if data_sentence_index >= MAX_NUM_sentence:
                            continue
                        #print line
                        words = line.split()
                        label = int(words[-1])
                        data_label[data_sentence_index,label]=1
                        words.pop(-1)
                        for word in words:
                            if words_num >= MAX_LEN_sentence:
                                continue
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            data_feature[data_sentence_index,words_num] = vocab.word2id(word)
                            words_num += 1
                        data_sentence_index += 1
                    if part_mark == 2:
                        continue
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                    if part_mark == 3:
                        continue
            #print data_feature
            #print data_label
            #break
            print D, " : writing ", file_path
            #print data_feature
            data_feature_bytes = data_feature.tobytes()
            data_label_bytes = data_label.tobytes()
            features = genRecord(img_data=data_feature_bytes, label=data_label_bytes)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()

if __name__ == '__main__':
    main()