# -*- coding: utf-8 -*-

from cnn_model import *
from read_data_into_datasets import *
import tensorflow as tf

MAX_NUM_sentence = 30
#学习率
learn_rate = tf.placeholder("float")

def main():
    f_path = r'/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/train.data'
    model1 = cnn_model.myModel()
    data_feature_old, data_label_old = read_and_decode(f_path)

    # 可以随机打乱输入
    data_feature, data_label = tf.train.shuffle_batch([data_feature_old, data_label_old],
                                                      batch_size=model1.batch_size, capacity=40,
                                                      min_after_dequeue=5)
    # print data_feature, data_label
    y_ = model1.gennet()
    y_new = tf.reshape(y_,[model1.batch_size,MAX_NUM_sentence,3])
    data_label_new = tf.reshape(data_label,[model1.batch_size,MAX_NUM_sentence,3])
    loss_mse = tf.reduce_mean(tf.square(y_new - data_label_new))

    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss_mse)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 线程需要同步
        coord = tf.train.Coordinator()
        # 启动队列
        threads = tf.train.start_queue_runners(coord=coord)
        # threads = tf.train.start_queue_runners(sess=sess)
        new_learn_rate = 1e-4
        rate = 10 ** (-4.0 / (model1.loop / 5000))

        for i in range(model1.loop):
            if (i + 1) % 2000 == 0:
                new_learn_rate = new_learn_rate * rate
                save_path = saver.save(sess, model1.model_save_path)
                # 输出保存路径
                print('Save to path: ', save_path)
            # with tf.device("/gpu:0"):
            # mse,_=sess.run([model,train_step],feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})
            
            if i % 10 == 0:
                input_x = data_feature.eval()
                mse, _ = sess.run([loss_mse, train_step], feed_dict={model1.x_placeholder: input_x,model1.keep_prob: 1.0,learn_rate:new_learn_rate})
                # print(sess.run(y_1))
                # print(sess.run(y_conv, feed_dict={keep_prob: 1}))
                print("step %d, training loss %g" % (i, mse))
                print("###########")
            else:
                mse, _ = sess.run([loss_mse, train_step], feed_dict={model1.x_placeholder: input_x,model1.keep_prob: 0.8,learn_rate:new_learn_rate})

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
