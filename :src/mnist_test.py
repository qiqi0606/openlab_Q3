# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

TEST_INTERVAL_SECS = 5  # 规定程序5秒的循环间隔时间


def test(mnist):
    with tf.Graph().as_default() as g:  # 利用tf.Graph()复现之前定义的计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])  # 占位
        y = mnist_forward.forward(x, None)  # 计算训练数据集上的预测结果y

        # 实例化具有滑动平均的saver对象，从而在会话被加载时模型中的所有参数被赋值为各自的滑动平均值，增强模型的稳定性
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 计算模型在测试集上的准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # with结构中，加载指定路径下的ckpt,若模型存在，则加载出模型到当前对话，在测试集上进行准确率验证，并打印出当前轮数下的准确率
        # 如果模型不存在，则打印出模型不存在的提示
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print('After', global_step, 'training step(s), test accuracy = ', accuracy_score)
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets('/Users/wangyunqi/PycharmProjects/pythonProject2_tensorflow1.1_python3.5/ /Users/wangyunqi/Desktop/兴趣开放/2021/手写数字识别-MNIST-train-images/ ', one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
