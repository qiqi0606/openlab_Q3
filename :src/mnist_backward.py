import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200  # 每轮喂入神经网络的图片数
LEARNING_RATE_BASE = 0.1  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZER = 0.0001  # 正则化系数
STEPS = 50000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率,一般会赋接近 1 的值
MODEL_SAVE_PATH = './model/'  # 模型保存路径
MODEL_NAME = 'mnist_model'  # 模型保存名称


def backward(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])  # 利用placeholder占位，并设置正则化
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)  # 调用mnist_forward文件中的前向传播过程forward()函数，并设置正则化，计算训练集上预测结果y
    global_step = tf.Variable(0, trainable=False)  # 给当前计算轮数计数器赋值，设置为不可训练类型

    # 以下三步实现输出y经过softmax函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 设定指数衰减学习率learning_rate
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY, staircase=True)

    # 将反向传播的方法设置为梯度下降算法，对模型进行优化，降低损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 定义参数的滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 加入断点续训后
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)  # 将输入数据和标签数据输入神经网络
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('After', step, 'training step(s), loss on training batch is', loss_value)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)  # 将当前会话加载到指定路径


# 加载指定路径下的训练数据集
def main():
    mnist = input_data.read_data_sets('/Users/wangyunqi/PycharmProjects/pythonProject2_tensorflow1.1_python3.5/ /Users/wangyunqi/Desktop/兴趣开放/2021/手写数字识别-MNIST-train-images/ ', one_hot=True)
    backward(mnist)  # 调用规定的backward()函数训练模型


if __name__ == '__main__':
    main()
