import tensorflow as tf

# 首先定义了神经网络结构的相关参数
INPUT_NODE = 784  # 神经网络的输入节点是784个。因为输入的是图片像素值，每张图片是28*28=784个像素点。
OUTPUT_NODE = 10  # 输出10个数，每个数表示对应的索引号出现的概率。实现了10分类。
LAYER1_NODE = 500  # 隐藏层节点的个数


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))  # 随机生成参数w
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))  # 如果使用正则化，则将每一个w的

    # 正则化损失加入到总损失集合losses
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


# 搭建网络，描述从输入到输出的数据流
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2  # 这个结果是直接输出的，因为要对输出使用softmax函数，使它符合概率分布，所以输出y,不过relu函数。
    return y
