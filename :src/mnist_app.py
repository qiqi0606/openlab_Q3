# coding:utf-8

import numpy as np
import tensorflow as tf
from PIL import Image
import mnist_backward
import mnist_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:  # 重现计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])  # 仅需要给输入x占位
        y = mnist_forward.forward(x, None)  # 计算求得输出y
        preValue = tf.argmax(y, 1)  # y的最大值对应的索引列表号，就是预测结果

        # 实例化带有滑动平均值的saver
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)  # 恢复ckpt的信息到当前会话

                preValue = sess.run(preValue, feed_dict={x: testPicArr})  # 把图片喂入网络，执行预测操作
                return preValue
            else:
                print("No checkpoint found")
                return -1


def pre_pic(picName):
    img = Image.open(picName)  # 首先打开图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)  # 将图片改为28*28像素，Image.ANTIALIAS表示用消除锯齿的方法resize
    im_arr = np.array(reIm.convert('L'))  # 为了符合模型对颜色的要求，把reIm用convert('L')变成灰度图。用np.array()把reIm转化成矩阵的形式。赋给im_arr

    # 给图像进行二值化处理，让图片只有纯白色点和纯黑色点。不要有灰色点。这样可以滤掉手写图片中的噪声，留下图片主要特征。
    threshold = 50  # 阈值。可以自己调节阈值，让图片尽量包含手写数字的完整信息，也可以尝试其他方法来滤掉噪声。
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]  # 由于模型要求的是黑底白字，而我们的图片是白底黑字，所以要给图片反色。
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0  # 小于阈值的点认为是纯黑色0.
            else:
                im_arr[i][j] = 255  # 大于阈值的点认为是纯白色255.
    nm_arr = im_arr.reshape([1, 784])  # im_arr整理形状为1行784列。
    nm_arr = nm_arr.astype(np.float32)  # 由于模型要求像素点是0~1之间的浮点数，先让nm_arr变成浮点型。
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)  # 再让现有的rgb图从0到255之间的数，变为0~1之间的浮点数。

    return img_ready


def application():
    testNumtmp = input("input the number of test picture:")  # 读数字
    testNum = int(testNumtmp)
    for i in range(testNum):
        testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic)  # 预处理。由于要使用已有的模型作应用，所以图片必须符合输入要求.
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)


def main():
    application()


if __name__ == '__main__':
    main()
