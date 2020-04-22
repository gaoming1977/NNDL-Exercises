#chapter 3. softmax regression exercise multi-classification

import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import numpy as np


#生成数据集
def generate_data(dot_num):
    #正态分布
    x_p = np.random.normal(3., 1, dot_num)
    y_p = np.random.normal(6., 1, dot_num)
    y = np.ones(dot_num) #全1向量
    c1 = np.array([x_p, y_p, y]).T

    x_n = np.random.normal(6., 1, dot_num)
    y_n = np.random.normal(3., 1, dot_num)
    y = np.zeros(dot_num) #全0向量
    c2 = np.array([x_n, y_n, y]).T

    x_b = np.random.normal(7., 1, dot_num)
    y_b = np.random.normal(7., 1, dot_num)
    y = np.ones(dot_num)*2 #全2向量
    c3 = np.array([x_b, y_b, y]).T

    data_set = np.concatenate((c1, c2, c3), axis=0)
    np.random.shuffle(data_set)

    return data_set, c1, c2, c3

class SoftmaxRegression:
    def __init__(self):
        #分3类，w1, w2, w3, b1, b2, b3
        self.W = tf.Variable(shape=[2, 3], dtype=tf.float32,
                             initial_value=tf.random.uniform(shape=[2, 3], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(shape=[1, 3], dtype=tf.float32, initial_value=tf.zeros(shape=[1, 3]))
        self.trainable_variables = [self.W, self.b]
        pass

    def __call__(self, input):
        if not isinstance(input, tf.Tensor):
            input_data = tf.constant(input, dtype=tf.float32)
        else:
            input_data = input
        logits_y = tf.matmul(input_data, self.W) + self.b #shape(N,3)
        pred = tf.nn.softmax(logits_y)
        #输出为shape(N,3), 每个元素[0,1]，且每行元素之和为1，表示分3类之和概率为1
        return pred
        pass

def loss_func(pred, label):
    # pred shape(N,3) label shape(N,3)
    #transform label(N,3) as one-hot tensor. as: [0,0,1], [0,1,0],[0,0,1]
    label = tf.one_hot(tf.cast(label, dtype=tf.int32), depth=3, dtype=tf.float32)

    #losses shape(N,3)
    losses = - (tf.multiply(label, tf.math.log(pred)) + tf.multiply((1 - label), tf.math.log(1 - pred)))
    loss = tf.reduce_mean(losses)

    label_c = tf.argmax(label, axis=1) #row based argmax
    pred_c = tf.argmax(pred, axis=1) #row based argmax

    accuracy = tf.reduce_mean(tf.cast(tf.equal(label_c, pred_c), dtype=tf.float32))

    return loss, accuracy
    pass

def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss, accuracy = loss_func(pred, y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy, model.W, model.b
    pass

def main():
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('Softmax Regression Example', fontsize=8)
    plt.ylabel('Y')
    plt.xlabel('X')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    #init fig
    C1_dots, = ax.plot([], [], '+', c='b', label='actual_dots')
    C2_dots, = ax.plot([], [], 'o', c='g', label='actual_dots')
    C3_dots, = ax.plot([], [], '*', c='r', label='actual_dots')

    frame_text = ax.text(0.02, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=8)

    #draw data
    C1_dots.set_data(c1[:, 0], c1[:, 1])
    C2_dots.set_data(c2[:, 0], c2[:, 1])
    C3_dots.set_data(c3[:, 0], c3[:, 1])

    #draw split lines
    xx = np.arange(10, step=0.01)
    yy = np.arange(10, step=0.01)

    x, y = np.meshgrid(xx, yy)
    input_data = np.array(list(zip(x.reshape(-1), y.reshape(-1))), dtype=np.float32)
    z = model(input_data)
    z = np.argmax(z, axis=1)
    z = z.reshape(x.shape)
    plt.contour(x, y, z)

    frame_text.set_text(f'loss(train): {loss.numpy():.4}\n '
                        f'accuracy(train): {accuracy.numpy():.4}\n ')

    plt.show()
    pass


if __name__ == '__main__':
    model = SoftmaxRegression()
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    data_set_train, c1, c2, c3 = generate_data(100)
    x1, x2, y = list(zip(*data_set_train))
    x = list(zip(x1, x2))

    #train process
    for i in range(1000):
        loss, accuracy, w_opt, b_opt = train_one_step(model, opt, x, y)
        if i % 50 == 0:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4} ')

    main()
    pass

