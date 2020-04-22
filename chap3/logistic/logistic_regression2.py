#chapter 3. logistic regression self exercise bi-classification
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm


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

    data_set = np.concatenate((c1, c2), axis=0)
    np.random.shuffle(data_set)

    return data_set, c1, c2

class LogisticRegression:
    def __init__(self):
        # y = w'.x + b  w' 为w的转置wT
        self.W = tf.Variable(shape=[2, 1], dtype=tf.float32,
                            initial_value=tf.random.uniform(shape=[2, 1], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(shape=[1], dtype=tf.float32, initial_value=tf.zeros(shape=[1]))

        self.trainable_variables = [self.W, self.b]
        pass

    @tf.function
    def __call__(self, input_x):
        logits_y = tf.matmul(input_x, self.W) + self.b # y = xw + b
        pred = tf.nn.sigmoid(logits_y) #sigmod g(z) = 1/(1 + e^(-z)) ; z = y = xw + b
        #sigmoid函数生成独立的[0,1]
        return pred

@tf.function
def loss_func(pred, label):
    if not isinstance(label, tf.Tensor):
        label = tf.constant(label, dtype=tf.float32)
    pred = tf.squeeze(pred, axis=1)

    '''========================'''
    #sigmod的交叉熵损失函数
    #tensorflow sigmoid loss function
    #losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label)

    #自定义sigmoid损失函数
    """
        二分类，y为标签值， y = 0 or y = 1
        y = 1, p(y=1|x) = p
        y = 0, p(y=0|x) = 1-p
        L_lr = - ( y * log(p) + (1 - y) * log( 1- p))
        p 为预测值，经过sigmoid，取值[0,1] 
    """
    losses = - (tf.multiply(label, tf.math.log(pred)) + tf.multiply((1 - label), tf.math.log(1 - pred)))
    '''========================'''
    loss = tf.reduce_mean(losses)

    #置信度为0.5, 对于预测向量中，大于0.5的分类结果为1， 否则分类结果为0
    pred = tf.where(pred > 0.5, tf.ones_like(pred), tf.zeros_like(pred))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pred), dtype=tf.float32))
    return loss, accuracy

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss, accuracy = loss_func(pred, y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy, model.W, model.b

def main():
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('Logistic Regression Example2', fontsize=8)
    plt.ylabel('Y')
    plt.xlabel('X')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    #init fig
    line_d, = ax.plot([], [], label='fit_line')
    C1_dots, = ax.plot([], [], '+', c='b', label='actual_dots')
    C2_dots, = ax.plot([], [], 'o', c='g', label='actual_dots')
    frame_text = ax.text(0.02, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=8)

    C3_dots, = ax.plot([], [], '^', c='r', label='actual_dots')
    C4_dots, = ax.plot([], [], 'v', c='r', label='actual_dots')

    #draw train data
    C1_dots.set_data(c1[:, 0], c1[:, 1])
    C2_dots.set_data(c2[:, 0], c2[:, 1])

    C3_dots.set_data(c3[:, 0], c3[:, 1])
    C4_dots.set_data(c4[:, 0], c4[:, 1])
    #draw logistic regression line
    """
    分界面 np.dot(w.T,x') + b = 0
    其中x' 为 [x, y]的二维向量，w为[wx, wy]的二维向量，展开为
    wx*x + wy*y + b = 0
    推到出y为：
    y = - (wx*x + b)/wy
    """
    xx = np.arange(10, step=0.1)
    wx = w_opt.numpy()[0, 0]
    wy = w_opt.numpy()[1, 0]
    b = b_opt.numpy()[0]
    print(f'w(x): {wx :.4} \t w(y): {wy :.4} \t b: {b :.4}')
    yy = - wx/wy * xx - b / wy
    line_d.set_data(xx, yy)
    frame_text.set_text(f'loss(train): {loss.numpy():.4}\n '
                        f'accuracy(train): {accuracy.numpy():.4}\n '
                        f'loss(test): {loss_test.numpy():.4}\n'
                        f'accuracy(test): {accuracy_test.numpy():.4}')

    plt.show()
    pass


if __name__ == '__main__':
    model = LogisticRegression()
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    data_set_train, c1, c2 = generate_data(100)
    x1, x2, y = list(zip(*data_set_train))
    x = list(zip(x1, x2))

    #train process
    for i in range(200):
        loss, accuracy, w_opt, b_opt = train_one_step(model, opt, x, y)
        if i % 20 == 0:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4} ')

    #test data
    data_set_test, c3, c4 = generate_data(10)
    test_x1, test_x2, test_y = list(zip(*data_set_test))
    test_x = list(zip(test_x1, test_x2))

    test_pred = model(test_x)
    loss_test, accuracy_test = loss_func(pred=test_pred, label=test_y)

    main()
    pass
