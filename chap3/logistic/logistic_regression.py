#chapter 3. logistic regression exercise bi-classification
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
    d1 = np.array([x_p, y_p, y]).T

    x_n = np.random.normal(6., 1, dot_num)
    y_n = np.random.normal(3., 1, dot_num)
    y = np.zeros(dot_num) #全0向量
    d2 = np.array([x_n, y_n, y]).T

    #plt.scatter(d1[:, 0], d1[:, 1], c='b', marker='+')
    #plt.scatter(d2[:, 0], d2[:, 1], c='g', marker='o')

    data_set = np.concatenate((d1, d2), axis=0)
    np.random.shuffle(data_set)

    return data_set, d1, d2

class LogisticRegression:
    def __init__(self):
        # y = w'.x + b  w' 为w的转置wT
        self.W = tf.Variable(shape=[2, 1], dtype=tf.float32,
                            initial_value=tf.random.uniform(shape=[2, 1], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(shape=[1], dtype=tf.float32, initial_value=tf.zeros(shape=[1]))

        self.trainable_variables = [self.W, self.b]
        pass

    #@tf.function
    def __call__(self, input):
        if not isinstance(input, tf.Tensor):
            input_x = tf.constant(input, dtype=tf.float32)
        else:
            input_x = input
        logits_y = tf.matmul(input_x, self.W) + self.b # y = xw + b
        pred_y = tf.sigmoid(logits_y) #sigmod g(z) = 1/(1 + e^(-z)) ; z = y = xw + b
        return pred_y

def loss_func(pred, label):
    if not isinstance(label, tf.Tensor):
        label_c = tf.constant(label, dtype=tf.float32)
    else:
        label_c = label

    pred_c = tf.squeeze(pred, axis=1)

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

    losses = - (tf.multiply(label_c, tf.math.log(pred_c)) + \
                tf.multiply((1 - label_c), tf.math.log(1 - pred_c)))
    '''========================'''

    loss_r = tf.reduce_mean(losses)

    #置信度为0.5, 对于预测向量中，大于0.5的分类结果为1， 否则分类结果为0
    pred_d = tf.where(pred_c > 0.5, tf.ones_like(pred_c), tf.zeros_like(pred_c))
    accuracy_r = tf.reduce_mean(tf.cast(tf.equal(label_c, pred_d), dtype=tf.float32))
    return loss_r, accuracy_r

def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred_y = model(x)
        loss_t, accuracy_t = loss_func(pred=pred_y, label=y)

    grads = tape.gradient(loss_t, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_t, accuracy_t, model.W, model.b

def main():
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('Logistic Regression Example', fontsize=15)
    plt.ylabel('Y')
    plt.xlabel('X')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    line_d, = ax.plot([], [], label='fit_line')
    C1_dots, = ax.plot([], [], '+', c='b', label='actual_dots')
    C2_dots, = ax.plot([], [], 'o', c='g', label='actual_dots')

    frame_text = ax.text(0.02, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    # ax.legend()

    def init():
        line_d.set_data([], [])
        C1_dots.set_data([], [])
        C2_dots.set_data([], [])
        return (line_d,) + (C1_dots,) + (C2_dots,)

    def animate(i):
        xx = np.arange(10, step=0.1)
        a = animation_fram[i][0]
        b = animation_fram[i][1]
        c = animation_fram[i][2]
        """
        分界面 np.dot(w.T,x') + b = 0
        其中x' 为 [x, y]的二维向量，w为[wx, wy]的二维向量，展开为
        wx*x + wy*y + b = 0
        推到出y为：
        y = - (wx.x + b)/wy
        """
        #yy = - (a * xx + c) / b
        yy = a/-b * xx +c/-b
        line_d.set_data(xx, yy)

        C1_dots.set_data(c1[:, 0], c1[:, 1])
        C2_dots.set_data(c2[:, 0], c2[:, 1])

        frame_text.set_text('Timestep = %.1d/%.1d\nLoss = %.3f' % (i, len(animation_fram), animation_fram[i][3]))

        return (line_d,) + (C1_dots,) + (C2_dots,)

    anim = animation.FuncAnimation(f, animate, init_func=init,
                                   frames=len(animation_fram), interval=30, blit=True)


    #HTML(anim.to_html5_video())

    with open("output.html", "w") as f:
        print(anim.to_html5_video(), file=f)

    pass


if __name__ == '__main__':
    model = LogisticRegression()
    opt = tf.keras.optimizers.SGD(learning_rate=1e-2)
    data_set_train, c1, c2 = generate_data(100)
    x1, x2, y = list(zip(*data_set_train))
    x = list(zip(x1, x2))
    animation_fram = []

    for i in range(200):
        loss_i, accuracy_i, w_opt, b_opt = train_one_step(model, opt, x, y)
        animation_fram.append((w_opt.numpy()[0, 0], w_opt.numpy()[1, 0], b_opt.numpy(), loss_i.numpy()))
        if i % 20 == 0:
            print(f'loss: {loss_i.numpy():.4}\t accuracy: {accuracy_i.numpy():.4}')

    main()
    pass
