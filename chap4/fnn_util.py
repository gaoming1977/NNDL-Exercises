#  chapter4. neural network exercise, some utility functions

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets

g_epsilon = 1e-12

def softmax(x):
    """
    softmax activate function
    softmax(x) = exp(x_i)/sum(exp(x_j)) j = 0,... ,m
    :param x: shape(N,m)
    :return: prob_x shape(N, m), each row sum(prob_x.row) == 1
    """
    sum_c = np.sum(np.exp(x), axis=1)
    sum_c = np.expand_dims(sum_c, axis=1)
    pred_x = np.divide(np.exp(x), sum_c)
    return pred_x


def sigmoid(x):
    """
    sigmoid activate function
    sigmoid(x) = 1 / (1 + exp(-x)) or sigmoid(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    :param x: shape(N, m)
    :return: pred_x shape(N, m)
    """
    #pred_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    pred_x = 1.0 / (1.0 + np.exp(-x))
    return pred_x
    pass


def diff_sigmoid(z):
    """
    the difference of sigmoid activate function
    diff_sigmoid(z) = z * (1-z)
    :param z: shape(N, m)
    :return: difference value of z
    """
    diff_z = np.multiply(z, (1.0 - z))
    return diff_z
    pass


def tanh(x):
    pred_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return pred_x
    pass


def diff_tanh(z):
    diff_z = 1 - z ** 2
    return diff_z


def relu(x):
    pred_x = np.maximum(0.0, x)
    return pred_x


def diff_relu(z):
    z_ = np.copy(z)
    diff_z = np.where(z_ > 0.0, 1.0, 0.0)
    return diff_z

def cos_vecs(x, y):
    """
    calc the cos value between x, y
    cos = x*y / |x|*|y|
    :param x:
    :param y:
    :return:
    """
    _t = np.sum((x * y), axis=1)
    norm_x = np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = np.linalg.norm(y, axis=1, keepdims=True)
    _t = np.reshape(_t, (-1, 1))
    ret = _t / (norm_x * norm_y + 1e-10)
    return ret


def rmse_loss(y, label):
    """
    row based mse loss function

    :param y:
    :param label:
    :return:
    """
    losses = np.sqrt(np.mean((y - label) ** 2, axis=1))
    return losses


def softmax_cross_entropy(y, label):
    """
    softmax cross entropy loss function
    cross_entropy(y, y~) = - sum(y * log(y~))
    usually apply for multi classification
    :param y: shape(N, m), each row y sum(y_row) = 1
    :param label: shape(N, c), label is row base one-hot.
    :return: losses: shape(N, 1) the row based cross entropy
    """
    losses = np.sum((- np.log(y + g_epsilon) * label), axis=1)
    return losses
    pass


def sigmoid_cross_entropy(y, label):
    """
    sigmoid cross entropy loss function
    cross_entropy(y, y~) = - y * log(y~) - (1-y) * log(1-y~)
    :param y: shape(N, m)
    :param label: shape(N, c), label[c] = 0,1
    :return: losses: shape(N, 1) row based cross entropy
    """
    losses = - np.log(y + g_epsilon) * label - np.log(1.0 - y + g_epsilon) * (1.0 - label)
    return losses

def vec_to_onehot(vec, m=0):
    """
    tansform vector to a one-hot
    :param m: the dimension of return one-hot matrix if m==0, using vec element category
    :param vec: vec shape(1, N) to transform
    :return:
    """

    n = vec.shape[0]
    if m == 0:
        m = len(np.unique(vec))

    one_hot = np.zeros((n, m))
    one_hot[range(n), vec] = 1
    return one_hot

    pass

def mnist_dataset(num=-1):
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # normalize
    x = x/255.0
    x_test = x_test/255.0

    if num != -1:
        x = x[0:num]
        y = y[0:num]
        x_test = x_test[0:num]
        y_test = y_test[0:num]

    return (x, y), (x_test, y_test)



if __name__ == "__main__":
    test_data = np.random.normal(size=[10, 5])

    pred_y = softmax(test_data)
    pred_y_tf = tf.nn.softmax(test_data, axis=-1)
    losses = (pred_y - pred_y_tf.numpy())**2

    label_data = np.zeros_like(test_data)
    label_data[range(10), np.random.randint(0, 5, size=10)] = 1

    loss_cross_entropy = softmax_cross_entropy(pred_y, label_data)
    loss_cross_entropy_tf = tf.nn.softmax_cross_entropy_with_logits(label_data, test_data)

    accuracy = tf.reduce_mean(losses)
    loss_accuracy = tf.reduce_mean((loss_cross_entropy - loss_cross_entropy_tf) ** 2)
    print(f"softmax function accuracy is :{accuracy.numpy() :4f}")
    print(f"softmax function cross entropy loss accuracy is :{loss_accuracy.numpy() :4f}")

    pred_y = sigmoid(test_data)
    pred_y_tf = tf.nn.sigmoid(test_data)
    losses = (pred_y - pred_y_tf.numpy())**2
    label_data[np.random.randint(0, 10, size=5), range(5)] = 1

    loss_cross_entropy = sigmoid_cross_entropy(pred_y, label_data)
    loss_cross_entropy_tf = tf.nn.sigmoid_cross_entropy_with_logits(label_data, test_data)

    accuracy = tf.reduce_mean(losses)
    loss_accuracy = tf.reduce_mean((loss_cross_entropy - loss_cross_entropy_tf) ** 2)
    print(f"sigmoid function accuracy is :{accuracy.numpy() :4f}")
    print(f"sigmoid function cross entropy loss accuracy is :{loss_accuracy.numpy() :4f}")

    pass

