# utility functions of cnn

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets

import torchvision
import torch.utils.data as Data



def TF_load_mnist_dataset(num=-1):
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y = tf.one_hot(tf.cast(y, dtype=tf.int32), depth=10, dtype=tf.float32)
    y_test = tf.one_hot(tf.cast(y_test, dtype=tf.int32), depth=10, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    if num != -1:
        ds = ds.take(num).shuffle(num)
    else:
        ds = ds.shuffle(5000)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    if num != -1:
        test_ds = test_ds.take(num).shuffle(num)
    else:
        test_ds = test_ds.shuffle(1000)

    return ds, test_ds

    pass


def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y
    pass


def TORCH_load_mnist_dataset(num=-1):
    _DOWNLOAD_MNIST = False
    if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        _DOWNLOAD_MNIST = True
    else:
        _DOWNLOAD_MNIST = False

    train_data = torchvision.datasets.MNIST(root='./mnist/',
                                            train=True,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=_DOWNLOAD_MNIST)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)

    test_data = torchvision.datasets.MNIST(root='./mnist/',
                                           train=False,
                                           transform=torchvision.transforms.ToTensor())

    test_loader = Data.DataLoader(dataset=test_data, batch_size=30, shuffle=True)

    return train_loader, test_loader


class CNN_Base(object):
    def __init__(self, lr, maxiter, **kwargs):
        super(CNN_Base, self).__init__()
        self._learning_rate = lr
        self._maxIter = maxiter

        pass

    def Train(self, x, y):
        pass

    def __call__(self, x):
        pass

    def _calc_forward(self, x):
        pass

    def _calc_loss_and_accuracy(self, pred_y, labels):
        """
        softmax cross entropy loss and accuracy
        :param pred_y: shape(N, 10), numpy array, logits result
        :param labels: shape(N, 10), numpy one-hot array, labels
        :return:
        """
        losses = np.sum((- np.log(pred_y + 1e-12) * labels), axis=1)
        _loss = np.mean(losses)

        _y_1 = np.argmax(pred_y, axis=1)  # row based argmax
        _lables = np.argmax(labels, axis=1)

        _acc = np.mean(np.equal(_y_1, _lables).astype(np.float))
        return _loss, _acc

