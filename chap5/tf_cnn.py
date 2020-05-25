# tf_cnn, the cnn of tensorflow version
# reference: https://www.jianshu.com/p/33c81b3a5d65 https://www.cnblogs.com/yangmang/p/7528935.html

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
import cnn_util


g_run_version = "CPU"
#g_run_version = "GPU"

class CNN_tf(cnn_util.CNN_Base):
    def __init__(self, opt=optimizers.Adam(learning_rate=1e-3)):
        super().__init__(lr=1e-3, maxiter=2000)
        self._opt = opt

        # step1. declare input_x, input_y as placeholder
        """
        input_x shape(M, 28, 28, 1),
        input_y shape(M, 10)
        """
        pass

        # topolgy
        """
        layer conv-1
        input shape: (M, 28, 28, 1)
        output shape: (M, 28, 28, 32)
        conv-1 2D kernel 7*7, channel 32, so, the conv-1 kernel shape (7, 7, 32) 
        padding: same
        stride: 1
        activate function: relu
        """
        _c_num = 32 if g_run_version == "GPU" else 1
        self.W_conv1 = tf.Variable(shape=[7, 7, 1, _c_num],
                                   initial_value=
                                   tf.random.uniform(shape=[7, 7, 1, _c_num],
                                                     dtype=tf.float32,
                                                     minval=-0.1, maxval=0.1))
        self.b_conv1 = tf.Variable(shape=[1],
                                   initial_value=
                                   tf.constant(0.1, shape=[1]))

        """
        layer max pooling-1
        input shape:(M, 28, 28, 32)
        output shape:(M, 14, 14, 32)
        kernel as 2*2
        padding: same
        stride: 2
        """
        pass  # no parameter here

        """
        layer conv-2
        input shape: (M, 14, 14, 32)
        output shape: (M, 14, 14, 64)
        conv-2 2D kernel 5*5, channel 64, so, the conv-2 kernel shape (5, 5, 64) 
        padding: same
        stride: 1
        activate function: relu
        """
        _c_num = 64 if g_run_version == "GPU" else 1
        self.W_conv2 = tf.Variable(shape=[5, 5, 1, _c_num],
                                   initial_value=
                                   tf.random.uniform(shape=[5, 5, 1, _c_num],
                                                     dtype=tf.float32,
                                                     minval=-0.1, maxval=0.1))
        self.b_conv2 = tf.Variable(shape=[1],
                                   initial_value=
                                   tf.constant(0.1, shape=[1]))

        """
        layer max pooling-2
        input shape:(M, 14, 14, 64)
        output shape:(M, 7, 7, 64)
        kernel as 2*2
        padding: same
        stride: 2
        """
        pass  # no parameter here

        """
        layer: full connection fc-1
        input shape: (M, 7, 7, 64)  ---(flatten)---> (M, 7*7*64)
        output shape: (M, 1024)
        activate function: relu
        """
        _c_num = 64 if g_run_version == "GPU" else 1
        self.W_fc1 = tf.Variable(shape=[7*7*_c_num, 1024],
                                 initial_value=
                                 tf.random.uniform(shape=[7*7*_c_num, 1024],
                                                   dtype=tf.float32,
                                                   minval=-0.1, maxval=0.1))
        self.b_fc1 = tf.Variable(shape=[1024],
                                 initial_value=
                                 tf.constant(0.1, shape=[1024]))

        """
        layer: full connection fc-2
        input shape:(M, 1024)
        output shape:(M, 10)
        activate function: softmax
        """
        self.W_fc2 = tf.Variable(shape=[1024, 10],
                                 initial_value=
                                 tf.random.uniform(shape=[1024, 10],
                                                   dtype=tf.float32,
                                                   minval=-0.1, maxval=0.1))
        self.b_fc2 = tf.Variable(shape=[10],
                                 initial_value=
                                 tf.constant(0.1, shape=[10]))

        pass

    def __call__(self, x):
        # if x is only one row, it should reshape to [-1, 28, 28, 1]
        vdim = len(x.get_shape())
        if vdim != 4:
            x = tf.reshape(x, [-1, 28, 28, 1])

        y_h = self._calc_forward(x)
        y_h = tf.argmax(y_h, axis=1)
        return y_h
        pass

    def _calc_forward(self, x):
        """
        calc predict of CNN
        :param x: input x shape(M, 28, 28, 1)
        :return: softmax of CNN, shape{M, 10)
        """
        # input -> conv#1
        h_conv_1 = self.__conv2d(x, self.W_conv1) + self.b_conv1  # y = W ⊗ X + b
        h_conv_1 = tf.nn.relu(h_conv_1)

        # conv#1 -> max pooling#1
        h_max_pooling_1 = self.__max_pooling_2x2(h_conv_1)

        # max pooling#1 -> conv#2
        h_conv_2 = self.__conv2d(h_max_pooling_1, self.W_conv2) + self.b_conv2
        h_conv_2 = tf.nn.relu(h_conv_2)

        # conv#2 -> max pooling#2
        h_max_pooling_2 = self.__max_pooling_2x2(h_conv_2)

        # max pooling#2 -> FC#1
        """
        here should flatten reshape the max pooling result to fit FC layer
        """
        # important !!! reshape the output of max pooling
        h_max_pooling_2 = tf.reshape(h_max_pooling_2, shape=[-1, self.W_fc1.shape[0]])
        h_fc_1 = tf.matmul(h_max_pooling_2, self.W_fc1) + self.b_fc1
        h_fc_1 = tf.nn.relu(h_fc_1)

        # FC#1 -> FC#2
        h_out = tf.matmul(h_fc_1, self.W_fc2) + self.b_fc2
        h_out = tf.nn.softmax(h_out, axis=1)

        return h_out

    def __conv2d(self, x, w):
        ret = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        return ret
        pass

    def __max_pooling_2x2(self, x):
        ret = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return ret
        pass

    def _train_one_step(self, x, y):
        with tf.GradientTape() as tape:
            pred_y = self._calc_forward(x)
            loss, acc = self._calc_loss_and_accuracy(pred_y, y)

        train_vars = [self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2,
                      self.b_conv1, self.b_conv2, self.b_fc1, self.b_fc2]

        grads = tape.gradient(loss,train_vars)
        self._opt.apply_gradients(grads_and_vars=zip(grads, train_vars))
        return loss, acc

    def Train(self, ds):
        print("\t\t\t============= MODEL TRAINING BEGIN =============")
        epoch = 0
        _old_acc = 0.0
        ds_batch = ds.batch(batch_size=100).repeat()
        _iter = ds_batch.__iter__()
        while epoch < self._maxIter:
            epoch += 1
            batch_xs, batch_ys = _iter.next() # 100 training sample
            _loss, _acc = self._train_one_step(batch_xs, batch_ys)

            if epoch % (self._maxIter / 10) == 0:
                print('epoch', epoch, ': loss', _loss.numpy(), '; accuracy', _acc.numpy())

            # accuracy increase little and accuracy > 0.96
            if _acc > 0.96:
                print('\t\t\tIteration Terminated!!!\nepoch', epoch, ': loss', _loss.numpy(),
                      '; accuracy', _acc.numpy())
                break
        print("Final training accuracy: ", _acc.numpy())
        print("\t\t\t============= MODEL TRAINING FINISHED =============")

        return _acc
        pass

    def _calc_loss_and_accuracy(self, pred_y, labels):
        # convert tf tensor to numpy
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=labels)
        loss = tf.reduce_mean(losses)

        # from one-hot to 1
        y_1 = tf.argmax(pred_y, axis=1)
        labels_1 = tf.argmax(labels, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_1, y_1), dtype=tf.float32))

        return loss, accuracy


if __name__ == '__main__':
    ds, ds_test = cnn_util.TF_load_mnist_dataset()
    myModel = CNN_tf()
    _acc = myModel.Train(ds)

    # test model
    _acc_test = np.array([])
    for x_test, y_test in ds_test:
        x_test = tf.reshape(x_test, [-1, 28, 28, 1])
        y_test = tf.reshape(y_test, [-1, 10])
        y_pred = myModel(x_test)
        y_test = tf.argmax(y_test, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, y_pred), dtype=tf.float32))
        _acc_test = np.append(_acc_test, accuracy.numpy())

    _acc = np.mean(_acc_test)
    print("\nFinal test dataset accuracy: ", _acc)

    import matplotlib.pyplot as plt

    batch_ds = ds_test.shuffle(buffer_size=1000).batch(30)
    _iter = batch_ds.__iter__()
    img, label = _iter.next()
    img = tf.reshape(img, [-1, 28, 28, 1])
    pred = myModel(img)
    label = tf.argmax(label, axis=1)
    for i in range(30):
        ax = plt.subplot(5, 6, i+1)
        ax.axis('off')
        ax.imshow(img.numpy()[i, :, :, 0], cmap=plt.get_cmap('gray'))

        if tf.equal(label[i], pred[i]):
            ax.set_title(f"{pred.numpy()[i] :d}", color='green', fontsize=15)
        else:
            ax.set_title(f"{pred.numpy()[i] :d}", color='red', fontsize=15)

    # show the plot
    plt.show()





