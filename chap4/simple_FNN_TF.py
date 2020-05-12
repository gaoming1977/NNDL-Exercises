#  chapter4. neural network exercise, simple FNN by TF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


import numpy as np
import fnn_util

g_learning_rate_param = 0.01

class FNN_Model_TF():
    """
    simple FNN model implement by tensorflow
    4 layers, 1- input layer; 2, 3- hidden layer; 4- output layers
    input layer has 28*28 nodes
    hidden layer1 has 100 nodes
    hidden layer2 has 60 nodes
    output layer has 10 nodes, as 10 category from 0 to 9
    optimizer: Adam is prefer optimizer, reference : https://keras.io/optimizers/

    input data from MNIST dataset
    it has (60000, 28, 28) train data and (10000, 28, 28)
    each data is 28 * 28 image
    """
    def __init__(self, maxIter=500, precision=1e-7, opt=None):
        self.input_nodes = 28 * 28
        self.output_nodes = 10
        self.h1_nodes = 100
        self.h2_nodes = 60
        self.maxIter = maxIter
        self.precs = precision

        # step2. initialize W, b parameters
        """
        w1, b1: input->h1, w1.shape(in,h1), b1(1, h1)
        """
        self.W1 = tf.Variable(shape=[self.input_nodes, self.h1_nodes], dtype=tf.float32,
                              initial_value=tf.random.uniform(shape=[self.input_nodes, self.h1_nodes],
                                                              minval=-0.1, maxval=0.1))
        self.b1 = tf.Variable(shape=[1, self.h1_nodes], dtype=tf.float32,
                              initial_value=tf.constant(0.1, shape=[1, self.h1_nodes]))

        """
        w2, b2: h1->h2, w1.shape(h1, h2), b1(1, h2)
        """
        self.W2 = tf.Variable(shape=[self.h1_nodes, self.h2_nodes],  dtype=tf.float32,
                              initial_value=tf.random.uniform(shape=[self.h1_nodes, self.h2_nodes],
                                                              minval=-0.1, maxval=0.1))
        self.b2 = tf.Variable(shape=[1, self.h2_nodes], dtype=tf.float32,
                              initial_value=tf.constant(0.1, shape=[1, self.h2_nodes]))

        """
        w3, b3: h2->h3, w1.shape(h2, out), b1(1, out)
        """
        self.W3 = tf.Variable(shape=[self.h2_nodes, self.output_nodes], dtype=tf.float32,
                              initial_value=tf.random.uniform(shape=[self.h2_nodes, self.output_nodes],
                                                              minval=-0.1, maxval=0.1))
        self.b3 = tf.Variable(shape=[1, self.output_nodes], dtype=tf.float32,
                              initial_value=tf.constant(0.1, shape=[1, self.output_nodes]))

        if opt is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=g_learning_rate_param)
        else:
            self.optimizer = opt
        pass

    def __call__(self, x):
        """
        predict the result by trained model
        :param x: shape[N, 28*28], one row is one image.
        :return: shape[N, 1], the final result of predict and classification
        """
        assert (x.shape[1] == 28*28)
        logits, _, _ = self.__calc_forward(x)

        pred_y = tf.nn.softmax(logits)
        pred_y = tf.argmax(pred_y, axis=1)

        return pred_y.numpy()

    def __train_one_step(self, x, label):
        """
        one iteration step in training loop
        :param x: shape(N, 28*28)
        :param label: shape(N, 10), the ONE-HOT transform original label
        :return: this step loss and accuracy
        """
        with tf.GradientTape() as tape:
            logits_y = self.__calc_forward(x)[0]
            loss, accuracy = self.__calc_loss_accuracy(logits_y, label)

        trainable_vars = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
        grads = tape.gradient(loss, trainable_vars)
        '''
        for g, v in zip(grads, trainable_vars):
            v.assign_sub(0.5*g)
        '''
        self.optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_vars))
        return loss, accuracy

    def __calc_forward(self, x):
        """
        FNN forward calculation
        hj_logits = wj * h(j-1)_alpha + bj
        hj_alpha = sigmoid(hj_logits)
        :param x: shape(N, 28*28)
        :return: network forward every layer activate value (alpha):
            shape(N,10), shape(N, 100), shape(N, 60)
        """
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype=tf.float32)

        x = tf.cast(x, dtype=tf.float32)

        # input -> h1
        h1_logits = tf.matmul(x, self.W1) + self.b1
        h1_alpha = tf.nn.relu(h1_logits)

        # h1 -> h2
        h2_logits = tf.matmul(h1_alpha, self.W2) + self.b2
        h2_alpha = tf.nn.relu(h2_logits)

        # h2 -> output
        h3_logits = tf.matmul(h2_alpha, self.W3) + self.b3
        out_alpha = tf.nn.relu(h3_logits)

        return out_alpha, h1_alpha, h2_alpha

    @tf.function
    def __calc_loss_accuracy(self, logits, y):
        """
        calculate loss and accuracy
        :param logits: shape(N, 10), the result of __cal_forward
        :param y:  the ONE-HOT label shape(N, 10)
        :return: the loss and accuracy
        """
        assert(y.shape[1] == self.output_nodes)

        y_1 = tf.argmax(y, axis=1)
        #losses = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_1, logits=logits)  # y is one-hot vectors
        loss = tf.reduce_mean(losses)

        pred_y = tf.nn.softmax(logits)
        pred_y = tf.argmax(pred_y, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_y, y_1), dtype=tf.float32))

        return loss, accuracy
        pass

    def train(self, x_train, y_train):
        """
        model train function
        :param x_train:  shape(N, 28, 28)
        :param y_train:  shape(N, )
        :return:
        """
        print("============= MODEL TRAINING =============")
        #  step1. preprocess input training data
        assert(x_train.shape[0] == y_train.shape[0])
        #sample_N = x_train.shape[0]
        assert(x_train.shape[1] == x_train.shape[2] == 28)

        #  reshape the train x data from (N, 28, 28) -> (N, 28*28)
        #  reshape the train y data to one-hot
        x_train_trans = np.reshape(x_train, [-1, self.input_nodes])
        #label_train = fnn_util.vec_to_onehot(y_train, m=self.output_nodes)
        label_train = tf.one_hot(tf.cast(y_train, dtype=tf.int32), depth=self.output_nodes, dtype=tf.float32)

        # step2. train loop
        accuracy = tf.constant(0.0, dtype=tf.float32)
        for epoch in range(self.maxIter):  # batch 100 steps
            accuracy_old = accuracy
            loss, accuracy = self.__train_one_step(x_train_trans, label_train)
            if epoch % (self.maxIter / 10) == 0:
                print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

            # accuracy increase little and accuracy > 0.9
            if tf.abs(accuracy - accuracy_old) < self.precs and tf.abs(accuracy) > 0.9:
                print('\t\t\tIteration Terminated!!!\nepoch', epoch, ': loss', loss.numpy(),
                      '; accuracy', accuracy.numpy())
                break
        # step final test
        f_alpha = self.__calc_forward(x_train_trans)[0]
        loss, accuracy = self.__calc_loss_accuracy(f_alpha, label_train)

        print("============= MODEL TRAINING FINISHED =============\n")
        return loss, accuracy

    pass


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = fnn_util.mnist_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    """
    SGD optimizer
    """
    #myModel = FNN_Model_TF(opt=tf.keras.optimizers.SGD(lr=g_learning_rate_param))
    """
    default Adam optimizer
    """
    myModel = FNN_Model_TF()
    loss, accuracy = myModel.train(x_train, y_train)
    print('\t\t\tTRAIN DATASET \n final loss', loss.numpy(), '; accuracy', accuracy.numpy())

    print("================ MODEL TESTING ==============")
    x_test_1 = np.copy(x_test)
    x_test_1 = np.reshape(x_test_1, [-1, 28*28])
    pred_y = myModel(x_test_1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_y, y_test), dtype=tf.float32))
    print('\t\t\tTEST DATASET \n final accuracy', accuracy.numpy())

    print("\n================ MODEL TESTING END ===========")

    import matplotlib.pyplot as plt

    Js = np.random.randint(0, len(y_test), size=30)
    for i in range(30):
        ax = plt.subplot(5, 6, i+1)
        ax.axis('off')
        img = x_test[Js[i]]
        ax.imshow(img, cmap=plt.get_cmap('gray'))

        img = np.reshape(img, [-1, 28*28])
        label = y_test[Js[i]]
        pred = myModel(img)
        if pred == label:
            ax.set_title(f"{pred[0] :d}", color='green', fontsize=15)
        else:
            ax.set_title(f"{pred[0] :d}", color='red', fontsize=15)

    # show the plot
    plt.show()

    pass

