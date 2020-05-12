#  chapter4. neural network exercise, simple FNN by Numpy
# using mnist dataset

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import fnn_util

g_lr = 0.001


class FNN_Model_NP:
    """
    FNN Model implement by Numpy
    train and test by MNIST dataset
    """
    def __init__(self, maxIter=1000,  precision=1e-7):
        self.maxIter = maxIter
        self.precs = precision

        #  network topology
        self.input_nodes = 28*28
        self.h1_nodes = 100
        self.h2_nodes = 50
        self.output_nodes = 10

        self.act_f = fnn_util.sigmoid
        self.act_f_diff = fnn_util.diff_sigmoid
        # self.loss_f = fnn_util.mse_loss
        self.loss_f = fnn_util.softmax_cross_entropy
        # self.act_f = fnn_util.relu
        # self.act_f_diff = fnn_util.diff_relu

        # initial parameter
        """
        h1 hidden layer
        w1, b1: input->h1, w1.shape(in,h1), b1(1, h1)
        w1 shape(748, 100), b1 shape(100, )
        """
        self.W1 = np.random.normal(size=[self.input_nodes, self.h1_nodes])
        self.b1 = np.random.normal(size=[1, self.h1_nodes])

        """
        h2 hidden layer
        w2, b2: h1->h2, w1.shape(h1, h2), b1(1, h2)
        w2 shape(100, 50), b2 shape(50, )
        """
        self.W2 = np.random.normal(size=[self.h1_nodes, self.h2_nodes])
        self.b2 = np.random.normal(size=[1, self.h2_nodes])

        """
        output layer
        w3, b3: h2->h3, w1.shape(h2, out), b1(1, out)
        w3 shape(50, 10), b3 shape(10, )
        """
        self.Wo = np.random.normal(size=[self.h2_nodes, self.output_nodes])
        self.bo = np.random.normal(size=[1, self.output_nodes])

        pass

    def __calc_forward(self, x):
        assert(x.shape[1] == self.input_nodes)
        # input -> h1
        h1_logits = np.dot(x, self.W1) + self.b1
        h1_alpha = self.act_f(h1_logits)  # fnn_util.sigmoid(h1_logits)

        # h1 -> h2
        h2_logits = np.dot(h1_alpha, self.W2) + self.b2
        h2_alpha = self.act_f(h2_logits)  # fnn_util.sigmoid(h2_logits)

        # h2 -> output
        out_logits = np.dot(h2_alpha, self.Wo) + self.bo
        out_alpha = self.act_f(out_logits)  # fnn_util.sigmoid(out_logits)

        return out_alpha, h1_alpha, h2_alpha

    def __calc_loss_and_accuracy(self, logits, labels):

        pred_y = fnn_util.softmax(logits)
        losses = self.loss_f(pred_y, labels)  # fnn_util.softmax_cross_entropy(pred_y, labels)
        loss = np.mean(losses)

        y_1 = np.argmax(labels, axis=1)
        pred_y = np.argmax(pred_y, axis=1)

        accuracy = np.mean(np.equal(pred_y, y_1).astype(np.float32))

        return loss, accuracy
        pass

    def train(self, x_train, y_train):
        print("============= MODEL TRAINING =============")

        #  step1. preprocess input training data
        assert(x_train.shape[0] == y_train.shape[0])
        sample_N = x_train.shape[0]
        assert(x_train.shape[1] == x_train.shape[2] == 28)

        #  reshape the train x data from (N, 28, 28) -> (N, 28*28)
        #  reshape the train y data to ONE-HOT
        x_train = np.reshape(x_train, [-1, self.input_nodes])
        y_train_oh = fnn_util.vec_to_onehot(y_train, m=self.output_nodes)

        #  step2. BP process
        step = 0
        old_acc = 0.0
        while step < self.maxIter:
            step += 1
            #  step1. forward calculation and save every layer input and output
            out_alpha, h1_alpha, h2_alpha = \
                self.__calc_forward(x_train)

            # check iteration terminal condition
            loss, acc = self.__calc_loss_and_accuracy(out_alpha, y_train_oh)

            if step % (self.maxIter / 10) == 0:
                print('epoch', step, ': loss', loss, '; accuracy', acc)
            if acc > 0.9 and np.abs(acc - old_acc) < self.precs:
                print('\t\t\tIteration Terminated!!!\nepoch', step, ': loss', loss,
                      '; accuracy', acc)
                break
            old_acc = acc

            # step2. calc backward every layer difference delta
            #
            delta_out = -(y_train_oh - out_alpha) * self.act_f_diff(out_alpha)
            # h2
            delta_h2 = np.dot(delta_out, self.Wo.T) * self.act_f_diff(h2_alpha)
            # h1
            delta_h1 = np.dot(delta_h2, self.W2.T) * self.act_f_diff(h1_alpha)

            # step3. update every layers W and b
            self.Wo = self.Wo - g_lr * \
                      np.dot(h2_alpha.T, delta_out)
            self.W2 = self.W2 - g_lr * \
                      np.dot(h1_alpha.T, delta_h2)
            self.W1 = self.W1 - g_lr * \
                      np.dot(x_train.T, delta_h1)

            self.bo = self.bo - g_lr * np.mean(delta_out, axis=0)
            self.b2 = self.b2 - g_lr * np.mean(delta_h2, axis=0)
            self.b1 = self.b1 - g_lr * np.mean(delta_h1, axis=0)

            pass

        # step. final calculate train dataset loss and accuracy
        out_alpha, _, _ = self.__calc_forward(x_train)
        loss, accuracy = self.__calc_loss_and_accuracy(out_alpha, y_train_oh)

        print("============= MODEL TRAINING FINISHED =============")
        return loss, accuracy


    def __call__(self, x):
        assert (x.shape[1] == 28*28)
        logits, *_t = self.__calc_forward(x)

        pred_y = fnn_util.softmax(logits)
        pred_y = np.argmax(pred_y, axis=1)
        return pred_y


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = fnn_util.mnist_dataset(num=5000)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    myModel = FNN_Model_NP()
    _loss, _acc = myModel.train(x_train, y_train)
    print('\t\t\tTRAIN DATASET \n final loss', _loss, '; accuracy', _acc)

    print("================ MODEL TESTING ==============")
    x_test_1 = np.copy(x_test)
    x_test_1 = np.reshape(x_test_1, [-1, 28*28])
    pred_y = myModel(x_test_1)
    _acc = np.mean(np.equal(pred_y, y_test).astype(np.float32))
    print('\t\t\tTEST DATASET \n final accuracy', _acc)
    print("\n================ MODEL TESTING END ===========")

    import matplotlib.pyplot as plt

    Js = np.random.randint(0, len(y_test), size=30)
    for i in range(30):
        ax = plt.subplot(5, 6, i+1)
        ax.axis('off')
        img = x_test[Js[i]]
        ax.imshow(img, cmap=plt.get_cmap('gray'))

        img = np.reshape(img, [-1, 28*28])
        pred = myModel(img)
        label = y_test[Js[i]]
        if pred[0] == label:
            ax.set_title(f"{pred[0] :d}", color='green', fontsize=15)
        else:
            ax.set_title(f"{pred[0] :d}", color='red', fontsize=15)

    # show the plot
    plt.show()


    pass


