# chap4. ReLU FNN network regression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import fnn_util

def gen_dataset(num=1000):
    """
    generate data sets
    :return: (x1, x2, x3, x4, x5) , (y1, y2)
    z = (x1, x2, x3, x4, x5)
    f(z) = y = x * sin(x)
    x * cos(x)
    """
    #  the constant
    x_vec = np.random.normal(size=(num, 5)).astype(np.float64)
    y_vec = np.zeros((num, 2))

    def f_0(x, b=True):
        if b:
            ret = 0.5 * x
        else:
            ret = -0.5 * x
        return ret

    def f_1(x, b=True):
        if b:
            ret = x * np.sin(x)
        else:
            ret = x * np.cos(x)
        return ret

    def f_2(x, b=True):
        if b:
            ret = x * np.exp(x)
            pass
        else:
            ret = x * np.log(1 + x ** 2)
            pass
        return ret

    f = f_1
    for j in range(5):
        _x = x_vec[:, j]
        y_vec[:, 0] += f(_x, b=True)
        y_vec[:, 1] += f(_x, b=False)
    return x_vec, y_vec

g_lr = 9e-8

class FNN_Model_ReLU():

    def __init__(self, maxIter=10000,  precision=1e-2):
        self.maxIter = maxIter
        self.precs = precision

        #self.act_f = fnn_util.sigmoid
        #self.act_f_diff = fnn_util.diff_sigmoid
        self.loss_f = fnn_util.rmse_loss
        #self.loss_f = fnn_util.softmax_cross_entropy

        #  network topology
        self.i_nodes = 5
        self.h1_nodes = 20
        self.h2_nodes = 10
        self.o_nodes = 2


        # h1 layer
        self.W1 = np.random.rand(self.i_nodes, self.h1_nodes).astype(np.float64)
        self.b1 = np.random.rand(1, self.h1_nodes).astype(np.float64)

        # h2 hidden layer
        self.W2 = np.random.rand(self.h1_nodes, self.h2_nodes).astype(np.float64)
        self.b2 = np.random.rand(1, self.h2_nodes).astype(np.float64)

        # output layer
        self.Wo = np.random.rand(self.h2_nodes, self.o_nodes).astype(np.float64)
        self.bo = np.random.rand(1, self.o_nodes).astype(np.float64)

        pass

    def __calc_forward(self, x):
        assert(x.shape[1] == self.i_nodes)
        # input -> h1
        h1_logits = np.dot(x, self.W1) + self.b1
        h1_alpha = fnn_util.relu(h1_logits)  # ReLu(a1)

        # h1 -> h2
        h2_logits = np.dot(h1_alpha, self.W2) + self.b2
        h2_alpha = fnn_util.relu(h2_logits)

        # h2 -> output
        o_logits = np.dot(h2_alpha, self.Wo) + self.bo
        o_alpha = np.copy(o_logits)  # linear regression as activation fun. g(x) = x

        return o_alpha, h2_alpha, h1_alpha

    def calc_loss_and_accuracy(self, pred_y, labels):
        losses = self.loss_f(pred_y, labels)  # mse loss
        loss = np.mean(losses)
        accuracy = np.mean(np.where(losses < 1.0,
                                    1.0, 0.0))  # the dist between y and label
        return loss, accuracy
        pass

    def train(self, x_t, y_t):
        print("============= MODEL TRAINING =============")
        assert (x_t.shape[0] == y_t.shape[0])

        x_t = np.reshape(x_t, [-1, self.i_nodes])
        y_t = np.reshape(y_t, [-1, self.o_nodes])

        #  step2. BP process
        step = 0
        old_loss = 0.0
        old_o_alpha = np.zeros_like(y_t)
        while step < self.maxIter:
            step += 1
            #  step1. forward calculation and save every layer input and output
            o_alpha, h2_alpha, h1_alpha = self.__calc_forward(x_t)
            _loss, _acc = self.calc_loss_and_accuracy(o_alpha, y_t)

            if step % (self.maxIter / 10) == 0:
                print('epoch', step, ': loss', _loss, ': accuracy', _acc)

            if _acc > 0.9 and np.abs(_loss - old_loss) < self.precs:
                print('\t\t\tIteration Terminated!!! - Loss small enough\nepoch', step,
                      ': loss', _loss, ': accuracy', _acc)
                break
            if (old_o_alpha == o_alpha).all():
                print('\t\t\tIteration Terminated !!! - gradient vanished\nepoch', step,
                      ': loss', _loss, ': accuracy', _acc)
                break
            old_loss = _loss

            # step2. calc backward every layer difference delta
            #
            delta_o = o_alpha - y_t
            # h2
            delta_h2 = np.dot(delta_o, self.Wo.T) * fnn_util.diff_relu(h2_alpha)
            # h1
            delta_h1 = np.dot(delta_h2, self.W2.T) * fnn_util.diff_relu(h1_alpha)

            old_o_alpha = np.copy(o_alpha)

            # step3. update every layers W and b
            self.Wo = self.Wo - g_lr * \
                      np.dot(h2_alpha.T, delta_o)
            self.W2 = self.W2 - g_lr * \
                      np.dot(h1_alpha.T, delta_h2)
            self.W1 = self.W1 - g_lr * \
                      np.dot(x_t.T, delta_h1)

            self.bo = self.bo - g_lr * \
                      np.mean(delta_o, axis=0)
            self.b2 = self.b2 - g_lr * \
                      np.mean(delta_h2, axis=0)
            self.b1 = self.b1 - g_lr * \
                      np.mean(delta_h1, axis=0)

        # step. final calculate train dataset loss and accuracy
        o_alpha, *_t = self.__calc_forward(x_t)
        _loss, _acc = self.calc_loss_and_accuracy(o_alpha, y_t)

        print("============= MODEL TRAINING FINISHED =============")
        return _loss, _acc

    def __call__(self, x):
        assert (x.shape[1] == self.i_nodes)
        return self.__calc_forward(x)[0]
        pass


if __name__ == "__main__":
    x_train, y_train = gen_dataset(num=3000)
    x_test, y_test = gen_dataset(num=2000)

    myModel = FNN_Model_ReLU(maxIter=10000)
    _loss, _acc = myModel.train(x_train, y_train)
    print('\t\t\tTRAIN DATASET \n final loss ', _loss, 'accuracy ', _acc)

    print("================ MODEL TESTING ==============")
    pred_y = myModel(x_test)
    _loss, _acc = myModel.calc_loss_and_accuracy(pred_y, y_test)
    print('\t\t\tTEST DATASET \n final accuracy', _acc)
    print("\n================ MODEL TESTING END ===========")




