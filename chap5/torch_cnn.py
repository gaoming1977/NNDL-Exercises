# tf_cnn, the cnn of pytorch version

import torch
import torch.nn as nn
import cnn_util
from torch.autograd import Variable
import time

#g_run_version = "CPU"
g_run_version = "GPU"

class CNN_torch(cnn_util.CNN_Base, nn.Module):
    def __init__(self, lr=1e-3, loss_f=nn.CrossEntropyLoss(),  maxiter=5):
        nn.Module.__init__(self)
        cnn_util.CNN_Base.__init__(self, lr=lr, maxiter=maxiter)

        """
        layer: convolution #1 + max pooling#1
        input: shape (M, 28, 28, 1)
        output: shape (M, 14, 14, 32)
        kernel: shape (7, 7, 32), 32 channels, 
        padding: same, padding = 3,
        stride: 1
        activate function: relu
        max pooling: 2 * 2
        """
        _c_out_num = 32 if g_run_version == "GPU" else 1
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=_c_out_num,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        """
        layer: convolution #2 + max pooling#2
        input: shape (M, 14, 14, 32)
        output: shape (M, 7, 7, 64)
        kernel: shape (5, 5, 64), 64 channels, 
        padding: same, padding = 2,
        stride: 1
        activate function: relu
        max pooling: 2 * 2
        """
        _c_in_num = 32 if g_run_version == "GPU" else 1
        _c_out_num = 64 if g_run_version == "GPU" else 1
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=_c_in_num,
                out_channels=_c_out_num,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )
        """
        layer: full connection FC_#1
        input: shape (M, 7, 7, 64) ---(flatten)---> (M, 7*7*64)
        output: shape (M, 1024)
        activate function: relu
        """
        _c_num = 64 if g_run_version == "GPU" else 1
        self.fc_1 = nn.Sequential(
            nn.Linear(
                in_features=7*7*_c_num,
                out_features=1024,
                bias=True
            ),
            nn.ReLU()
        )

        """
        layer: full connection 2 out
        input: shape (M, 1024)
        output: shape (M, 10)
        activate function: softmax
        """
        self.fc_2 = nn.Sequential(
            nn.Linear(
                in_features=1024,
                out_features=10,
                bias=True
            ),
            nn.Softmax(dim=1)
        )

        self._opt = torch.optim.Adam(params=self.parameters(), lr=self._learning_rate)
        self._loss_f = loss_f

        pass

    def _calc_forward(self, x):
        h_conv_1 = self.conv_1(x)
        h_conv_2 = self.conv_2(h_conv_1)

        # reshape the h_conv_2
        _c_num = 64 if g_run_version == "GPU" else 1
        h_conv_2 = torch.reshape(h_conv_2, [-1, 7*7*_c_num])

        h_fc_1 = self.fc_1(h_conv_2)
        h_out = self.fc_2(h_fc_1)
        return h_out
        pass

    def __call__(self, x):
        h_out = self._calc_forward(x)
        _out = torch.argmax(h_out, dim=1, keepdim=False)  # row based argmax
        return _out

    def __train_one_step(self, x, y):
        pred_y = self._calc_forward(x)  # shape(M, 10)

        losses = self._loss_f(pred_y, y)
        _loss = losses.mean(dtype=torch.float32)

        pred_y_1 = torch.argmax(pred_y, dim=1, keepdim=False)  # row based argmax
        _acc = torch.where(torch.eq(pred_y_1, y),
                           torch.full_like(y, 1.0, dtype=torch.float32),
                           torch.full_like(y, 0.0, dtype=torch.float32))
        _acc = torch.mean(_acc)

        self._opt.zero_grad()
        losses.backward()
        self._opt.step()
        return _loss.detach().numpy(), _acc.detach().numpy()
        pass

    def Train(self, ds):
        _begin_t = time.time()
        print("\t\t\t============= MODEL TRAINING BEGIN =============")
        epoch = 0
        _old_acc = 0.0
        print("maxIter: ", self._maxIter, " learning rate: ", self._learning_rate)
        while epoch < self._maxIter:
            _acc_step = 0
            for batch, (x_, y_) in enumerate(ds):
                x, y = Variable(x_), Variable(y_)
                _loss, _acc = self.__train_one_step(x, y)

                _acc_step += _acc
                if batch % 100 == 0:
                    print('epoch', epoch, ', batch', batch, ': loss', _loss, '; accuracy', _acc)

            _acc_mean = _acc_step / len(ds)
            print("")
            print('epoch ', epoch, ' accuracy', _acc_mean)
            print("")

            # accuracy increase little and accuracy > 0.96
            if _acc_mean > 0.96:
                print('\t\t\tIteration Terminated!!!\nepoch', epoch, ': loss', _loss,
                      '; accuracy', _acc_mean)
                break
            epoch += 1

        print("Final training accuracy: ", _acc, " Time elapsed: ", time.time() - _begin_t, ' seconds')
        print("\t\t\t============= MODEL TRAINING FINISHED =============")

        return _acc_mean
        pass


if __name__ == '__main__':
    _train_ds, _test_ds = cnn_util.TORCH_load_mnist_dataset()
    myModel = CNN_torch()
    _acc = myModel.Train(_train_ds)

    import matplotlib.pyplot as plt

    # test data
    _acc_step = 0.0
    for batch_step, (_x_test, _y_test) in enumerate(_test_ds):
        x_test, y_test = Variable(_x_test), Variable(_y_test)
        y_pred = myModel(x_test)
        _acc = torch.where(torch.eq(y_pred, y_test),
                           torch.full_like(y_test, 1.0, dtype=torch.float32),
                           torch.full_like(y_test, 0.0, dtype=torch.float32))
        _acc = torch.mean(_acc)

        if batch_step == 3:  # select the group 3 to display
            img = x_test
            img = torch.reshape(img, [-1, 28, 28, 1])
            pred_y = y_pred.detach()
            for i in range(30): # batch size is 30
                ax = plt.subplot(5, 6, i + 1)
                ax.axis('off')

                ax.imshow(img.numpy()[i, :, :, 0], cmap=plt.get_cmap('gray'))

                if torch.equal(y_test[i], pred_y[i]):
                    ax.set_title(f"{pred_y.numpy()[i] :d}", color='green', fontsize=15)
                else:
                    ax.set_title(f"{pred_y.numpy()[i] :d}", color='red', fontsize=15)

            pass

        _acc_step += _acc
    _acc_mean = _acc_step / len(_test_ds)
    print("\nFinal test dataset accuracy: ", _acc_mean.detach().numpy())

    plt.show()


    pass

