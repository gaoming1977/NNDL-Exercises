#chapter 5. Convolutional Neural Networks, CNN exercises
"""
exercise 1: tensorflow
exercise 2: pytorch

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cnn_util


def main():
    pass


if __name__ == '__main__':
    print("\n\t\tchapter5. Convolutional Neural Network, CNN exercise ")
    print("1. CNN of Tensorflow for MNIST")
    print("2. CNN of Pytorch for MNIST")
    print("other, bye-bye !")



    ds, test_ds = cnn_util.TF_load_mnist_dataset()
    pass

