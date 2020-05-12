#  chapter4. Neural Network exercise

import numpy as np
import simple_FNN_numpy as s_fnn_np
import simple_FNN_TF as s_fnn_tf
import fnn_util
import tensorflow as tf


def main():
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


if __name__ == '__main__':
    print("\n\t\tchapter4. exercise ")
    print(" simple feedforward neural network (FNN)\n")
    print("1: implemented MNIST dataset with Numpy")
    print("2: implemented MNIST dataset with Tensorflow")
    print("other, bye-bye !")

    myModel = None
    try:
        choice = int(input("please give your option:"))
    except IOError as err:
        choice = -1
    if choice == 1:
        myModel = s_fnn_np.FNN_Model_NP()
        sample_num = 5000
    elif choice == 2:
        myModel = s_fnn_tf.FNN_Model_TF()
        sample_num = -1
    else:
        myModel = None
        exit(code=-1)

    if myModel is not None:
        print("========================= program begin ==================\n")
        (x_train, y_train), (x_test, y_test) = fnn_util.mnist_dataset(num=sample_num)
        print("MNIST sample size:")
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

        loss, accuracy = myModel.train(x_train, y_train)
        if choice == 1:
            _loss = loss
            _acc = accuracy
        elif choice == 2:
            _loss = loss.numpy()
            _acc = accuracy.numpy()
        print('\t\t\tTRAIN DATASET \n final loss', _loss, '; accuracy', _acc)

        print("================ MODEL TESTING ==============")
        x_test_1 = np.copy(x_test)
        x_test_1 = np.reshape(x_test_1, [-1, 28 * 28])
        pred_y = myModel(x_test_1)
        if choice == 1:
            _acc = np.mean(np.equal(pred_y, y_test).astype(np.float32))
        elif choice == 2:
            _acc = tf.reduce_mean(tf.cast(tf.equal(pred_y, y_test), dtype=tf.float32)).numpy()
        print('\t\t\tTEST DATASET \n final accuracy', _acc)

        print("\n================ MODEL TESTING END ===========")

    main()
    print("\n========================= program end ====================")
    pass




