#chapter 3.svm multi classification exercise
"""
reference:
https://www.jianshu.com/p/ba59631855a3
https://www.jianshu.com/p/ce96f1a04b72

https://blog.csdn.net/Big_Pai/article/details/89482752
"""


import numpy as np
import matplotlib.pyplot as plt

import svm_bi_classification as svm_bi
import svm_multi_classification as svm_multi


def load_data(filename):
    with open(filename) as f:
        data = []
        f.readline()  # 跳过第一行
        for line in f:
            line = line.strip().split()
            x1_f = float(line[0])
            x2_f = float(line[1])
            t_f = int(line[2])
            data.append([x1_f, x2_f, t_f])
        ret_f = np.array(data)
        np.random.shuffle(ret_f)
        return ret_f


def eval_accuracy(label, pred):
    return np.sum(label == pred)/len(label)


def draw(xy_points, t_val, b_t):
    if len(xy_points) != len(t_val):
        return
    u_t_val = np.unique(t_val)
    # draw dots
    for i in range(len(xy_points)):
        if b_t:  # test data
            ax.scatter(xy_points[i, 0], xy_points[i, 1], s=10, marker='*', c='c')
        else:  # train data
            if t_val[i] == u_t_val[0]:
                ax.scatter(xy_points[i, 0], xy_points[i, 1], s=10, marker='^', c='b')
            elif t_val[i] == u_t_val[1]:
                ax.scatter(xy_points[i, 0], xy_points[i, 1], s=10, marker='v', c='g')
            else:
                ax.scatter(xy_points[i, 0], xy_points[i, 1], s=10, marker='d', c='k')


if __name__ == '__main__':
    print("======== Welcome, SVM Exercise =========")
    print("1. linear SVM")
    print("2. non-linear SVM - Poly, Gaussian(RBF), or Sigmoid kernel")
    print("3. multi classification SVM")
    print("4. sklearn SVM - sigmoid kernel")
    print(" . other, exit")

    try:
        choice = int(input())
    except IOError as err:
        print("invalid input, exit")
        choice = -1


    # load data
    title = 'SVM Example '
    if choice == 1:
        data_train = load_data(r'.\data\train_linear.dat')
        data_test = load_data(r'.\data\test_linear.dat')
        title += '- linear'
    elif choice == 2 or choice == 4:
        data_train = load_data(r'.\data\train_kernel.dat')
        data_test = load_data(r'.\data\test_kernel.dat')
        title += '- non linear'
    elif choice == 3:
        data_train = load_data(r'.\data\train_multi.dat')
        data_test = load_data(r'.\data\test_multi.dat')
        title += '- multi classification'
    else:
        exit()

    # extract dataset
    x1_x2_train = data_train[:, :-1]
    t_train = data_train[:, 2]
    #
    x1_x2_test = data_test[:, :-1]
    t_test = data_test[:, 2]

    if choice == 1:
        model = svm_bi.SVM(svm_bi.linear_kernel)
    elif choice == 2:
        #model = SVM(polynormal_kernel)
        model = svm_bi.SVM(svm_bi.gaussian_kernel)
        #model = svm_bi.SVM(svm_bi.sigmoid_kernel)
    elif choice == 3:
        model3 = svm_multi.SVM_Multi()
    elif choice == 4:
        from sklearn.svm import SVC
        model4 = SVC(kernel='sigmoid')
    else:
        exit()

    support_vec = None
    if choice == 4:
        model4.fit(x1_x2_train, t_train)
        pred_train = model4.predict(x1_x2_train)
        pred_test = model4.predict(x1_x2_test)
    elif choice == 1 or choice == 2:
        support_vec = model.train(data_train)
        # shape(N,1) [pred_t]
        pred_train = model(x1_x2_train)
        pred_test = model(x1_x2_test)
    elif choice == 3:
        support_vec = model3.train(data_train)
        pred_train = model3(x1_x2_train)
        pred_test = model3(x1_x2_test)
    else:
        support_vec = None
        pred_train = None
        pred_train = None

    # evaluate model
    acc_train = eval_accuracy(label=t_train, pred=pred_train)
    acc_test = eval_accuracy(label=t_test, pred=pred_test)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))


    def init_canvas():
        x_min = min(np.min(data_train[:, 0]), np.min(data_test[:, 0]))
        y_min = min(np.min(data_train[:, 1]), np.min(data_test[:, 1]))
        x_max = max(np.max(data_train[:, 0]), np.max(data_test[:, 0]))
        y_max = max(np.max(data_train[:, 1]), np.max(data_test[:, 1]))

        fig, ax = plt.subplots(figsize=(8, 5))
        plt.title(title, fontsize=10)
        plt.ylabel('Y')
        plt.xlabel('X')
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)

        return fig, ax, x_min, x_max, y_min, y_max

    fig, ax, x_min, x_max, y_min, y_max = init_canvas()
    draw(x1_x2_train, t_train, b_t=False)
    #draw(x1_x2_test, t_test, b_t=True)

    # draw support vectors
    if support_vec is not None:
        sc = ax.scatter(support_vec[:, 0], support_vec[:, 1], s=40, marker='o')
        sc.set_facecolor('none')
        sc.set_edgecolor('r')

    # draw hyper plane
    step = 0.5
    x = np.arange(x_min, x_max, step)
    y = np.arange(y_min, y_max, step)

    X, Y = np.meshgrid(x, y)
    input_data = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
    if choice == 4:
        z = model4.predict(input_data)
    elif choice == 1 or choice == 2:
        z = model(input_data)
    elif choice == 3:
        z = model3(input_data)
    else:
        z = None
    Z = z.reshape(X.shape)
    ax.contourf(X, Y, Z, alpha=0.1)

    plt.show()
    pass

