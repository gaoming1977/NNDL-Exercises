#chapter 3.svm multi classification exercise
"""
reference:
https://www.jianshu.com/p/ba59631855a3
https://www.jianshu.com/p/ce96f1a04b72

https://blog.csdn.net/Big_Pai/article/details/89482752
"""


import numpy as np
import matplotlib.pyplot as plt


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


def exclusive_randint(low, high, ex):
    ret = ex
    while ret == ex:
        ret = np.random.randint(low, high)
    return ret


class SVM:
    def __init__(self, f, max_iter=1000, C=1.0, epsilon=0.01):
        # external parameters
        self.max_iter = max_iter
        self.C = C
        self.epsilon = epsilon
        self.kernel_f = f

        # internal parameters
        self.train_m = None  # train matrix [[x1, t1], ..., [xN, tN]]
        self.alpha = None
        #self.w = None
        self.b = 0.0
        pass

    def train(self, data_train):
        """
        SVM model trainer,
        :param data_train: data_train [x1, x2, t], shape(N,3)
        :return: support vectors
        """
        # save train matix
        self.train_m = data_train

        x_train = data_train[:, 0:-1]  # x_train shape(N, m) [x1, x2, ..., xm]
        t_train = data_train[:, -1]  # t_train shape(N, 1) [t] t={-1,1}

        """
        SMO 算法
        reference: https://blog.csdn.net/suipingsp/article/details/41645779/
        https://zhuanlan.zhihu.com/p/29212107
        """
        row_num = x_train.shape[0]
        col_num = x_train.shape[1]
        self.alpha = np.zeros((row_num), dtype=np.float32)

        # self.k_trans = self.calc_K_trans_train(data_train)
        iter_cnt = 0
        k_trans = self.calc_K_trans_vecs(x_train, x_train)

        pairs_alpha_changed = 0

        # main loop
        while iter_cnt < self.max_iter:
            alpha_prev = np.copy(self.alpha)
            iter_cnt += 1

            # inner loop
            for i in range(row_num):
                x_i, y_i = x_train[i, :], t_train[i]
                # step 1. check one sample not fit KKT condition
                """
                ai == 0 ---> yi*f(xi) >=1 
                0 < ai < C --> yi*f(xi) = 1
                ai == C --> yi*f(xi) <= 1
                
                f(xi) = (j=1,N)Σ(aj*yj*K(xi, xj) + b
                """

                f_xi = self(np.reshape(x_i, (-1, col_num)))
                yi_f_xi = y_i * f_xi

                if self.alpha[i] == 0 and yi_f_xi >= 1:
                    continue
                if (self.alpha[i] >0 and self.alpha[i] < self.C) and yi_f_xi == 1:
                    continue
                if (self.alpha[i] == self.C) and yi_f_xi <= 1:
                    continue

                # step 2. select j not equal i
                j = exclusive_randint(0, row_num-1, i)  # Get random i <> j
                x_j, y_j = x_train[j, :], t_train[j]

                #  step3. η = 2k<xi, xj> - k<xi, xi> - k<xj, xj>
                eta_ij = 2.0 * k_trans[i, j] - k_trans[i, i] - k_trans[j, j]
                if eta_ij == 0:
                    continue

                #  step4. calc alpha up-bounder 'H' and low-bounder 'L'
                alpha_i_old = float(self.alpha[i])
                alpha_j_old = float(self.alpha[j])
                if y_i == y_j:
                    L = max(0, alpha_j_old + alpha_i_old - self.C)
                    H = min(self.C, alpha_i_old + alpha_j_old)
                else:
                    L = max(0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)

                #  step5. calc Ei, Ej
                E_i = self(np.reshape(x_i, (-1, col_num))) - y_i
                E_j = self(np.reshape(x_j, (-1, col_num))) - y_j

                #  step6. calc new alpha[j]
                self.alpha[j] = alpha_j_old - float(y_j * (E_i - E_j)) / eta_ij

                # clip aj
                if self.alpha[j] >= H:  # aj > H
                    alpha_temp = H
                elif self.alpha[j] >= L:  # L <= aj <= H
                    alpha_temp = self.alpha[j]
                else:  # aj < L
                    alpha_temp = L

                #  step7. calc new alpha[i]
                self.alpha[i] = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_temp)

                # step8. calc new b
                # b1 = b - Ei - yi*(ai - ai<old>)* k<xi, xi> - yj(aj - aj<old>)*k<xi,xj>
                # b2 = b - Ej - yi*(ai - ai<old>)* k<xi, xi> - yj(aj - aj<old>)*k<xj,xj>
                b1 = self.b - E_i - y_i*(self.alpha[i] - alpha_i_old) * k_trans[i, i] - \
                    y_j * (self.alpha[j] - alpha_j_old) * k_trans[i, j]
                b2 = self.b - E_j - y_i*(self.alpha[i] - alpha_i_old) * k_trans[i, j] - \
                     y_j * (self.alpha[j] - alpha_j_old) * k_trans[j, j]

                if(self.alpha[i] >0 and self.alpha[i] < self.C):
                    self.b = b1
                elif(self.alpha[j] >0 and self.alpha[j] < self.C):
                    self.b = b2
                else:
                    self.b = (b1 + b2)/2.0

                pairs_alpha_changed += 1

            diff = np.linalg.norm(self.alpha - alpha_prev)
            if iter_cnt % 10 == 0:
                print(f'Iteration {iter_cnt :d}, diff is {diff :.4f}, update {pairs_alpha_changed :d} alpha pairs')
            if diff < self.epsilon:
                print(f'Iteration terminate, as diff {diff :.4f} smaller than convergence epsilon {self.epsilon :.4f}')
                print(f'{iter_cnt :d} iterations')
                break
            if iter_cnt >= self.max_iter:
                print(f'Iteration terminate, as {iter_cnt :d} reach the max {self.max_iter :d}')
                return

        # collect support vectors
        support_vector_idx = np.where(abs(self.alpha) > 0)[0]
        support_vectors = x_train[support_vector_idx, :]
        return support_vectors

    def __call__(self, inp):
        """
        y = wx + b
        :param inp: inp [x1, x2, ... , xm] ,shape(N,m)
        :return: y, shape(N,)

        f(x) = sgn(<i=1,N>Σai*yi*k(xi, x) + b)
        """
        #
        if self.train_m is None:
            print("SVM model is not trained")
            return None

        x_train = self.train_m[:, 0:-1]
        t_train = self.train_m[:, -1]

        #n, m = x_train.shape[0], x_train.shape[1]

        # using kernel function to transform input vector
        inp_trans = self.calc_K_trans_vecs(x_train, inp)

        ret = np.dot(np.multiply(self.alpha, t_train).T, inp_trans) + self.b
        #ret = np.dot(inp, self.w) + self.b
        ret = np.sign(ret).astype(int)

        return ret

    def calc_K_trans_vecs(self, X, vecs):
        """
        calc K matrix of kernel transform
        :param X: X: shape(N,m) [xi1,xi2,..., xim] - xi
        :param y: y, shape(d,m) [yj1, yj2,..., yjm] - yj
        :return: K, shape(N,d) = [k(x1,y1), k(x1,y2), ..., k(x1, yd),
                                k(x2,y1), k(x2,y2), ..., k(x2, yd),
                                ...
                                k(xN,y1), k(xN,2), ..., k(xN,yd)]
        """
        n, m = X.shape[0], X.shape[1]

        if X.shape[1] != vecs.shape[1]:
            print('input vector dim is not equal to train vectors, reshape input!')
            vecs = vecs.reshape((-1, m))

        d = vecs.shape[0]
        ret_K = np.zeros((n, d), np.float32)

        for i in range(n):
            x_i = X[i, :]
            for j in range(d):
                v_j = vecs[j, :]
                ret_K[i, j] = self.kernel_f(x_i, v_j)
        return ret_K
'''
    def calc_w(self, alpha, y, x):
        """
        calculate W parameter
        w = Σ(i,m) ai * yi • xi
        :param alpha: shape(N,)
        :param y: shape(N,)
        :param x: shape(N,m)
        :return: w: shape(m,)
        """
        ret = np.dot(x.T, np.multiply(alpha, y))
        return ret

    def calc_b(self, y, x):
        """
        calculate b parameter, MUST call after calc_w
        b = y -  x • w
        :param y: shape(N,)
        :param x: shape(N, m)
        :return: b: shape(N, ) reduce_mean()
        """
        ret = np.mean(y - np.dot(x, self.w))
        return ret
        pass
'''

"""
kernel function. 核函数，多用于样本数据非线性可分，即不能通过线性超平面将训练样本分类，需要将原始空间映射到
一个更高维的特征空间。参见《机器学习-周志华》P126
常见的核函数：
linear kernel 线性核函数, 样本数据需要满足线性可分
polynormal kernel 多项式核
gaussine kernel 高斯核 RBF
拉普拉斯核
sigmoid核
"""

def linear_kernel(xi, xj):
    """
    k(xi, xj ) = xi • xj^T
    linear kernel function
    :param : xi, xj is shape(1, m) vector
    :return: ret is a value
    """
    ret = np.dot(xi, xj)
    return ret

def polynormal_kernel(xi, xj, d=3):
    """
    k(xi, xj) = (xi • xj^T + 1)**d
    :param xi:
    :param xj:
    :param d:
    :return:
    """
    ret = (np.dot(xi, xj) + 1) ** d
    return ret

def gaussian_kernel(xi, xj, sigma=0.05):
    """
    k(xi, xj) = exp( - sigma * ||xi - xj||**2)
    :param xi:
    :param xj:
    :param sigma:
    :return:
    """
    ret = np.exp(- sigma * np.linalg.norm(xi - xj) ** 2)
    return ret

def sigmoid_kernel(xi, xj, theta=-1):
    """
    k(xi, xj) = tanh( xi • xj^T + θ)
    :param xi:
    :param xj:
    :param beta:
    :param therta:
    :return:
    """
    ret = np.tanh(np.dot(xi, xj) + theta)
    return ret


def draw(xy_points, t_val, b_t):
    if len(xy_points) != len(t_val):
        return
    # draw dots
    for i in range(len(xy_points)):
        if b_t:  # test data
            ax.scatter(xy_points[i, 0], xy_points[i, 1], s=10, marker='*', c='c')
        else:  # train data
            if t_val[i] == 1:
                ax.scatter(xy_points[i, 0], xy_points[i, 1], s=10, marker='^', c='b')
            elif t_val[i] == -1:
                ax.scatter(xy_points[i, 0], xy_points[i, 1], s=10, marker='v', c='g')
            else:
                ax.scatter(xy_points[i, 0], xy_points[i, 1], s=10, marker='o', c='r')



if __name__ == '__main__':
    print("======== Welcome, SVM Exercise =========")
    print("1. linear SVM")
    print("2. non-linear SVM - Poly, Gaussian(RBF), or Sigmoid kernel")
    print("3. multi classification SVM")
    print(" . other, exit")

    try:
        choice = int(input())
    except IOError as err:
        print("invalid input, exit")
        choice = 4


    # load data
    title = 'SVM Example '
    if choice == 1:
        data_train = load_data(r'.\data\train_linear.dat')
        data_test = load_data(r'.\data\test_linear.dat')
        title += '- linear'
    elif choice == 2:
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

    if choice == 1 or choice == 3:
        model = SVM(linear_kernel)
    else:# choice == 2:
        #model = SVM(polynormal_kernel)
        model = SVM(gaussian_kernel)
        #model = SVM(sigmoid_kernel)

    support_vec = model.train(data_train)

    # shape(N,1) [pred_t]
    pred_train = model(x1_x2_train)
    pred_test = model(x1_x2_test)

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
        sc.set_edgecolor('m')

    # draw hyper plane
    step = 0.5
    x = np.arange(x_min, x_max, step)
    y = np.arange(y_min, y_max, step)

    X, Y = np.meshgrid(x, y)
    input_data = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
    z = model(input_data)
    Z = z.reshape(X.shape)
    ax.contourf(X, Y, Z, alpha=0.1)

    plt.show()
    pass

