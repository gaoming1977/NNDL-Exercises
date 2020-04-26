# svm bi-classification

import numpy as np

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
        # self.w = None
        self.b = 0.0
        pass

    def train(self, data_train):
        """
        SVM model trainer,
        :param data_train: data_train [x1, x2, t], shape(N,3)
        :return: support vectors shape(n,2)
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

        # main loop
        while iter_cnt < self.max_iter:
            alpha_prev = np.copy(self.alpha)
            iter_cnt += 1
            pairs_alpha_changed = 0

            # inner loop
            for i in range(row_num):
                x_i, y_i = x_train[i, :], t_train[i]
                # step 1. select sample i which not fit KKT condition
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
                if (self.alpha[i] > 0 and self.alpha[i] < self.C) and yi_f_xi == 1:
                    continue
                if (self.alpha[i] == self.C) and yi_f_xi <= 1:
                    continue

                # step 2. select j not equal i
                j = exclusive_randint(0, row_num - 1, i)  # Get random i <> j
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
                b1 = self.b - E_i - y_i * (self.alpha[i] - alpha_i_old) * k_trans[i, i] - \
                     y_j * (self.alpha[j] - alpha_j_old) * k_trans[i, j]
                b2 = self.b - E_j - y_i * (self.alpha[i] - alpha_i_old) * k_trans[i, j] - \
                     y_j * (self.alpha[j] - alpha_j_old) * k_trans[j, j]

                if (self.alpha[i] > 0 and self.alpha[i] < self.C):
                    self.b = b1
                elif (self.alpha[j] > 0 and self.alpha[j] < self.C):
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2.0

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

        # n, m = x_train.shape[0], x_train.shape[1]

        # using kernel function to transform input vector
        inp_trans = self.calc_K_trans_vecs(x_train, inp)

        ret = np.dot(np.multiply(self.alpha, t_train).T, inp_trans) + self.b
        # ret = np.dot(inp, self.w) + self.b
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


def sigmoid_kernel(xi, xj, k=0.0003, c=-1):
    """
    k(xi, xj) = tanh( k* xi • xj^T + c)
    :param xi:
    :param xj:
    :param beta:
    :param therta:
    :return:
    """
    var = k * np.dot(xi, xj) + c
    # ret = 1.0 / (1.0 + np.exp(- var))
    ret = np.tanh(var)
    return ret


def main():
    pass


if __name__ == '__main__':
    main()
