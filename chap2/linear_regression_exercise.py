# chapter2 linear regression exercise

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib


def load_data(filename):
    """load data"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


def evaluate(ys, ys_pred):
    """evaluate model"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred)**2))
    return std

def identity_basis(x, feature_num=10):
    """x^, 增广向量"""
    ret = np.expand_dims(x, axis=1)
    return ret

def multinomial_basis(x, feature_num=10):
    """多项式基函数"""
    """
    y = b + w1x + w2x^2 + w3x^3 + ... + w(feature_num)x^(feature_num)
    ret as [x,x^2,x^3,...,x^(feature_num)]
    """
    x = np.expand_dims(x, axis=1)
    """ add code"""
    ret = np.copy(x)
    for i in range(2, feature_num):
        x1 = np.power(x, i)
        ret = np.concatenate([ret, x1], axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    """高斯基（核）函数"""
    """
    作用：特征升维，维度为feature_num参数。进行特征矩阵x(n,1)的变换为x(n,feature_num)
    xj为分组中心点c
    K(xi,c) = exp(-sigma*(xi-c)^2)
    m为特征数量 feature_num
    x分为m组，c为第j组中的中心点mean()
    特征矩阵变换为K(n,m) = [[K(x1,c1), K(x1,c2),...,K(x1,cm)],
                    ...
                    [K(xn,c1), K(xn,c2),...,K(xn,cm)]]

    ref: https://blog.csdn.net/qq_23981335/article/details/83747274
        https://www.jianshu.com/p/5cc427f0df33
    """
    #sigma是一个参数
    sigma = 0.3
    x = np.expand_dims(x, axis=1)

    width = (np.max(x) - np.min(x))/feature_num
    #c为中心点集合
    c = np.zeros((feature_num, 1), dtype=np.float64)

    for i in range(feature_num):
        c[i] = np.min(x) + (2 * i + 1) * width / 2.0

    ret = np.zeros((x.shape[0], feature_num))
    for i in range(feature_num):
        k = np.exp(-sigma * (((x - c[i])/width) ** 2))
        ret[:, i] = k[:, 0]

    return ret


def sigmoid_basis(x, feature_num=10):
    """Sigmoid基函数"""
    """
    x切分为feature_num组，c为每组的中心点
    K(x,c) = 1/(1+e^-(x - c))
    m为特征数量 feature_num
    x分为m组，c为第j组中的中心点mean()
    特征矩阵变换为K(n,m) = [[K(x1,c1), K(x1,c2),...,K(x1,cm)],
                    ...
                    [K(xn,c1), K(xn,c2),...,K(xn,cm)]]
    """
    width = (np.max(x) - np.min(x))/feature_num
    #c为中心点集合
    x = np.expand_dims(x, axis=1)
    c = np.zeros((feature_num, 1), dtype=np.float64)
    for i in range(feature_num):
        c[i] = np.min(x) + (2 * i + 1) * width / 2.0

    ret = np.concatenate([x]*feature_num, axis=1)
    for i in range(feature_num):
        k = 1 / (1 + np.exp(-(x - c[i])/width))
        ret[:, i] = k[:, 0]

    return ret


def risk_LSM_fun(x, y, alpha=0, precision=0):
    """
    最小二乘法 Least Square Method
    w = (X'X)^-1X'y 或者 pinv(X).y
    pinv(X)为输入X矩阵的伪逆矩阵，也是广义逆矩阵。
    X‘为X转置，^-1为矩阵的逆
    :param: 为输入矩阵X，y
    :return: 线性回归模型参数w
    """
    print("alpha is {0}, precision is {1} , but no use in LSM algorithm".format(alpha, precision))
    x_pinv = np.linalg.pinv(x)
    w = np.dot(x_pinv, y)
    return w


def risk_LMS_fun(x, y, alpha=1e-10, precision=1e-7):
    """
    最小均方算法，或者叫梯度下降法，Least Mean Squares
    w(i+1) = w(i) + alpha*X'(Xw-y)
    alpha：为常量，学习率
    w(i+1) - w(i) < delta 迭代停止
    w应该为X的列，
    :param x:
    :param y:
    :return: 线性回归模型参数w
    """
    #需要先扩充y的维度
    y0 = np.expand_dims(y, axis=1)
    print("alpha is {0}, precision is {1} , and VERY important in LMS algorithm".format(alpha, precision))
    #alpha = 1e-10 #学习率，初始化常量

    w = np.zeros((x.shape[1], 1))#w初始矩阵
    count = 0
    while True:
        count += 1
        w0 = np.copy(w)
        y1 = np.dot(x, w)
        err = y0 - y1
        """
        gradient = np.dot(err.T, x)
        w = w + alpha * gradient.T
        """
        gradient = np.dot(x.T, err)
        w = w + alpha*gradient
        w_gradient = abs(w - w0).mean()

        if w_gradient < precision or count > 1e5:#趋近于0 或者迭代1e5次
            print("{0} Iterrate, the gradient is : {1}".format(count, w_gradient))
            break
    return w
    pass


def train_func(x_train, y_train):
    """trainer function"""
    #basis_fun = identity_basis
    #basis_fun = multinomial_basis
    #basis_fun = gaussian_basis
    basis_fun = sigmoid_basis
    f_num = 10
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_fun(x_train, feature_num=f_num)
    phi = np.concatenate([phi0, phi1], axis=1)

    print("phi.shape is ", phi.shape)
    print("y_train.shape is ", y_train.shape)
    """计算w"""
    '''最小二乘法计算w'''
    #risk_fun = risk_LSM_fun
    '''梯度下降法计算w'''
    risk_fun = risk_LMS_fun
    w = risk_fun(phi, y_train, alpha=1e-5, precision=1e-5)

    print("w is :\n", w)
    ##

    def linear_regression_model(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_fun(x, feature_num=f_num)
        phi = np.concatenate([phi0, phi1], axis=1)
        y = np.dot(phi, w)
        return y
    pass

    return linear_regression_model

# main function
def main():
    train_file = r'.\train.dat'
    test_file = r'.\test.dat'

#    load data
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    f = train_func(x_train, y_train)

    y_train_pred = f(x_train)
    std = evaluate(y_train, y_train_pred)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

    y_test_pred = f(x_test)
    std = evaluate(y_test, y_test_pred)
    print('测试集预测值与真实值的标准差：{:.1f}'.format(std))

    plt.plot(x_train, y_train, 'ro', markersize=3)
    plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()


    pass


if __name__ == '__main__':
    main()
    pass

