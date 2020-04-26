# this is the multi-classification SVM
"""
SVM multi classifier
Usually, SVM is using as bi classifying. It should be combine several bi-classifier to achieve multi classfication.
The method could be 'one-versus-rest', or 'one-versus-one'
Here, using 'one-versus-one' method
reference:
https://blog.csdn.net/xfchen2/article/details/79621396

"""
import numpy as np
import svm_bi_classification as svm_bi


class SVM_Multi:

    def __init__(self, max_iter=1000, C=1.0, epsilon=0.01):
        self.f = svm_bi.linear_kernel
        self.max_iter = max_iter
        self.C = C
        self.epsilon = epsilon

        self.svm_classifier = dict()

        self.train_data = None
        self.train_label = None
        self.cat_label = None  # label category [-1, 0, 1] order to up

        pass

    def train(self, train_data):
        # get category number
        label_data = train_data[:, -1]  # t_train shape(N, 1) [t] t={-1,0,1}

        cat_label = np.unique(label_data)  # label category
        cat_n = len(cat_label)
        k = cat_n * (cat_n - 1)/2
        if k <= 0:
            print("error, category number should > 2")
            return
        self.cat_label = cat_label

        # save original train_data and train_labels
        self.train_data = train_data
        self.train_label = label_data

        ret_support_vec = None

        # train classifier
        for i in range(cat_n - 1):
            # select the classifier and related samples
            # for example,3 classifier (AB, AC , BC), the related samples as ((0,-1), (0,1), (-1,1))
            label_ci = int(self.cat_label[i])

            for j in range(i + 1, cat_n):
                label_cj = int(self.cat_label[j])

                cls_idx = gen_cls_key(label_ci, label_cj)
                cls_train_data_sel_i = np.where(label_data == label_ci)
                cls_train_data_sel_j = np.where(label_data == label_cj)
                cls_train_data_sel = np.unique(np.concatenate(
                        (cls_train_data_sel_i, cls_train_data_sel_j)))
                cls_train_data = train_data[cls_train_data_sel, :]

                # format the label as (-1, 1)

                idx_1 = np.where(cls_train_data[:, -1] == label_ci)
                cls_train_data[range(cls_train_data.shape[0]), -1] = -1
                cls_train_data[idx_1, -1] = 1

                #  initialize one classifier
                o_svm = svm_bi.SVM(svm_bi.linear_kernel,
                                   self.max_iter,
                                   self.C,
                                   self.epsilon)

                print(f"... classifier {cls_idx} begin training...")
                cls_vec = o_svm.train(cls_train_data)

                if cls_vec is not None:
                    if ret_support_vec is None:
                        ret_support_vec = np.copy(cls_vec)
                    else:
                        ret_support_vec = np.concatenate((ret_support_vec, cls_vec), axis=0)
                        #np.append(support_vec, ret_vec, axis=0)

                #  save classifier
                print(f"... classifier {cls_idx} trained ...\n")
                self.svm_classifier[cls_idx] = o_svm
                pass

        return np.array(list(set([tuple(t) for t in ret_support_vec])))  # remove duplicated support vectors
        pass

    def __call__(self, input_data):
        """
        do the predict
        :param input_data: shape(N,m) [x1, x2, ..., xm]-Xi [X1,X2,...,XN]
        :return: y shape(N,)
        """
        if self.train_data is None:
            print("model was not trained, abort!!!")
            return

        cat_n = len(self.cat_label)

        # votes, shape(N, cat_n)
        votes = np.zeros((input_data.shape[0], cat_n), dtype=np.int)

        for i in range(cat_n - 1):
            label_ci = self.cat_label[i].astype(int)
            for j in range(i + 1, cat_n):
                label_cj = self.cat_label[j].astype(int)

                cls_idx = gen_cls_key(label_ci, label_cj)
                o_svm = self.svm_classifier[cls_idx]
                if o_svm is None:
                    print(f"error: classifier {cls_idx} no found!")
                    continue

                pred_label_ij = o_svm(input_data)
                idx_pred_i_p = np.where(pred_label_ij == 1)
                idx_pred_i_n = np.where(pred_label_ij == -1)

                # which predict as "1" means AB as A class, "-1" as B class
                votes[idx_pred_i_p, i] += 1
                votes[idx_pred_i_n, j] += 1
            pass

        # summarize the votes and
        sample_n = input_data.shape[0]
        ret_vec = np.zeros(sample_n)
        max_cols = np.argmax(votes, axis=1)
        for i in range(sample_n):
            ret_vec[i] = self.cat_label[max_cols[i]].astype(int)
            pass
        return ret_vec
        pass

def gen_cls_key(label_i, label_j):
    str = f"cls[{label_i :d}, {label_j :d}]"
    return str

def vec_to_onehot(vec):
    n = vec.shape[0]
    c = len(np.unique(vec))
    oh = np.zeros((n, c))
    """
    循环赋值，等效为
    for i in range(n):
        oh[i, vec[i]] = 1
    """
    oh[range(n), vec] = 1
    return oh


def main():
    vector = np.array([1, -1, 1, 0, -1, 2, -2, 1, 0, -2, -1])
    print(vector)
    print(vec_to_onehot(vector))


if __name__ == "__main__":
    main()

