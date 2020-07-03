"""
load peom.txt files as training data
refence: https://blog.csdn.net/Irving_zhang/article/details/76038710
"""
import numpy as np
import collections
import random

import io
import json

start_token = 'G'
end_token = 'E'

stop_tokens = [start_token, '_', '(', ')', '（', '）', '《', '》', '[', ']', end_token]


class poem_dataset:
    def __init__(self):
        self._term_dic = None
        self._term_idx_map = None
        self._peoms_vecs = None
        pass

    def load_dataset(self, filename, b_shuffle=False):
        """
        process poems.txt split terms and count term frequency as term vectors
        :param filename: the train dataset file
        :return:
            peom_vecs, as term and its frequency
                data type: 2-dimension list
                    [term1_idx, term2_idx, ...., termN_idx], ---> one poem
                    [term1_idx, term2_idx, ..., termM_idx], ---> one poem
                    ...
                    [term1_idx, term2_idx, ..., termC_idx] ---> one poem
                    ]
                    M != N,..., != C
            term_idx_map_dic, as map table from term to index
                data type: dict
                    { term1: term1_idx, term2: term2_idx, ..., termN : termN_idx}
            terms_dic, as terms table
                data type: tuple
                    ( term1, term2, ... , termN)
        """
        peoms_content = []

        with open(file=filename, mode='r', encoding='utf-8') as f:
            print("--- begin parse dataset ---")
            for line in f.readlines():
                try:
                    _title, _content = line.strip().split(':')
                    _content = _content.replace(' ', '')
                    _content = _content.replace('，', ',')
                    _content = _content.replace(',', '')
                    _content = _content.replace('。', '')
                    _b_continue = False
                    for _stoken in stop_tokens:
                        if _stoken in _content:
                            _b_continue = True
                            break
                    if len(_content) < 5 or len(_content) > 80:
                        _b_continue = True
                    if _b_continue:
                        continue
                    _content = start_token + _content + end_token
                    peoms_content.append(_content)

                except ValueError as err:
                    print("line: ", line, "error: ", str(err))
                    continue
        # sort by content length
        peoms_content = sorted(peoms_content, key=lambda _x: len(_x))
        # count term frequency
        all_terms = []
        for poem in peoms_content:
            all_terms += [term for term in poem]
        counter = collections.Counter(all_terms)  # term frequency
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # sort by frequency, from high to low

        terms_dic, _ = zip(*count_pairs)
        terms_dic = terms_dic[:len(terms_dic)] + (' ',)
        term_idx_map_dic = dict(zip(terms_dic, range(len(terms_dic))))
        peoms_vecs = [list(map(term_idx_map_dic.get, poem)) for poem in peoms_content]

        print("total terms count is: ", len(terms_dic))
        print("--- end parse dataset ---")

        if b_shuffle:
            random.shuffle(peoms_vecs)

        self._term_dic = terms_dic
        self._term_idx_map = term_idx_map_dic
        self._peoms_vecs = peoms_vecs
        return peoms_vecs, term_idx_map_dic, terms_dic

    def word_to_idx(self, words):
        return self._term_idx_map[words]
        pass

    def idx_to_word(self, indexs):
        return list(self._term_idx_map.keys())[indexs]
        pass

    def shuffle_dataset(self):
        random.shuffle(self._peoms_vecs)

    def generate_batch(self, batch_num):
        x_batch = []
        y_batch = []
        batch_size = len(self._peoms_vecs) // batch_num
        for i in range(batch_num):
            i_begin, i_end = i * batch_size, (i + 1) * batch_size
            batches = self._peoms_vecs[i_begin: i_end]
            length = max(map(len, batches))

            x_data = np.full((batch_size, length),
                             self.word_to_idx(' '),
                             np.int64)

            for row, batch in enumerate(batches):
                x_data[row, :len(batch)] = batch

            y_data = np.copy(x_data)
            y_data[:, :-1] = x_data[:, 1:]  # move left 1 char

            x_batch.append(x_data)
            y_batch.append(y_data)

        return x_batch, y_batch
    pass

def load_json(filename):
    with open(filename, 'r+', encoding="utf-8") as f:
        _data = json.load(f)
        print(_data["TF-Keras"])
        print(_data["PYTORCH"])
        f.close()


if __name__ == '__main__':
    load_json(r".\model\model.json")


    ds = poem_dataset()
    poem_vecs, *_ = ds.load_dataset(r'.\data\poems.dat', b_shuffle=True)
    print(ds.idx_to_word(0))
    x_data, y_data = ds.generate_batch(50)
    print(x_data)
    print("==============reshuffle==========")
    ds.shuffle_dataset()
    x_data, y_data = ds.generate_batch(50)
    print(x_data)

