"""
chapter 6 exercise. Generate Tang peoms with RNN, both implemented by Tensorflow
reference by :
https://blog.csdn.net/Irving_zhang/article/details/76664998
https://tensorflow.google.cn/tutorials/text/text_generation?hl=zh_cn

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import json
import time
import peom_process_util



class RNN_TF():
    def __init__(self, term_size, batch_size):
        #super(RNN_TF, self).__init__()

        self._seq_model = None
        self._rnn_units = 1024
        self._embedding_dim = 64
        self._term_size = term_size  # term dictionary size
        self._batch_size = batch_size  # batch size could be different between training and calling
        self._model_type = 'GRU'  # 'SimpleRNN', 'GRU' or 'LSTM'
        self.optimizer = None

        self.SetupModel(term_size, batch_size)
        self.BuildModel()

        pass

    def SetupModel(self, term_size, batch_size):  # simpleRNN, GRU, LSTM
        """

        :param term_size: the size of term dictionary
        :param batch_size: the size of batch input,
        :param modelType:
        :return:
        """
        self._term_size = term_size  # term dictionary size
        self._batch_size = batch_size  # batch size could be different between training and calling

        """
        1st layer. input embedding layer. align input word vector to the same length
        input: shape 3D (batch_num, batch_size, sequence_len))
                        batch_num = 100, batch_size = 366, sequence_len = 17 
                eg. [[b0:0:0], [b0:1:1], ... ,[b0:365:16], --->batch 0
                     [b1:0:0], [b1:1:1], ... ,[b1:365:16], --->batch 1
                     ....
                     [b99:0:0], [b99:1:1], ... ,[b99:365:16]] --->batch 99
        output: shape (batch_size, sequence ,embedding_dim) eg. (366, 17, 64) ---> one batch
        """
        _Embedding = keras.layers.Embedding(input_dim=self._term_size,
                                            output_dim=self._embedding_dim,
                                            batch_input_shape=[batch_size, None],
                                            name="RNN_TF._Embedding")

        """
        2nd layer, RNN layer. simpleRNN, GRU or LSTM
        input: embedding shape
        output: shape (batch_size, sequence, rnnunit) , base one return sequence is True.
        """
        _RNN = None
        if self._model_type == 'SimpleRNN':
            _RNN = keras.layers.SimpleRNN(units=self._rnn_units,
                                            stateful=True,
                                            return_sequences=True,
                                          name="RNN_TF._SimpleRNN")
        elif self._model_type == 'GRU':
            _RNN = keras.layers.GRU(units=self._rnn_units,
                                         stateful=True,
                                         return_sequences=True,
                                    name="RNN_TF._GRU")
        elif self._model_type == 'LSTM':
            _RNN = keras.layers.LSTM(units=self._rnn_units,
                                          stateful=True,
                                          return_sequences=True,
                                     name="RNN_TF._LSTM")
        else:
            print("[ERROR]: mode type [ ", self._model_type, " ] is not supported!")
            raise ValueError("mode type [ ", self._model_type, " ] is not supported!")

        """
        3rd layer, Full connection layer, output as classification logits, output should call softmax
        input: rnn shape
        output: shape (batch_size, sequence, term_size) , category number is term size

        """
        _DNN = keras.layers.Dense(units=self._term_size, name="RNN_TF._Dense")

        self._seq_model = keras.Sequential()
        self._seq_model.add(_Embedding)
        self._seq_model.add(_RNN)
        self._seq_model.add(_DNN)

        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        pass

    @tf.function
    def _train_one_step(self, x_input, y_target):
        with tf.GradientTape() as tape:
            y_pred = self._calc_forward(x_input)
            loss, acc = self.cal_loss_and_accuracy(y_pred, y_target)

        grads = tape.gradient(loss, self._seq_model.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(grads, self._seq_model.trainable_variables))
        return loss, acc
        pass

    def Train(self, x_batch, y_batch):
        print("\t\t\t============= MODEL TRAINING BEGIN =============")
        batch_num = len(x_batch)
        self.ResetState()

        for i in range(batch_num):
            x_data = x_batch[i]
            y_data = y_batch[i]
            loss, acc = self._train_one_step(x_data, y_data)
            if i % 10 == 0:
                print('batch', i,
                      ': loss ', loss.numpy(), '; accuracy ', acc.numpy())
            if acc > 0.8:
                print('\t\t\tIteration Terminated!!!\n',
                      '; accuracy', acc.numpy())
                break
        _f_chpt_file = self.SaveModel()
        print("\t\t\t============= MODEL TRAINING FINISHED =============")

        return loss, acc, _f_chpt_file

    def LoadModel(self, filename):
        print("----- load model -------")
        self._seq_model.load_weights(filepath=filename, by_name=False)
        pass

    def BuildModel(self):
        print("----- build model -------")
        self._seq_model.build(input_shape=tf.TensorShape([self._batch_size, None]))
        self._seq_model.summary()

    def ResetState(self):
        self._seq_model.reset_states()

    def SaveModel(self):
        print("---- save model ------")
        checkpoint_dir = r".\model\Keras-TF"
        checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{int(time.time()):d}.tf")  # save as TF format
        self._seq_model.save_weights(checkpoint_prefix)
        return checkpoint_prefix

    def __call__(self, inp):
        y_pred = self._calc_forward(inp)
        out_logits = tf.argmax(y_pred, axis=-1)
        return out_logits

    def _calc_forward(self, inp):
        output = self._seq_model(inp)
        return output

    def cal_loss_and_accuracy(self, y_pred, y_target):
        y_target_1 = tf.one_hot(y_target, depth=self._term_size)
        losses = tf.nn.softmax_cross_entropy_with_logits(y_target_1, y_pred)

        loss = tf.reduce_mean(losses)

        out_pred = tf.argmax(y_pred, axis=-1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(out_pred, y_target), dtype=tf.float32))

        return loss, accuracy

def gen_sentence(model, ds):
    if ds is None:
        return None
    if model is None:
        return None
    num_generate = 20
    with open(r".\data\input.txt", mode='r', encoding='utf-8') as f:
        strText = f.readline()
        start_Text = strText.split('、')
        print("starting text as:")
        print(start_Text)
        f.close()
    input_eval = list(map(ds.word_to_idx, start_Text))
    print(input_eval)
    with open(r".\data\output.txt", mode='w+', encoding='utf-8') as f_1:
        for inp_x in input_eval:
            inp_val = tf.expand_dims([inp_x], 0)
            text_generated = [ds.idx_to_word(inp_x)]
            for i in range(num_generate):
                pred_y = model(inp_val)
                pred_y = tf.squeeze(pred_y, 0).numpy()[0]
                text_generated.append(ds.idx_to_word(pred_y))
                inp_val = tf.expand_dims(list(map(ds.word_to_idx, text_generated)), 0)
            print(text_generated)

            str_generated = ""
            for c in text_generated:
                if (c == peom_process_util.start_token) or \
                        (c == peom_process_util.end_token) or \
                        (c == ' '):
                    continue
                str_generated += "".join(c)
                if len(str_generated) >= 7:
                    break
            str_generated += '\n'
            f_1.writelines(str_generated)
    f_1.close()
    pass


if __name__ == '__main__':
    print("\n\t\tChapter6. Recurrent Neural Network, RNN exercise (TF) ")
    print("STEP 1. Load training data")
    print("STEP 2. Training model")
    print("STEP 3: Generate text by model")
    print("other, bye-bye !")

    b_loop = True
    ds = None
    terms = None
    x_batch = None
    y_batch = None
    myModel = None
    _latest_ckpt_file = None
    while (b_loop):
        try:
            step = int(input("please input step:"))
            if step == 1:
                ds = peom_process_util.poem_dataset()
                peom_vecs, terms_in_map, terms = ds.load_dataset(r'.\data\poems.dat', b_shuffle=True)
                pass
            elif step == 2:
                x_batch, y_batch = ds.generate_batch(batch_num=100)
                myModel = RNN_TF(term_size=len(terms), batch_size=len(x_batch[0]))
                myModel.BuildModel()
                for epoch in range(10):
                    print('----------------- epoch ', epoch, ' -----------------------')
                    _loss, _acc, _latest_ckpt_file = myModel.Train(x_batch, y_batch)
                    ds.shuffle_dataset()
                    x_batch, y_batch = ds.generate_batch(batch_num=100)
                    print('loss ', _loss.numpy(), '; accuracy ', _acc.numpy())
                    print('----------------- epoch ', epoch, ' -----------------------\n')

                #_loss, _acc, _latest_ckpt_file = myModel.Train(x_batch, y_batch)
                with open(r".\model\model.json", mode='r+', encoding="utf-8") as f:
                    _config = json.load(f)
                    _config["TF-Keras"] = _latest_ckpt_file
                    f.seek(0, 0)
                    json.dump(_config, f)
                    f.close()
                pass
            elif step == 3:
                """
                input shape is different, it should reload model
                """
                if myModel is None:
                    myModel = RNN_TF(term_size=len(terms), batch_size=1)
                    with open(r".\model\model.json", mode='r+') as f:
                        _config = json.load(f)
                        _latest_ckpt_file = _config["TF-Keras"]
                        f.close()
                else:
                    myModel.SetupModel(term_size=len(terms), batch_size=1)
                myModel.LoadModel(_latest_ckpt_file)
                myModel.BuildModel()

                myModel.ResetState()
                gen_sentence(myModel, ds)
                pass
            else:
                b_loop = False
        except:
            exit(code=-1)

