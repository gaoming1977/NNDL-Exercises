"""
chapter 6 exercise. Generate Tang peoms with RNN, both implemented by PyTorch
reference by :
https://blog.csdn.net/Irving_zhang/article/details/76664998
https://tensorflow.google.cn/tutorials/text/text_generation?hl=zh_cn

"""
import os

import torch
import torch.nn as nn

from torch.autograd import Variable

import peom_process_util
import json
import time


class RNN_TORCH():
    def __init__(self, term_size, batch_size, model_type='LSTM'):
        self._term_size = term_size
        self._batch_size = batch_size
        self._model_type = model_type  #'SimpleRNN', 'GRU', 'LSTM'
        self._optimizer = None
        self._loss_f = None

        """
        state parameters
        """
        self._hn = None
        self._cn = None

        #torch.autograd.set_detect_anomaly(True)

        self.SetupModel()
        pass

    def SetupModel(self):

        """

        :return:
        """

        """
        1st layer: embedding layer
        """
        self._seq_Embedding = nn.Sequential(
            nn.Embedding(num_embeddings=self._term_size,
                         embedding_dim=64))
        """
        2nd layer: RNN layer
        direct using Module, not sequence module
        """
        self._seq_RNN = None
        if self._model_type == 'SimpleRNN':
            self._seq_RNN = nn.Sequential(
                nn.RNN(input_size=64,
                       hidden_size=1024,
                       num_layers=1,
                       batch_first=True))
        elif self._model_type == 'GRU':
            self._seq_RNN = nn.Sequential(
                nn.GRU(input_size=64,
                       hidden_size=1024,
                       num_layers=1,
                       batch_first=True))
        elif self._model_type == 'LSTM':
            self._seq_RNN = nn.Sequential(
                nn.LSTM(input_size=64,
                        hidden_size=1024,
                        num_layers=1,
                        batch_first=True))
        else:
            print("error: mode type [ ", self._model_type, " ] is not supported!")
            raise ValueError("error: mode type [ ", self._model_type, " ] is not supported!")

        """
        3rd layer: Full connected layer + softmax
        """
        self._seq_FNN = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=self._term_size,
                      bias=True))

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(params=[
                {'params': self._seq_Embedding.parameters()},
                {'params': self._seq_RNN.parameters()},
                {'params': self._seq_FNN.parameters()}],
                lr=0.001)
        if self._loss_f is None:
            self._loss_f = nn.CrossEntropyLoss()

        self.ResetState()

        self.PrintModel()
        pass

    def SaveModel(self):
        print("---- save model ------")
        checkpoint_dir = r".\model\Torch"
        checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{int(time.time()):d}.pt")  # save as pt format
        torch.save(
            {
                'Embedding_STATE_DICT': self._seq_Embedding.state_dict(),
                'RNN_STATE_DICT': self._seq_RNN.state_dict(),
                'FNN_STATE_DICT': self._seq_FNN.state_dict(),
            }, checkpoint_prefix
        )
        return checkpoint_prefix
        pass

    def LoadModel(self, filename):
        print("----- load model -------")

        checkpoint = torch.load(filename)
        self._seq_Embedding.load_state_dict(checkpoint['Embedding_STATE_DICT'])
        self._seq_Embedding.eval()
        self._seq_RNN.load_state_dict(checkpoint['RNN_STATE_DICT'])
        self._seq_RNN.eval()
        self._seq_FNN.load_state_dict(checkpoint['FNN_STATE_DICT'])
        self._seq_FNN.eval()

        self.PrintModel()

        pass

    def PrintModel(self):
        print("================= Model Summary =================\n")

        print("----------------- Embedding Layer -----------------")
        print(self._seq_Embedding)

        print(f"\n----------------- RNN Layer[ {self._model_type} ] -----------------")
        print(self._seq_RNN)

        print("\n----------------- FNN Layer -----------------")
        print(self._seq_FNN)

        print("----------------- Optimizer -----------------")
        print(self._optimizer)

        print("\n===============================================\n")

        pass


    def _train_by_batch(self, x_input, y_target):
        self._optimizer.zero_grad()
        y_pred, _, _ = self._calc_forward(x_input)

        ## squeeze the pred one-hot vec

        y_pred_t = torch.reshape(y_pred, [y_pred.size()[0]*y_pred.size()[1], self._term_size])
        y_target_t = torch.reshape(y_target, [y_target.size()[0]*y_pred.size()[1], ])

        """
        PYTORCH cross_entropy function, Target should be 1D, Pred should be argmax result 2D vector  
        need do TRANSFORM pred and target vector.
        pred, shape(batch_size * seq_num, _classification) classification as term_size
        target, shape(batch_size * seq_num,)
        """
        losses = self._loss_f(y_pred_t, y_target_t)

        _loss = losses.mean(dtype=torch.float32)

        y_pred_1 = torch.argmax(y_pred, dim=2, keepdim=False)  # argmax remove the last dimension

        _acc = torch.where(torch.eq(y_pred_1, y_target),
                           torch.full_like(y_target, 1.0, dtype=torch.float32),
                           torch.full_like(y_target, 0.0, dtype=torch.float32))
        _acc = torch.mean(_acc)

        losses.backward()
        self._optimizer.step()

        return _loss.detach().numpy(), _acc.detach().numpy()

    def Train(self, x_batch, y_batch):
        _begin_t = time.time()
        print("\t\t\t============= MODEL TRAINING BEGIN =============")
        batch_num = len(x_batch)

        for i in range(batch_num):
            x_data = x_batch[i]
            y_data = y_batch[i]

            if torch.is_tensor(y_data) is False:
                y_data = torch.from_numpy(y_data)
            # y_data_1_hot = nn.functional.one_hot(y_data, self._term_size)

            loss, acc = self._train_by_batch(x_data, y_data)
            if i % 10 == 0:
                print('batch', i,
                      ': loss ', loss, '; accuracy ', acc)
            if acc > 0.8:
                print('\t\t\tIteration Terminated!!!\n',
                      '; accuracy', acc.numpy())
                break
        _f_chpt_file = self.SaveModel()
        print("\t\t\t============= MODEL TRAINING FINISHED =============")

        return loss, acc, _f_chpt_file

    def ResetState(self):
        self._hn = None
        self._cn = None


    def _calc_forward(self, x_input, h_0=None, c_0=None):
        if torch.is_tensor(x_input) is False:
            x_input = torch.from_numpy(x_input)
        h_e = self._seq_Embedding(x_input)
        # print("embedding output size is: ", h_e.size())

        _RNN_Module = self._seq_RNN[0]
        h_r = None
        h_hn = None
        h_cn = None
        if self._model_type == 'SimpleRNN' or \
            self._model_type == 'GRU':
            if h_0 is None:
                h_r, h_hn = _RNN_Module(h_e)
            else:
                h_r, h_hn = _RNN_Module(h_e, h_0)
        else:  # LSTM
            if h_0 is None:
                h_r, (h_hn, h_cn) = _RNN_Module(h_e)
            else:
                h_r, (h_hn, h_cn) = _RNN_Module(h_e, (h_0, c_0))

        # print("rnn output size is: ", h_r[0].size())

        out_logits = self._seq_FNN(h_r)
        # print("fnn output size is: ", out_logits.size())

        return out_logits, h_hn, h_cn

    def __call__(self, x_input):
        out, self._hn, self._cn = self._calc_forward(x_input, h_0=self._hn, c_0=self._cn)
        r_val = torch.argmax(out, dim=2, keepdim=False)
        return r_val


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
            inp_val = torch.unsqueeze(torch.Tensor([inp_x]), dim=0)
            text_generated = [ds.idx_to_word(inp_x)]
            model.ResetState()
            for i in range(num_generate):
                pred_y = model(inp_val.long())
                pred_y = torch.squeeze(pred_y, dim=0).detach().numpy()[0]
                text_generated.append(ds.idx_to_word(pred_y))
                inp_val = torch.unsqueeze(torch.Tensor(list(map(ds.word_to_idx, text_generated))), dim=0)
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
    print("\n\t\tChapter6. Recurrent Neural Network, RNN exercise (PyTorch) ")
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
    model_type = 'GRU'

    while b_loop:
        try:
            step = int(input("please input step:"))
            if step == 1:
                ds = peom_process_util.poem_dataset()
                _, _, terms = ds.load_dataset(r'.\data\poems.dat', b_shuffle=True)
                pass
            elif step == 2:  # train model
                x_batch, y_batch = ds.generate_batch(batch_num=100)
                if myModel is None:
                    myModel = RNN_TORCH(term_size=len(terms),
                                        batch_size=len(x_batch[0]),
                                        model_type=model_type)
                else:
                    with open(r".\model\model.json", mode='r+') as f:
                        _config = json.load(f)
                        _latest_ckpt_file = _config["PYTORCH"]
                        f.close()
                    myModel.LoadModel(_latest_ckpt_file)
                    myModel.ResetState()

                for epoch in range(10):
                    print('----------------- epoch ', epoch, ' -----------------------')
                    myModel.ResetState()
                    _loss, _acc, _latest_ckpt_file = myModel.Train(x_batch, y_batch)
                    ds.shuffle_dataset()
                    x_batch, y_batch = ds.generate_batch(batch_num=100)
                    print('loss ', _loss, '; accuracy ', _acc)
                    print('----------------- epoch ', epoch, ' -----------------------\n')

                #loss, _acc, _latest_ckpt_file = myModel.Train(x_batch, y_batch)
                with open(r".\model\model.json", mode='r+', encoding="utf-8") as f:
                    _config = json.load(f)
                    _config["PYTORCH"] = _latest_ckpt_file
                    f.seek(0, 0)
                    json.dump(_config, f)
                    f.close()
                pass
            elif step == 3:  # predict using trained model
                """
                input shape is different, it should reload model
                """
                myModel = RNN_TORCH(term_size=len(terms),
                                    batch_size=1,
                                    model_type=model_type)
                with open(r".\model\model.json", mode='r+') as f:
                    _config = json.load(f)
                    _latest_ckpt_file = _config["PYTORCH"]
                    f.close()
                myModel.LoadModel(_latest_ckpt_file)
                myModel.ResetState()
                gen_sentence(myModel, ds)
                pass
            else:
                b_loop = False
        except ValueError as err:
            print(err)
            exit(code=-1)
    pass

