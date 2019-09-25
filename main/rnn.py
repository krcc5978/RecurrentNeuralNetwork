import numpy as np

from main.RNN.RNN_Config import RNN_Config
from main.RNN.utils import make_teacher_data
from main.const_value import *


def train():
    train_data, teacher_data = make_teacher_data(base_path, teacher_directory_list, timesteps)
    rnn_config = RNN_Config(train_data[0], hidden_neuron_number)
    rnn_config.trainning_start(train_data, teacher_data)


def predict():
    test_data, teacher_data = make_teacher_data(base_path, teacher_directory_list, timesteps)
    rnn_config = RNN_Config(test_data[0], hidden_neuron_number)
    rnn_config.load_weight(weight_path)

    # テストデータの件数分認証を行う
    for predict_data, label in zip(test_data, teacher_data):
        # 画像の読み込み、リサイズ

        # 認証結果をnumpy形式で取得
        result = np.array(rnn_config.recognition([predict_data])[0])

        # 正解データのインデックスとスコアの最大値の取得
        answer = np.array(label)

        print('-----------------------------------------------------------------')
        print('answer : {} '.format(str(answer)))
        print('result : {} '.format(str(result)))
        print('-----------------------------------------------------------------')


if __name__ == '__main__':
    """
    学習 → train
    認証 → predict
    """
    # train()
    predict()
