import numpy as np
from keras import optimizers
from main.RNN.callback_method import model_checkpoint


class RNN_Config:

    def __init__(self, input_data_list, hidden_neuron_number, use_model_name='RNN'):
        if use_model_name == 'RNN':
            from main.RNN.models.RNN_models import RNN as use_model
        elif use_model_name == 'LSTM':
            from main.RNN.models.LSTM_models import LSTM as use_model
        else:
            from main.RNN.models.load_model import load_model as use_model

        self.model = use_model(input_data_list, hidden_neuron_number).get_model()
        self.model.summary()

    def load_weight(self, weight_path):
        """
        :param weight_path: 使用する重みファイルのパス
        :return:
        """
        self.model.load_weights(weight_path)

    def recognition(self, sequence_data, batch_size=1):
        """
        :param sequence_data: 時系列データ
        :param batch_size: バッチサイズ
        :return: 認証結果
        """
        test_seq_array = np.array(sequence_data)
        return self.model.predict(test_seq_array, batch_size=batch_size)

    def trainning_start(self, train_data, teacher_data, model_save_path='./'):
        # オプティマイザーの設定
        optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-4)

        # 使用するモデルのコンパイル
        self.model.compile(optimizer=optimizer,
                           loss='mape',
                           metrics=['accuracy'])

        # モデルの保存
        model_json_str = self.model.to_json()
        open(model_save_path + 'face_model.json', 'w').write(model_json_str)

        # コールバック関数の宣言
        checkpoint = model_checkpoint('./logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                      'val_loss',
                                      True,
                                      True,
                                      3)

        self.model.fit(train_data, teacher_data, batch_size=10, epochs=100, validation_split=0.1, callbacks=[checkpoint])
        # 結果の出力
        self.model.save_weights('./main/face_model_weights.h5')