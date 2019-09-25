# RNN/LSTMの中間ユニットの数
hidden_neuron_number = 100

#時系列の長さ
timesteps = 50

#入力ユニットの数
input_size = 10

# バッチサイズ
batch_size = 32

# 入力データパス
base_path = 'D:\\data\\時系列\\'
teacher_directory_list = ['1']

# 重みファイル出力場所
weight_path = './logs/ep090-loss10.745-val_loss16.150.h5'

# 使用するCNNモデル
use_model_name = 'AlexNet'
