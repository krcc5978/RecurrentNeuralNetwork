from keras.layers import SimpleRNN, Dense, Activation
from keras.models import Sequential


class RNN:

    def __init__(self, input_data_list, hidden_neuron_number):

        self.model = Sequential()
        self.model.add(SimpleRNN(hidden_neuron_number, batch_input_shape=(None, len(input_data_list), len(input_data_list[0])), return_sequences=False))
        self.model.add(Dense(100))
        self.model.add(Activation("linear"))

        self.model.add(Dense(len(input_data_list[0])))

    def get_model(self):
        return self.model
