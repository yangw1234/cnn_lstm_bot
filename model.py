from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import Sequential


def LSTM_model(output_size, input_frame_size, output_activation=None):

    model = Sequential()
    model.add(Reshape((input_frame_size, 128), input_shape=(input_frame_size, 128), name='net_layer1'))
    model.add(LSTM(256, return_sequences=True, name='net_layer2'))
    model.add(Dropout(0.6, name='net_layer3'))
    model.add(LSTM(256, return_sequences=False, name='net_layer4'))
    model.add(Dropout(0.6, name='net_layer5'))
    model.add(Dense(output_size, activation=output_activation, name='net_layer6'))
