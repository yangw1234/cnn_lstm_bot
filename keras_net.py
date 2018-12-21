from keras import metrics
from keras.layers import MaxPooling2D, regularizers

from parameters import MOVEMENT_NUM

import keras
def keras_CNN(input_shape,
        net_name,
        filter_num,
        keep_prob=1.0):
    #create model
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(filter_num[0], 3, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.01),
                                  name=net_name + 'conv1',
                                  input_shape=input_shape))
    model.add(keras.layers.Conv2D(filter_num[1], 3, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.01),
                                  name=net_name + 'conv2'))
    model.add(keras.layers.Dropout(keep_prob, name=net_name + '_dp1'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(filter_num[2], activation='relu',
                                 kernel_regularizer=regularizers.l2(0.01),
                                 name=net_name + '_fc1'))
    model.add(keras.layers.Dropout(keep_prob, name=net_name + '_dp2'))

    if net_name == 'movement':
        model.add(keras.layers.Dense(MOVEMENT_NUM, activation='softmax', name=net_name + '_fc2'))

    else:
        model.add(keras.layers.Dense(1, activation='sigmoid', name=net_name + '_fc2'))

    return model

def keras_DNN(input_shape,
        net_name,
        units_num,
        keep_prob=0.5,
        lr=0.001):
    #create model
    model = keras.Sequential()
    model.add(keras.layers.Dense(units_num[0], activation='relu',
                                 kernel_regularizer=regularizers.l2(0.01),
                                 name=net_name + '_fc1',
                                 input_shape=input_shape))
    model.add(keras.layers.Dropout(keep_prob, name=net_name + '_dp1'))

    model.add(keras.layers.Dense(units_num[1], activation='relu',
                                 kernel_regularizer=regularizers.l2(0.01),
                                 name=net_name + '_fc2'))


    model.add(keras.layers.Dropout(keep_prob, name=net_name + '_dp2'))

    if net_name == 'movement':
        model.add(keras.layers.Dense(MOVEMENT_NUM, activation='softmax', name=net_name + '_fc3'))
    else:
        model.add(keras.layers.Dense(1, activation='sigmoid', name=net_name + '_fc3'))

    return model