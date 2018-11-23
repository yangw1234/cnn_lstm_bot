import tflearn
from keras import metrics
from keras.layers import MaxPooling2D, regularizers
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

from parameters import MOVEMENT_NUM


def CNN(input_shape,
        net_name,
        filter_num,
        keep_prob=0.5,
        lr=0.001,
        loss='categorical_crossentropy',
        metric='accuracy',
        model_name=None):
    network = input_data(shape=input_shape, name=net_name + '_input_layer')
    network = conv_2d(network, filter_num[0], 3, activation='relu', name=net_name + '_conv1')
    # network = max_pool_2d(network, 2, name=net_name + '_max_pool1')
    network = conv_2d(network, filter_num[1], 3, activation='relu', name=net_name + '_conv2')
    # network = max_pool_2d(network, 2, name=net_name + '_max_pool2')
    network = fully_connected(network, filter_num[2], activation='relu', name=net_name + '_fc1')
    network = dropout(network, keep_prob, name=net_name + '_dp1')
    network = fully_connected(network, MOVEMENT_NUM, activation='softmax', name=net_name + '_fc2')
    network = regression(network, optimizer='adam',
                         loss=loss,
                         metric=metric,
                         learning_rate=lr)

    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path=model_name + 'CNN' + '/')
    return model

def DNN(input_shape,
        net_name,
        units_num,
        keep_prob=0.5,
        lr=0.001,
        loss='categorical_crossentropy',
        metric='accuracy',
        model_name=None):
    input_layer = input_data(shape=input_shape, name=net_name + '_input_layer')
    dense1 = tflearn.fully_connected(input_layer, units_num[0], activation='relu',
                                     name=net_name + '_fc1')
                                    # regularizer='L2', weight_decay=0.001, name=net_name + '_fc1')
    dropout1 = tflearn.dropout(dense1, keep_prob, name=net_name + '_dp1')
    dense2 = tflearn.fully_connected(dropout1, units_num[1], activation='relu',
                                     name=net_name + '_fc2')
                                     # regularizer='L2', weight_decay=0.001, name=net_name + '_fc2')
    dropout2 = tflearn.dropout(dense2, keep_prob, name=net_name + '_dp2')
    softmax = tflearn.fully_connected(dropout2, MOVEMENT_NUM, activation='softmax', name=net_name + '_fc3')

    network = regression(softmax, optimizer='adam',
                         loss=loss,
                         metric=metric,
                         learning_rate=lr)
    # Regression using SGD with learning rate decay and Top-3 accuracy
    # sgd = tflearn.SGD(learning_rate=lr, lr_decay=0.96, decay_step=1000)
    # top_k = tflearn.metrics.Top_k(2)
    # net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
    #                          loss='categorical_crossentropy')
    model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path=model_name +'_DNN_' + '/')
    return model

def resnet(input_shape):
    net = tflearn.input_data(shape=[None, 28, 28, 1])
    net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
    # Residual blocks
    net = tflearn.residual_bottleneck(net, 3, 16, 64)
    net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 32, 128)
    net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 64, 256)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.1)
    return net


def LSTM(input_layer):
    net = tflearn.lstm(input_layer, n_units=256, return_seq=True, name='net1_layer2')
    net = tflearn.dropout(net, 0.6, name='net1_layer3')
    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net1_layer4')
    net = tflearn.dropout(net, 0.6, name='net1_layer5')
    net = tflearn.fully_connected(net, 5, activation='softmax', name='net1_layer6')
    net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.001,
                             name='net1_layer7')
    return net

import keras
def keras_CNN(input_shape,
        net_name,
        filter_num,
        keep_prob=0.5):
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