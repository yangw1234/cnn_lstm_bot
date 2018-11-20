import time

from keras.callbacks import TensorBoard

from tflearn_model import CNN, DNN, keras_CNN
from parameters import STACK_NUM, DATA_PATH, MOVEMENT_MODEL_PATH, SCORE_MODEL_PATH, RANDOM_DATA_PATH
import numpy as np
import os
import tensorflow as tf

class Predictor:
    def __init__(self, mode, epoch, lr, batch_size):
        self.net_name = None
        self.loss = None
        self.metric = None
        self.mode = mode
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.model_fullname = None

    def build_model(self, load_weights=True):
        if self.mode == 'full':
            input_shape = [None, 2 * STACK_NUM]
            model = DNN(input_shape, self.net_name,
                        filter_size=[32, 64],
                        keep_prob=1.0,
                        lr=self.lr,
                        loss=self.loss,
                        metric=self.metric,
                        model_name=self.model_fullname)


        elif self.mode == 'deep':
            input_shape = [1024 * STACK_NUM, 13, 13]
            model = keras_CNN(input_shape,
                        self.net_name,
                        filter_size=[20, 20, 20],
                        keep_prob=1.0, lr=self.lr,
                        loss=self.loss,
                        model_name=self.model_fullname)

        else:
            input_shape = [None, 128 * STACK_NUM, 52, 52]
            model = CNN(input_shape,
                        self.net_name,
                        filter_size=[100, 20, 10],
                        keep_prob=1.0,
                        lr=self.lr,
                        loss=self.loss,
                        model_name=self.model_fullname)
        # if load_weights:
        #     model.load(self.model_fullname + '.tflearn')

        return model

    def reshape_data(self, data):
        pass

    def load_data(self):
        pass

    def train(self):
        pass

    def inference(self, input_window):
        pass

class MovementPredictor(Predictor):
    """ Movement predictor, based on tflearn"""

    def __init__(self, mode, epoch, lr, batch_size):
        super(MovementPredictor, self).__init__(mode, epoch, lr, batch_size)
        self.net_name = 'movement'
        self.loss = 'categorical_crossentropy'
        self.metric = 'accuracy'
        self.model_name = 'feature_mode-' + self.mode \
                        + '_epoch-' + str(self.epoch) \
                        + '_lr-' + str(float(self.lr)) \
                        + '_batch_size-' + str(self.batch_size)
        self.model_fullname = MOVEMENT_MODEL_PATH + self.model_name

        self.model_movement = self.build_model(True)

    def reshape_data(self, data):
        trainX = []
        trainY_movement = []

        for i in range(0, len(data) - STACK_NUM + 1):
            window = data[i:i + STACK_NUM]

            sampleX = []
            for row in window:
                sampleX.append(row[0])
            sampleY_movement = np.array(window[-1][1]).reshape(-1)

            trainX.append(np.concatenate(np.array(sampleX)))
            trainY_movement.append(sampleY_movement)

        print("trainX shape:", np.array(trainX).shape)
        print("trainY_movement shape:", np.array(trainY_movement).shape)

        return trainX, list(trainY_movement)

    def load_data(self):
        X = []
        Y_movement = []
        feature_path = DATA_PATH + self.mode + "_feature/"

        names = os.listdir(feature_path)
        for filename in names:
            fullname = feature_path + filename
            data = np.load(fullname)
            trainX, trainY_movement = self.reshape_data(list(data))
            X.extend(trainX)
            Y_movement.extend(trainY_movement)

        X = np.asarray(X)
        Y_movement = np.asarray(Y_movement)
        return X, Y_movement

    def train(self):
        # with tf.Graph().as_default():
            # model_movement = self.build_model()
        X, Y_movement = self.load_data()
        # self.model_movement.fit(X, Y_movement,
        #                    batch_size=self.batch_size,
        #                    n_epoch=self.epoch,
        #                    validation_set=0.1,
        #                    show_metric=True,
        #                    shuffle=True,
        #                    run_id=self.model_name + str(time.time()))
        tensorboard = TensorBoard(log_dir="../keras_logs/" + self.model_name + "_{}".format(time.time()))
        self.model_movement.fit(X, Y_movement,
                           batch_size=self.batch_size,
                           epochs=self.epoch,
                           validation_split=0.1,
                           shuffle=True, callbacks=[tensorboard])

        self.model_movement.save(self.model_fullname + '.keras')

    def inference(self, input_window):
        self.model_movement.load(self.model_fullname + 'tflearn')
        self.model_movement.predict(input_window)

class ScorePredictor(Predictor):

    def __init__(self, mode, epoch, lr, batch_size):
        super(ScorePredictor, self).__init__(mode, epoch, lr, batch_size)
        self.net_name = 'score'
        self.loss = 'mean_square'
        self.metric = 'R2'
        self.model_name = 'feature_mode-' + self.mode \
                        + '_epoch-' + str(self.epoch) \
                        + '_lr-' + str(float(self.lr)) \
                        + '_batch_size-' + str(self.batch_size)
        self.model_fullname = SCORE_MODEL_PATH + self.model_name
        self.model_score = self.build_model()

    def reshape_data(self, data):
        window = data[-STACK_NUM : ]

        sampleX = []
        for row in window:
            sampleX.append(row[0])

        trainX = np.concatenate(np.array(sampleX))

        print("trainX shape:", trainX.shape)

        return trainX

    def _load_data(self, path, X, Y_Score):
        feature_path = path + self.mode + "_feature/"
        for filename in os.listdir(feature_path):
            fullname = feature_path + filename
            data = np.load(fullname)
            score = int(filename.split("_")[3][:-4])
            trainX = self.reshape_data(data)
            X.append(trainX)
            Y_Score.append(score)
        # return X, Y_Score

    def load_data(self):
        X = []
        Y_Score = []
        self._load_data(DATA_PATH, X, Y_Score)
        self._load_data(RANDOM_DATA_PATH, X, Y_Score)

        X = np.asarray(X)
        Y_Score = np.asarray(Y_Score)
        print("X shape:", X.shape)
        print("Y_score.shape:", Y_Score.shape)

        return X, Y_Score

    def train(self):
        # with tf.Graph().as_default():
        X, Y_score = self.load_data()
        self.model_score.fit(X, Y_score,
                        batch_size=self.batch_size,
                        n_epoch=self.epoch,
                        validation_set=0.1,
                        run_id=self.model_name)

        self.model_score.save(self.model_fullname + '.tflearn')

    def inference(self, input_window):
        self.model_score.load(self.model_fullname + 'tflearn')
        self.model_score.predict(input_window)