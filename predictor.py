import time

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
# from tensorflow.python.estimator import keras

from tflearn_model import CNN, DNN, keras_CNN, keras_DNN
from parameters import STACK_NUM, DATA_PATH, MOVEMENT_MODEL_PATH, SCORE_MODEL_PATH, RANDOM_DATA_PATH
import numpy as np
import os
import tensorflow as tf

class Predictor:
    def __init__(self, mode, epoch, lr, keep_prob, batch_size):
        self.net_name = None
        self.mode = mode
        self.epoch = epoch
        self.lr = lr
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.model_name = 'feature_mode-' + self.mode \
                          + '-epoch-' + str(self.epoch) \
                          + '-lr-' + str(float(self.lr)) \
                          + '-batch_size-' + str(self.batch_size) \
                          + '-stack_num-' + str(STACK_NUM) \
                          + '-keep_prob-' + str(keep_prob)

    def build_model(self):
        if self.mode == 'full':
            input_shape = [2 * STACK_NUM]
            model = keras_DNN(input_shape, self.net_name,
                        units_num=[32, 16],
                        keep_prob=self.keep_prob)


        elif self.mode == 'deep':
            input_shape = [1024 * STACK_NUM, 13, 13]
            model = keras_CNN(input_shape,
                        self.net_name,
                        filter_num=[20, 20, 20],
                        keep_prob=self.keep_prob)

        else:
            input_shape = [128 * STACK_NUM, 52, 52]
            model = keras_CNN(input_shape,
                              self.net_name,
                              filter_num=[16, 32, 32],
                              keep_prob=self.keep_prob)

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

    def __init__(self, mode, epoch, lr, keep_prob, batch_size):
        super(MovementPredictor, self).__init__(mode, epoch, lr, keep_prob, batch_size)
        self.net_name = 'movement'

        self.model_fullname = MOVEMENT_MODEL_PATH + self.model_name

        self.model_movement = self.build_model()

        self.logs_dir = "../keras_logs/movement/" + self.model_name + '_' + time.strftime("%Y%m%d-%H%M%S")

        self.loaded = False


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
        feature_path = DATA_PATH + self.mode + "_feature/"
        all_data_filename = feature_path + "all_data.npz"
        if os.path.isfile(all_data_filename):
            all_data = np.load(all_data_filename)
            print("Load data from saved files")
            X = all_data['X']
            Y_movement = all_data['Y_movement']

        else:
            X = []
            Y_movement = []

            names = os.listdir(feature_path)
            for filename in names:
                fullname = feature_path + filename
                data = np.load(fullname)
                trainX, trainY_movement = self.reshape_data(list(data))
                X.extend(trainX)
                Y_movement.extend(trainY_movement)

            X = np.asarray(X)
            Y_movement = np.asarray(Y_movement)
            np.savez(all_data_filename, X=X, Y_movement=Y_movement)

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
        tensorboard = TensorBoard(log_dir=self.logs_dir)

        checkpoint_path = self.logs_dir + "/weights-improvement-{epoch:02d}-{val_acc:.2f}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=False)

        self.model_movement.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy', metrics = ['accuracy'])

        self.model_movement.fit(X, Y_movement,
                           batch_size=self.batch_size,
                           epochs=self.epoch,
                           validation_split=0.1,
                           shuffle=True, callbacks=[tensorboard, checkpoint])

        self.model_movement.save(self.model_fullname + '.keras')

    def inference(self, input_window):
        if not self.loaded:
            if self.mode == 'full':
                restore_model = "../keras_logs/movement/" + \
                                "feature_mode-full-epoch-400-lr-0.0001-batch_size-256-stack_num-1-keep_prob-1.0_20181122-230634/" \
                                + "weights-improvement-400-0.68-0.87.hdf5"

            elif self.mode == 'deep':
                restore_model = "../keras_logs/" + \
                                "feature_mode-deep-epoch-20-lr-0.0001-batch_size-256-stack_num1_1542790347.5969057/" \
                                + "weights-improvement-20-0.75-0.85.hdf5"

            else:
                restore_model = self.logs_dir + ".hdf5"

            self.model_movement.load_weights(restore_model)
            self.model_movement.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy',
                                        metrics=['accuracy'])
            self.loaded = True
        movement = self.model_movement.predict(input_window)
        movement_index = np.argmax(movement)
        return movement_index

class ScorePredictor(Predictor):

    def __init__(self, mode, epoch, lr, keep_prob, batch_size):
        super(ScorePredictor, self).__init__(mode, epoch, lr, keep_prob, batch_size)
        self.net_name = 'score'
        self.model_fullname = SCORE_MODEL_PATH + self.model_name
        self.model_score = self.build_model()
        self.logs_dir = "../keras_logs/score/" + self.model_name + '_' + time.strftime("%Y%m%d-%H%M%S")
        self.max_data = 1251
        self.min_data = 0

        self.loaded = False


    def reshape_data(self, data):
        window = data[-STACK_NUM : ]

        sampleX = []
        for row in window:
            sampleX.append(row[0])

        trainX = np.concatenate(np.array(sampleX))

        # print("trainX shape:", trainX.shape)

        return trainX

    def _load_data(self, path, X, Y_Score):
        feature_path = path + self.mode + "_feature/"
        for filename in os.listdir(feature_path):
            if filename=='all_data.npz':
                continue
            fullname = feature_path + filename
            data = np.load(fullname)
            score = int(filename.split("_")[3][:-4])
            trainX = self.reshape_data(data)
            X.append(trainX)
            Y_Score.append(score)
        print(len(X), len(Y_Score))
        # return X, Y_Score

    def scale(self, data):
        self.min_data = np.min(data)
        self.max_data = np.max(data)
        print(self.min_data, self.max_data)
        return (data - self.min_data) / (self.max_data - self.min_data)

    def centralize(self, data):
        return data - 500

    def load_data(self):
        feature_path = RANDOM_DATA_PATH + self.mode + "_feature/"
        all_data_filename = feature_path + "all_data.npz"
        if os.path.isfile(all_data_filename):
        # if False:
            all_data = np.load(all_data_filename)
            print("Load data from saved files")
            X = all_data['X']
            Y_Score = all_data['Y_Score']

        else:
            X = []
            Y_Score = []
            self._load_data(DATA_PATH, X, Y_Score)
            # print("result", len(X), len(Y_Score))
            self._load_data(RANDOM_DATA_PATH, X, Y_Score)
            # print("result", len(X), len(Y_Score))

            X = np.asarray(X)
            Y_Score = np.asarray(Y_Score)
            print("X shape:", X.shape)
            print("Y_score.shape:", Y_Score.shape)
            print(np.mean(Y_Score))
            np.savez(all_data_filename, X=X, Y_Score=Y_Score)

        return X, Y_Score

    def train(self):
        # with tf.Graph().as_default():
        X, Y_Score = self.load_data()
        Y_Score = self.scale(Y_Score)

        #callback
        tensorboard = TensorBoard(log_dir=self.logs_dir)

        checkpoint_path = self.logs_dir + "/weights-improvement-{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False)

        self.model_score.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error', metrics=['mae'])

        self.model_score.fit(X, Y_Score,
                                batch_size=self.batch_size,
                                epochs=self.epoch,
                                validation_split=0.1,
                                shuffle=True, callbacks=[tensorboard, checkpoint])

        self.model_score.save(self.model_fullname + '.keras')

    def inference(self, input_window):

        if self.loaded:
            if self.mode == 'full':
                restore_model = "../keras_logs/score/" + \
                                "feature_mode-full-epoch-500-lr-0.0005-batch_size-256-stack_num-1-keep_prob-1.0_20181122-232340/" + \
                                "weights-improvement-500-0.07-0.06.hdf5"

            elif self.mode == 'deep':
                restore_model = "../keras_logs/score/" + \
                                "feature_mode-deep-epoch-400-lr-0.0001-batch_size-64-stack_num-1-keep_prob-0.8_20181122-144945/" \
                                + "weights-improvement-400-0.07-0.06.hdf5"

            else:
                restore_model = self.logs_dir + ".hdf5"

            self.model_score.load_weights(restore_model)
            self.model_score.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error', metrics=['mae'])
            self.loaded = True

        scaled_score = self.model_score.predict(input_window)

        score = scaled_score * (self.max_data - self.min_data) + self.min_data
        # metric = self.model_score.evaluate()p
        return score