from model import LSTM_model
from parameters import STEPS_OF_HISTORY, MOVEMENT_NUM, ACTION_NUM, TRAIN_DATA_PATH, SCORE_NET_TRAIN_DATA_PATH
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.net import Net
import time
from sklearn.manifold import TSNE


class ScorePredictor(object):
    """LSTM movement and action predictor, based on BigDL Model API."""

    def __init__(self):
        self.score_model_dir = 'bigdl_tf_model/score_model/'
        self.score_net = None

    def _init_model(self, new_model=False):
        """Initialize action network and movement network, load weight from local file."""
        if new_model:
            # Build model graph
            self.score_net = LSTM_model(ACTION_NUM, STEPS_OF_HISTORY)
        else:
            # Load model from file
            idx = self.get_newest_model_idx(self.score_model_dir)
            self.score_net = Net.load(os.path.join(self.score_model_dir, "model." + str(idx)))

    def get_newest_model_idx(self, model_dir):
        model_names = os.listdir(model_dir)
        idx = []
        for name in model_names:
            idx.append(int(name.split(".")[1]))

        return max(idx)

    def predict(self, input_window):
        """Feed-forward two LSTM models to get action and movement predictions."""
        self._init_model()
        action_res = self.score_net.predict_local(input_window)

        return np.squeeze(action_res)

    def train(self, batch_size=100, max_epoch=250, from_scratch=True):
        """Train the re-implemented LSTM prediction model."""
        if from_scratch:
            self._init_model(True)
        else:
            self._init_model()

        trainX = []
        trainY = []
        for filename in os.listdir(SCORE_NET_TRAIN_DATA_PATH):
            filename = SCORE_NET_TRAIN_DATA_PATH + filename
            d = np.load(filename)
            data = list(d)
            # print(d[-1][2])
            # print(d.shape)
            for X, Y in data:
                trainX.append(X)
                trainY.append(Y)

        trainX = np.asarray(trainX)
        trainY = np.asarray(trainY)

        # Train movement network
        # self.movement_net.set_checkpoint("../../FIFA_ckpt_movement/", )
        self.score_net.set_tensorboard("../../logs_score", "lstm_bot_%s" % (time.time()))
        self.score_net.compile(loss='mse', optimizer=Adam(learningrate=1e-3),
                                  metrics=['accuracy'])
        self.score_net.fit(trainX, trainY, batch_size=batch_size, nb_epoch=max_epoch)
        idx = self.get_newest_model_idx(self.score_model_dir) + 1
        self.score_net.saveModel(os.path.join(self.score_model_dir, "model." + str(idx)))