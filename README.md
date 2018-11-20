# A CNN + LSTM bot for full game mode in FIFA 18

## Introduction

This work is based on [ChintanTrivedi/DeepGamingAI_FIFA](https://github.com/ChintanTrivedi/DeepGamingAI_FIFA), re-implemented using [BigDL](https://github.com/intel-analytics/BigDL) and [Analytics-Zoo](https://github.com/intel-analytics/analytics-zoo) from Intel-analytics group.

It uses a *MobileNet* to extract visual features from screen-shots of FIFA game.
Then the feature is stacked with previous features (to enable time series prediction) and fed into two *LSTM* networks.
One *LSTM* network predicts where to move the human, another predicts which action to take.
Then the signal is converted to keyboard signal and send to the game.

![CNN_LSTM_bot](../Doc_img/DL-based-framework.PNG "CNN_LSTM_bot")


## Development state

Code has been tested with the following specifications:
- Windows 10 Enterprise
- CUDA V9.1.85
- Python 3.6.6 from Anaconda
- Analytics-zoo 0.3.0 (Nightly 20180802.173847-26), pre-built with bigdl 0.6.0, and spark 2.2.0

TODO:
- [x] Training code for LSTM models
- [ ] Training code for MobilNet
- [ ] Inference speed test
- [ ] More training samples for different situations


## Run demo

### Preparation

1. Change the path to `spark-submit` executable in `run_zoo.bat`.

1. Open an Anaconda prompt, direct to this project root.

1. Find the Kick Off full game mode in FIFA 18: Play -> Kick Off

1. Choose your team and start the game. You may skip all animations by pressing `space`.

### Testing

1. Run `run_zoo.bat`.

1. When the game is ready, enter 'p' in the prompt and click the game again (to move the focus point back to the game).

1. Your player should be controlled by the bot now.


### Training

The code used to collect training data can be found in `create_lstm_training_data.py`.
Start the game and run this script (it doesn't depend on BigDL or Zoo so just with python is fine), the training data will be stored in `rnn` folder.

Uncomment the following lines in `zoo_main.py`:
```
# train_LSTM = 1

# if train_LSTM:
    # lstm_predictor = LSTMPredictor()
    # lstm_predictor.train()
```
Then the training will be started.

## Project structure

The main access point for BigDL/Zoo adaption is `zoo_main.py`.

### CNN feature extractor

Find more of this part in `zoo_cnn.py`.
The CNN used in this project is a MobileNet-SSD detection framework, originally provided by Tensorflow [object detection model](https://github.com/tensorflow/models/tree/master/research/object_detection).
The original author of `DeepGaming_FIFA` finetunes the MobileNet-SSD model with some FIFA training samples, see `ssd_mobilenet_training` folder in `backup_files`.

To use the network in BigDL/Zoo, it is wraped into TFNet model, reference [here](https://analytics-zoo.github.io/master/#APIGuide/PipelineAPI/net/).
Since TFNet doesn't feature model training, the reader will have to fine-tune the original Tensorflow model using their API and then convert to TFNet in order to fit this project.
See `DetectionNet > convert_from_tf` for model conversion.

### LSTM model

Find more of this part in `zoo_lstm.py`.
The LSTM model is written with BigDL Keras API, and trained with self-collected data.
To train the model, you'll need more data for supervised learning.
You'll need to run `create_lstm_training_data.py` as you play, and the logged data is stored in `rnn` folder.
See `LSTMPredictor > train` for training code.
