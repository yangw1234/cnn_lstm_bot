import cv2
import os
import re
import argparse
import time

import keras
import numpy as np
from keras.engine.saving import model_from_json
from keras.optimizers import Adam, sgd

from agents.ramdom import RandomAgent
from direct_keys import *
from display_controller import get_controller_image
from env import FIFAEnv
from get_keys import key_check
from human_detector import HumanDetector
from parameters import STACK_NUM
from predictor import MovementPredictor, ScorePredictor
# from utils import label_map_util
# from utils import visualization_utils as vis_util



def movement_net():
    model = keras_CNN(input_shape=[1024 * 1, 13, 13], net_name="movement", filter_num=[20, 20, 20])
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics = ['accuracy'])
    return model

def action_net():
    model = keras_CNN(input_shape=[1024 * 1, 13, 13], net_name="action", filter_num=[20, 20, 20])
    model.compile(optimizer=sgd(lr=0.0001), loss='mse', metrics = ['mse'])
    return model

def keras_CNN(input_shape,
        net_name,
        filter_num,
        keep_prob=1.0):
    #create model
    model = keras.Sequential()

    # model.add(keras.layers.Reshape([1024, 13, 13]))

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
        model.add(keras.layers.Dense(5, activation='softmax', name=net_name + '_fc2'))

    else:
        model.add(keras.layers.Dense(2, name=net_name + '_fc2'))

    return model


def moving_average_diff(a, n=20):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def load_model():
    # load json and create model
    json_file = open('D:\sources\FIFA\Shooting_Brozen_RL\model_epoch1000\model_action.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("D:\sources\FIFA\Shooting_Brozen_RL\model_epoch1000\model_action.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='mse', optimizer='sgd')
    return loaded_model




parser = argparse.ArgumentParser()
# parser.add_argument('model_path', help="Path where the model is stored")
# parser.add_argument('img_path', help="Path where the images are stored")
# parser.add_argument('output_path',  help="Path to store the detection results")
parser.add_argument('--mode', type=str, choices=['deep', 'shallow', 'full'], default='deep')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--train_movement', type=bool, default=True)
parser.add_argument('--train_score', type=bool, default=False)
parser.add_argument('--keep_prob', type=float, default=1.0)

parser.add_argument('--batch_size', type=int, default=256)


prev_movement = []
def take_action(movement_index, action_index):
    """Send action to the game."""
    global prev_movement
    # Movements
    movement_custom_b = [[], [W], [S], [A], [D]]

    action = [[], [spacebar]]

    print('movement: ' + str(movement_index) + ' and action: ' + str(action_index))

    move = movement_custom_b[movement_index]
    act = action[action_index]

    if prev_movement != move:
        for index in prev_movement:
            ReleaseKey(index)
        for index in move:
            PressKey(index)

        prev_movement = move

    # for index in movement_custom_b[movement_index]:
    #     PressKey(index)
    if True:
        for index in action[action_index]:
            PressKey(index)
        time.sleep(0.18)
        # for index in movement_custom_b[movement_index]:
        # ReleaseKey(index)
        for index in action[action_index]:
            ReleaseKey(index)
    # time.sleep(0.5)

def check_pause(paused):
    """Pause/unpause the game using 'p'. Quit the game using 'q'."""
    keys = key_check()
    is_quit = False
    if 'P' in keys:
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            cv2.destroyAllWindows()
            time.sleep(1)
    elif 'O' in keys:
        print('Quitting!')
        cv2.destroyAllWindows()
        is_quit = True

    return is_quit, paused

def init_input_window(detector):
    if args.mode == 'deep':
        shape = (STACK_NUM, 1024, 13, 13)
    elif args.mode == 'shallow':
        shape = (STACK_NUM, 128, 52, 52)
    else:
        shape = (STACK_NUM, 2)
    input_window = np.zeros(shape=shape)

    for i in range(STACK_NUM):
        take_action(4, 0)
        # rep = detector.forward()
        input_window[i, :] = detector.forward()
    return input_window



if __name__ == "__main__":
    args = parser.parse_args()


    print("init agent")
    agent = RandomAgent()

    print("init FIFAEnv")
    env = FIFAEnv()
    detector = HumanDetector(mode="deep")
    # score_predictor = ScorePredictor(mode="deep",
    #                                  epoch=0,
    #                                  lr=0,
    #                                  keep_prob=1.0,
    #                                  batch_size=0)



    ref_center = cv2.imread("reference.jpg")


    start = True
    paused = True
    print('Game paused, press p to start.')
    while paused:
        is_quit, paused = check_pause(paused)

    screen = env.reset()
    detector.forward(screen)
    model_action = load_model()
    while True:
        if not paused:
            # if start:
            #     input_window = init_input_window(detector)
            #     start = False
            cur_time = time.time()

            movement_index, action_index = agent.take_action(screen)


            screen, reward, done = env.act(movement_index, action_index)
            # cv2.imwrite("/score_data/score_prediction_%s.jpg" % score, screen)

            print(reward)

            if done:
                screen = env.reset()
        # Check keys for pause / unpause / quit
        is_quit, paused = check_pause(paused)
        if is_quit:
            break


