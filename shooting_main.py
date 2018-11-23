import cv2
import os
import re
import argparse
import time
import numpy as np

from detectors import RandomDetector
from direct_keys import *
from display_controller import get_controller_image
from get_keys import key_check
from human_detector import HumanDetector
from parameters import STACK_NUM
from predictor import MovementPredictor, ScorePredictor
from utils import label_map_util
from utils import visualization_utils as vis_util


parser = argparse.ArgumentParser()
# parser.add_argument('model_path', help="Path where the model is stored")
# parser.add_argument('img_path', help="Path where the images are stored")
# parser.add_argument('output_path',  help="Path to store the detection results")
parser.add_argument('--mode', type=str, choices=['deep', 'shallow', 'full'], default='shallow')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--train_movement', type=bool, default=False)
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
    movement_predictor = MovementPredictor(mode=args.mode,
                                           epoch=args.epoch,
                                           lr=args.lr,
                                           keep_prob=args.keep_prob,
                                           batch_size=args.batch_size)
    score_predictor = ScorePredictor(mode=args.mode,
                                     epoch=args.epoch,
                                     lr=args.lr,
                                     keep_prob=args.keep_prob,
                                     batch_size=args.batch_size)
    if args.train_movement:
        movement_predictor.train()

    elif args.train_score:
        score_predictor.train()

    else:
        detector = HumanDetector(mode=args.mode)
        start = True

        play = 1

        last_time = time.time()
        frames_count = 0
        detection_time = []
        prediction_time = []
        paused = True
        print('Game paused, press p to start.')
        while True:
            if not paused:
                # time.sleep(0.1)
                # if start:
                #     input_window = init_input_window(detector)
                #     start = False
                cur_time = time.time()
                rep = detector.forward()
                cur_detection_time = time.time() - cur_time
                detection_time.append(cur_detection_time)
                # print("detection time is {}s".format(cur_detection_time))
                # print(rep.shape)
                # input_window[:-1, :] = input_window[1:, :]
                # input_window[-1, :] = rep
                cur_time = time.time()
                input_window = np.expand_dims(rep, axis=0)

                movement_index = movement_predictor.inference(input_window)
                score = score_predictor.inference(input_window)
                cur_prediction_time = time.time() - cur_time
                prediction_time.append(cur_prediction_time)

                # print("prediction time is {}s".format(cur_prediction_time))
                print("score is %s" % score)


                if score > 722:
                    action_index = 1
                else:
                    action_index = 0

                if play == 1:
                    # movement_index = 0
                    # action_index = 0
                    take_action(movement_index, action_index)
                    if action_index == 1:
                        time.sleep(2)
                        # input_window = init_input_window(detector)

                current_time = time.time()
                if current_time - last_time >= 1:
                    # print('{} frames per second'.format(frames_count))
                    last_time = current_time
                    frames_count = 0
                else:
                    frames_count = frames_count + 1

            # Check keys for pause / unpause / quit
            is_quit, paused = check_pause(paused)
            if is_quit:
                print("mean detection time is:", np.mean(detection_time))
                print("mean prediction time is:", np.mean(prediction_time))
                break

