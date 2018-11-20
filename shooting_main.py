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
from predictor import MovementPredictor, ScorePredictor
from utils import label_map_util
from utils import visualization_utils as vis_util


parser = argparse.ArgumentParser()
# parser.add_argument('model_path', help="Path where the model is stored")
# parser.add_argument('img_path', help="Path where the images are stored")
# parser.add_argument('output_path',  help="Path to store the detection results")
parser.add_argument('--mode', type=str, choices=['deep', 'shallow', 'full'], default='shallow')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--train_movement', type=bool, default=True)
parser.add_argument('--train_score', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()

prev_movement = []
def take_action(movement_index, action_index):
    """Send action to the game."""
    global prev_movement
    # Movements
    movement_custom_b = [[], [W], [S], [A], [D]]
    # movement_custom_b = [[U, E], [J, E], [H, E], [L, E], []]

    # Actions
    # Shoot/volley/header, short pass/header, through bal, lob pass/cross/header
    # stands for B, A, Y, X in joystick
    # action = [[], [spacebar], [L], [K], [J]]
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

if __name__ == "__main__":

    if args.train_movement:
        movement_predictor = MovementPredictor(mode=args.mode, epoch=args.epoch, lr=args.lr, batch_size=args.batch_size)
        movement_predictor.train()

    elif args.score_movement:
        score_predictor = ScorePredictor(mode=args.mode, epoch=args.epoch, lr=args.lr, batch_size=args.batch_size)
        score_predictor.train()

    else:
        pass
        # detector = HumanDetector(mode=args.mode)
        # # Keep track of 10 pieces of history
        # input_window = detector.init_input_window()
        # # Initialize motion and action LSTM model.
        # movement_predictor = MovementPredictor(mode=args.mode)
        # score_predictor=ScorePredictor(mode=args.mode)
        #
        # play = 1
        #
        # last_time = time.time()
        # frames_count = 0
        #
        # paused = True
        # print('Game paused, press p to start.')
        # while True:
        #     if not paused:
        #
        #         time.sleep(0.1)
        #
        #         rep = detector.forward()
        #         input_window[:-1, :] = input_window[1:, :]
        #         input_window[-1, :] = np.array(rep).reshape(-1, 128)
        #             # print('Feature: ', input_window[-1, :])
        #             # print('Mobile feature extractwsadwdwdswdadion time:', (time.time()-start_time)*1000, 'ms')
        #             # visualize_feature(rep[0], image_np)
        #
        #         movement_index = movement_predictor.inference(
        #             input_window.reshape(-1, detector.steps_of_history, 128))
        #         score = score_predictor.inference(input_window)
        #
        #         if score > 500:
        #             action_index = 1
        #         else:
        #             action_index = 0
        #
        #         if play == 1:
        #             take_action(movement_index, action_index)
        #             if action_index == 1:
        #                 time.sleep(2)
        #
        #         current_time = time.time()
        #         if current_time - last_time >= 1:
        #             # print('{} frames per second'.format(frames_count))
        #             last_time = current_time
        #             frames_count = 0
        #         else:
        #             frames_count = frames_count + 1
        #
        #     # Check keys for pause / unpause / quit
        #     is_quit, paused = check_pause(paused)
        #     if is_quit:
        #         break
        #
        #