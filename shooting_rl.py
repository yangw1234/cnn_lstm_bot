import cv2
import os
import re
import argparse
import time
import numpy as np

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

    agent = RandomAgent()

    env = FIFAEnv()

    ref_center = cv2.imread("reference.jpg")


    start = True
    paused = True
    print('Game paused, press p to start.')
    while paused:
        is_quit, paused = check_pause(paused)

    screen = env.reset()
    while True:
        if not paused:
            # if start:
            #     input_window = init_input_window(detector)
            #     start = False
            cur_time = time.time()

            movement_index, action_index = agent.take_action(screen)
            screen, reward, done = env.act(movement_index, action_index)

            print(reward)

            if done:
                screen = env.reset()
        # Check keys for pause / unpause / quit
        # is_quit, paused = check_pause(paused)
        # if is_quit:
        #     break


