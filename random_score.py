from collections import deque

import cv2
import time
import numpy as np
from direct_keys import *
from get_keys import *
from grab_screen import grab_screen
from parameters import *
from score_detector import ScoreDetector


class random_shoot():

    def __init__(self):
        self.last_shoot_time = time.time()
        self.movement_count = [0, 0, 0, 0]
        self.prev_movement = []
        self.movement_index = 0
        self.action_index = 0

        self.paused = True
        self.quit = False
        self.end_of_game = False
        self.wait = False

        # self.training_data = deque()
        self.training_data = []
        self.training_data_all = []
        self.score_data = []


    def shooting(self):
        self.action_index = 1
        this_shoot_time = time.time()
        self.last_shoot_time = this_shoot_time
        print("movement_count is ", self.movement_count)
        self.movement_count = [0, 0, 0, 0]
        # print("total movement count is", np.sum(self.movement_count))

    def shoot(self):
        # this_shoot_time = time.time()
        # if this_shoot_time - last_shoot_time > 8:
        no_right = self.movement_count[3] > 5
        if np.sum(self.movement_count) > 20:
            self.shooting()
            # print("total movement count is", str(self.movement_count))

        # elif this_shoot_time - self.last_shoot_time < 4:
        elif np.sum(self.movement_count) < 4:
            self.action_index = 0
        else:
            action_p = np.random.rand()
            if action_p > 0.7:
                self.shooting()
            else:
                self.action_index = 0

        rand_movement = np.random.choice(np.arange(1, 4), p=[0.25, 0.25, 0.5])
        choose_movement = np.random.randint(2)
        if not no_right and rand_movement == 3 :
            self.movement_index = 4
        elif rand_movement == 1 or (no_right and choose_movement == 0):
            self.movement_index = 1
        elif rand_movement == 2 or (no_right and choose_movement == 1):
            self.movement_index = 2
        else:
            self.movement_index = 0
        self.movement_count[self.movement_index - 1] += 1


    def take_action(self):
        """Send action to the game."""
        # Movements
        movement_custom_b = [[], [W], [S], [A], [D]]

        action = [[], [spacebar]]

        print('movement: ' + str(self.movement_index) + ' and action: ' + str(self.action_index))

        move = movement_custom_b[self.movement_index]
        act = action[self.action_index]

        if self.prev_movement != move:
            for index in self.prev_movement:
                ReleaseKey(index)
            for index in move:
                PressKey(index)

            self.prev_movement = move
        # for index in move:
        #     PressKey(index)
        # time.sleep(0.08)
        # # for index in movement_custom_b[movement_index]:
        # # ReleaseKey(index)
        # for index in move:
        #     ReleaseKey(index)

        # for index in movement_custom_b[movement_index]:
        #     PressKey(index)
        for index in action[self.action_index]:
            PressKey(index)
        time.sleep(0.18)
        # for index in movement_custom_b[movement_index]:
        # ReleaseKey(index)
        for index in action[self.action_index]:
            ReleaseKey(index)
        # time.sleep(0.5)

    def record(self):
        full_screen = grab_screen(region=None)

        screen = full_screen[33:753, 8:1288]
        score_screen = screen[58:93, 1126:1186]

        data = [screen, self.movement_index, self.action_index]
        self.training_data.append(data)
        # if len(self.training_data) > SCORE_RECORDING_NUM:
        #     self.training_data.popleft()

        #shoot record
        if self.action_index == 1:
            if len(self.training_data) >= STACK_NUM:
                self.training_data_all.append(self.training_data)
                self.score_data.append(score_screen)
            time.sleep(2.5)

            # for d in self.training_data:
            #     print("record movement index:", d[1], "record action index:", d[2])

            # self.training_data = deque()
            self.training_data = []

    def detect_score(self, score_screen):
        score_detector = ScoreDetector()
        score, is_success = score_detector.predict(score_screen)

        if not is_success:
            print("Cannot detect the score.")

        return score

    def save_data(self):
        #save the last score screen
        full_screen = grab_screen(region=None)
        screen = full_screen[33:753, 8:1288]
        score_screen = screen[58:93, 1126:1186]
        self.score_data.append(score_screen)

        # print("num of score recorded is", len(self.score_data))
        print("num of shoot recorded is", len(self.training_data_all))

        cur_time = int(time.time())
        image_name = RAND_IMAGE_PATH + str(cur_time)
        # score_name = SCORE_DIRECTORY + str(cur_time)

        score_detector = ScoreDetector()
        # last_score, is_success = score_detector.predict(self.score_data[0])
        # if not is_success:
        #     print("Cannot detect start score!")
        last_score = 0
        for shoot_index in range(len(self.training_data_all)):
            #1. detect score
            score, is_success = score_detector.predict(self.score_data[shoot_index + 1])
            this_score = score - last_score
            if this_score < 0:
                this_score = score
            last_score = score

            if not is_success:
                print("Cannot detect the score of shoot ", shoot_index)
                continue
            #save image
            shoot_data = self.training_data_all[shoot_index]
            for image_index in range(len(shoot_data)):
                image_filename = image_name + "_" + str(shoot_index) + "_" + str(image_index) \
                                 + "_movement_" + str(shoot_data[image_index][1]) \
                                 + "_action_" + str(shoot_data[image_index][2]) \
                                 + "_score_" + str(this_score)+ ".jpg"
                cv2.imwrite(image_filename, shoot_data[image_index][0])
                print("Writing", image_filename)
        print("Write done!")

    def reset(self):
        self.last_shoot_time = time.time()
        self.movement_count = [0, 0, 0, 0]
        self.prev_movement = []
        self.movement_index = 0
        self.action_index = 0
        #
        # self.paused = False
        # self.quit = False

        # self.training_data = deque()
        self.training_data = []
        self.training_data_all = []
        self.score_data = []

    def check_pause(self):
        """Pause/unpause the game using 'p'. Quit the game using 'q'."""
        """For recording, press 'u' as soon as one game ends. Press 'enter' to start a new game"""
        keys = key_check()

        if 'P' in keys:
            if self.paused:
                self.paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                self.paused = True
                # cv2.destroyAllWindows()
                time.sleep(1)

        elif 'O' in keys:
            print('Quitting!')
            # cv2.destroyAllWindows()
            self.quit = True

        if 'U' in keys:
            print('Game end')
            self.end_of_game = True
            self.wait = True

        if 'return' in keys:
            print('Start Game!')
            time.sleep(1)
            self.wait = False

    def main(self):
        print('Game paused, press p to start.')

        while True:
            self.check_pause()
            if not (self.paused or self.wait):
                # self.saved = False
                self.shoot()
                self.take_action()
                self.record()

            if self.end_of_game or self.quit:
                self.save_data()
                self.reset()
                self.end_of_game = False
                # self.saved = True

            if self.quit:
                break

if __name__ == '__main__':
    shoot = random_shoot()
    shoot.main()


