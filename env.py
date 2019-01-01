import time

from direct_keys import *
from grab_screen import grab_screen
from score_detector import ScoreDetector
from util import is_over
import cv2
import numpy as np


class FIFAEnv():

    def __init__(self):
        self.ref_center = cv2.imread("reference.jpg")
        self.ymin, self.ymax, self.xmin, self.xmax = 33, 753, 8, 1288
        self.pre_movement = []
        self.detector = ScoreDetector()
        self.score_tl_y = 65
        self.score_br_y = 100
        self.score_tl_x = 1200 # 1230
        self.score_br_x = 1260 # 1290
        self.score_tl_y = 58
        self.score_br_y = 93
        self.score_tl_x = 1126 # 1230
        self.score_br_x = 1186 # 1290

    def observe(self):
        screen = self._get_screen()
        if is_over(screen, self.ref_center):
            done = True
        else:
            done = False

        return screen, done

    def act(self, movement_index, action_index):
        pre_action_screen = None
        if action_index == 1:
            pre_action_screen = self._get_screen()
        movement_custom_b = [[], [W], [S], [A], [D]]

        action = [[], [spacebar]]

        print('movement: ' + str(movement_index) + ' and action: ' + str(action_index))

        move = movement_custom_b[movement_index]
        act = action[action_index]

        if self.pre_movement != move:
            for index in self.pre_movement:
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

        reward = 0
        if action_index == 1:

            score_screen = pre_action_screen[self.score_tl_y:self.score_br_y, self.score_tl_x + 2:self.score_br_x + 2]

            # cv2.imshow("score", score_screen)
            # cv2.waitKey(0)

            pre_score, success = self.detector.predict(score_screen)
            if not success:
                cv2.imwrite("score_screen_error.png", score_screen)
            for i in range(5):
                time.sleep(0.4)
                current_screen = self._get_screen()
                score_screen = score_screen = pre_action_screen[self.score_tl_y:self.score_br_y, self.score_tl_x + 2:self.score_br_x + 2]
                current_score, success = self.detector.predict(score_screen)

                if not success:
                    print("ERROR")

                if success and current_score - pre_score > 0:
                    reward = current_score - pre_score
                    break

            print("pre score is %s" % pre_score)
            print("current score is %s" % current_score)
        time.sleep(0.18)
        current_screen = self._get_screen()
        done = is_over(current_screen, self.ref_center)
        return current_screen, reward, done


    def reset(self):
        print("game reset ")
        PressKey(enter)
        ReleaseKey(enter)
        time.sleep(0.5)
        screen = self._get_screen()
        return screen

    def is_over(self, screen, ref):
        center = screen[100:600, 400:1000]
        diff = np.mean(np.square(center - ref))
        if diff < 90:
            return True
        else:
            return False

    def _get_screen(self):
        t0 = time.time()
        screen = grab_screen(region=None)
        screen = screen[self.ymin:self.ymax, self.xmin:self.xmax]
        t1 = time.time()
        # print("grab screen time is " + str(t1 - t0))
        return screen