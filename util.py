import time

from direct_keys import PressKey, ReleaseKey, enter
from grab_screen import grab_screen
import cv2
import numpy as np
import win32con as wcon

from score_detector import ScoreDetector

ymin, ymax, xmin, xmax = 33, 753, 8, 1288


def is_over(screen, ref):
    center = get_center(screen)
    diff = np.mean(np.square(center - ref))
    if diff <  90:
        return True
    else:
        return False



def get_center(screen):
    screen = screen[100:600, 400:1000]
    return screen

if __name__ == "__main__":

    detector = ScoreDetector()
    screen = grab_screen(region=None)
    screen = screen[ymin:ymax, xmin:xmax]
    score_screen = screen[58:93, 1126:1186]
    t1 = time.time()
    # print("grab screen time is " + str(t1 - t0))
    print(detector.predict(score_screen))



