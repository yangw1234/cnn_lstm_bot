import time

import numpy as np
import matplotlib.pyplot as plt

# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.zeros(N)
#
# print(x.shape)
# print(y.shape)
# print(colors.shape)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()
from get_keys import key_check

"""Pause/unpause the game using 'p'. Quit the game using 'q'."""
paused = False
while True:
    keys = key_check()

    if 'P' in keys:
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            # cv2.destroyAllWindows()
            time.sleep(1)

    elif 'O' in keys:
        print('Quitting!')
        # cv2.destroyAllWindows()
        quit = True
        time.sleep(0.5)

    if 'U' in keys:
        print('Game end')
        end_of_game = True
        time.sleep(0.5)

    if 'return' in keys:
        print('Start Game!')
        end_of_game = False
        time.sleep(0.5)