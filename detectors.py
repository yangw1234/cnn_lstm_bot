import time
import numpy as np

class Detector:

    def predict(self):
        # output: action_index, movement_index
        pass

class RandomDetector(Detector):

    def __init__(self):
        self.last_shoot_time = time.time()
        self.movement_count = [0, 0, 0, 0]

    def predict(self):
        difference = self.movement_count[0] -self.movement_count[1]
        no_right = self.movement_count[3] > 4
        no_up = difference > 5 or (difference < 0 and difference > -3)
        no_down = difference < -5 or (difference > 0 and difference < 3)

        this_shoot_time = time.time()
        # if this_shoot_time - last_shoot_time > 8:
        if np.sum(self.movement_count) > 8 or (no_right and no_up and no_down):
            action_index = 1
            # print("total movement count is", str(self.movement_count))
            print("total movement count is", np.sum(self.movement_count))

            self.last_shoot_time = this_shoot_time
            self.movement_count = [0, 0, 0, 0]
        elif this_shoot_time - self.last_shoot_time < 4:
            action_index = 0
        else:
            action_p = np.random.rand()
            if action_p > 0.8:
                print("total movement count is", np.sum(self.movement_count))
                action_index = 1
                self.last_shoot_time = this_shoot_time
                self.movement_count = [0, 0, 0, 0]
            else:
                action_index = 0

        rand_movement = np.random.choice(np.arange(1, 4), p=[0.25, 0.25, 0.5])
        choose_movement = np.random.randint(2)
        if not no_right and (
                rand_movement == 3 or ((no_up or no_down) and choose_movement == 1) or self.movement_count[3] < 2):
            movement_index = 4
        elif not no_up and (rand_movement == 1 or ((no_right or no_down) and choose_movement == 0)):
            movement_index = 1
        elif not no_down and (
                rand_movement == 2 or (no_up and choose_movement == 0) or (no_right and choose_movement == 1)):
            movement_index = 2

        else:
            movement_index = 0
        self.movement_count[movement_index - 1] += 1

        return action_index, movement_index

