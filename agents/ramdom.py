from agents.agent import Agent
import numpy as np


class RandomAgent(Agent):

    def __init__(self):
        pass

    def take_action(self, screen):

        return np.random.randint(0, 5), np.random.randint(0, 2)