
import gymnasium as gym
import numpy as np
from copy import deepcopy
import ns_gym as nsg

import ns_gym.base as base
import random


class RandomAgent(base.Agent):
    """A random agent that samples actions uniformly at random from the action space."""
    def __init__(self, env):
        self.env = env

    def act(self, observation, env):
        return self.env.action_space.sample()

