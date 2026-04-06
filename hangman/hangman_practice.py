'''Practing with Hangman Environment'''

import gymnasium as gym
from gymnasium.utils.play import play
import ale_py
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

env = gym.make('ALE/Hangman-v5', render_mode = "rgb_array")
play(env)
