'''Practing with Hangman Environment'''

import gymnasium as gym
from gymnasium.utils.play import play
import ale_py
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

env = gym.make('ALE/Hangman-v5', render_mode = "rgb_array")
#play(env)

'''
    from this observe that only actions are up (2), down (5), and fire (1) to navigate letters and entering them. 
    discarded letters are removed from the list, 
    agent should: guess a letter
    
    11 attempts before failure

    11 most common letters: e, t, a, o, i , n, s, h, r, d, l
    note that brute forcing could work in a lot of cases, but for a word like "ugly" would not work
    
    ideas
    a: force model to iterate through these letters first
    b: force model to iterate vowels first
    c: observe if model converges to guessing these letters first 

'''
