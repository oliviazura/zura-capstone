'''
    This module uses the Q-Learning agent to conduct training and testing for the Mountain Car environment.
    Videos are recorded in increments set by the training period constant.
'''


import gymnasium as gym
import constants as const
from gymnasium.wrappers import RecordVideo
from q_learner import Q_Learner
from util import train, test

env = gym.make('MountainCar-v0', render_mode = "rgb_array")
agent = Q_Learner(env)

#creates a wrapper using RecordVideo to record a video during specific segments of the training process
env = RecordVideo(
    env,
    video_folder = "project/videos",
    name_prefix = "mountaincar",
    episode_trigger = lambda x: x % const.TRAINING_PERIOD == 0
)

learned_policy = train(agent, env)

#tests the agent's performance for 1000 episodes
for _ in range(1000):
    test(agent, env, learned_policy)
env.close()