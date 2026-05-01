'''
    This module contains the constants used in the final model.
    These constants can be modified within this file to adust the model.
    Constants that can be modified are denoted with an asterisk.

    *MAX_NUM_EPISODES (int): The maximum number of episodes for training
    *STEPS_PER_EPISODE (int): The maximum number of steps the agent can take in an episode
    *TRAINING_PERIOD (int): The interval at which to record a video of the agent's performance
    *EPSILON_MIN (float): The minimum value of epsilon that can be decayed to during training
    max_num_steps (int): The total number of steps during training, used to calculate epsilon decay
    *EPSILON_DECAY (float): The rate at which the epsilon value decays each step
    *ALPHA (float): The learning rate used in the Q-learning update equation
    *GAMMA (float): The discount factor the agent uses when factoring in rewards,  used in the Q-learning update equation
    NUM_DISCRETE_BINS (int): The number of bins to discretize each observation dimension into
'''

MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
TRAINING_PERIOD = 10000

EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30