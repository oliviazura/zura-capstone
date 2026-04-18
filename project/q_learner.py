import gymnasium as gym
import numpy as np
import constants as const

class Q_Learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = const.NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low) / self. obs_bins
        self.action_shape = env.action_space.n

        self.Q = np.zeros((self.obs_bins, self.obs_bins,
                           self.action_shape))
        self.alpha = const.ALPHA
        self.gamma = const.GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):

        if isinstance(obs, tuple):
            obs = obs[0]

        obs = np.array(obs, dtype=float)

        ratios = (obs - self.obs_low) / self.bin_width
        bins = np.floor(ratios).astype(int)
        clipped = np.clip(bins, 0, self.obs_bins - 1)

        return tuple(clipped)
    
    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        #Epsilon Greedy action slection 
        if self.epsilon > const.EPSILON_MIN:
            self.epsilon -= const.EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])
        
    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discretized_next_obs])
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error
