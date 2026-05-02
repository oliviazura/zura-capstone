'''
    This module includes utility functions used for training and testing the model in main.py
'''

import numpy as np
import constants as const
import q_learner as Q

def train(agent, env):
    '''Uses the Q-learning "learn" method to train the agent.
        Args:
            agent: Q_learner object
            env: mountain car Gymnasium environment

        Returns:
            np.argmax(agent.Q, axis =2) : learned policy from training

    '''
    
    #sets the best reward value so that it will be updated form the first episode
    best_reward = -float('inf')


    for episode in range(const.MAX_NUM_EPISODES):
        
        #reset environment each iteration
        episode_done = False
        obs, info = env.reset()
        total_reward = 0.0

        while not episode_done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward 
            episode_done = terminated or truncated

        #updates the statistics file each time a new best reward is found
        if total_reward > best_reward:
            best_reward = total_reward
            with open("statistics.txt", "a") as file:
                file.write(f"\nEpisode  {episode} reward: {best_reward}")

        #prints episode details to console
        print("Episode#:{} reward:{} best_reward:{} epsilon:{}".format(episode,
                                     total_reward, best_reward, agent.epsilon))
        
    return np.argmax(agent.Q, axis =2)

def test(agent, env, policy):
    '''Tests the agent's performance for one episode 
        Args:
            agent: Q_learner object
            env: mountain car Gymnasium environment
            policy: ndarray trained policy returned from training
        
        Returns:
            total_reward : float value representing total reward for the test episode'''
    
    #reset environment
    episode_done = False
    obs, info = env.reset()
    total_reward = 0.0

    while not episode_done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        total_reward += reward
        episode_done = terminated or truncated
   
    return total_reward