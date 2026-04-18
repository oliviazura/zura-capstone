import gymnasium as gym
import numpy as np
import constants as const

def train(agent, env):
    best_reward = -float('inf')
    for episode in range(const.MAX_NUM_EPISODES):
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
        if total_reward > best_reward:
            best_reward = total_reward
            with open("statistics.txt", "a") as file:
                file.write(f"\nEpisode  {episode} reward: {best_reward}")
        print("Episode#:{} reward:{} best_reward:{} epsilon:{}".format(episode,
                                     total_reward, best_reward, agent.epsilon))
    return np.argmax(agent.Q, axis =2)

def test(agent, env, policy):
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
        
