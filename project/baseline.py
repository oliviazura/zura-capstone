"""
    This module contains baseline code to gain an understanding of how the agent interacts in the Mountain Car environment.
    The agent will take 200 randomized steps for a duriation  of 10 episodes.
    A video of the agent's last attempt is saved to a "videos" folder and the total reward is outputted to the terminal.
"""


import gymnasium as gym
from gymnasium.wrappers import RecordVideo

#create gymnasium environment, rgb array required for video
env = gym.make("MountainCar-v0", render_mode="rgb_array")

MAX_STEPS_PER_EPISODE = 200
MAX_NUM_EPISODES = 10

#allow video recording for environment, saves to main folder, records the tenth episode
env = RecordVideo(
    env,
    video_folder = "project",
    name_prefix = "baseline",
    episode_trigger = lambda x: x == 10
)

#iterates through each episode while taking random actions
for episode in range(MAX_NUM_EPISODES):
    #environment must be reset each loop
    terminated = False
    truncated = False
    obs = env.reset()
    total_reward = 0.0
    step = 0

    while (not terminated and not truncated):
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

    print("\n Episode #{} ended in {} steps. total_reward={}".format(episode, step, total_reward))

#close environment when finished
env.close()