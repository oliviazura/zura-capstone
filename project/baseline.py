import gymnasium as gym
from gymnasium.wrappers import  RecordVideo

env = gym.make("MountainCar-v0", render_mode="rgb_array")
MAX_STEPS_PER_EPISODE = 200
MAX_NUM_EPISODES = 10

env = RecordVideo(
    env,
    video_folder = "project",
    name_prefix = "baseline",
    episode_trigger = lambda x: x == 10
)

for episode in range(MAX_NUM_EPISODES):
    terminated = False
    obs = env.reset()
    total_reward = 0.0
    step = 0
    while (not terminated and step <= MAX_STEPS_PER_EPISODE):
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
    print("\n Episode #{} ended in {} steps. total_reward={}".format(episode, step+1, total_reward))
env.close()