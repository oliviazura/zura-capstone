import gymnasium as gym
env = gym.make("MountainCar-v0", render_mode="human")

MAX_STEPS_PER_EPISODE = 200
MAX_NUM_EPISODES = 5000

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