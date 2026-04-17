import gymnasium as gym
from q_learner import Q_Learner
from util import train, test

env = gym.make('MountainCar-v0')
agent = Q_Learner(env)
learned_policy = train(agent, env)
# Use the Gym Monitor wrapper to evalaute the agent and record video
gym_monitor_path = "./gym_monitor_output"
env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
for _ in range(1000):
    test(agent, env, learned_policy)
env.close()