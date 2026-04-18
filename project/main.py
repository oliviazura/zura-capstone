import gymnasium as gym
import constants as const
from gymnasium.wrappers import RecordVideo
from q_learner import Q_Learner
from util import train, test

env = gym.make('MountainCar-v0', render_mode = "rgb_array")
agent = Q_Learner(env)
env = RecordVideo(
    env,
    video_folder = "project",
    name_prefix = "mountaincar",
    episode_trigger = lambda x: x % const.TRAINING_PERIOD == 0
)
learned_policy = train(agent, env)
for _ in range(1000):
    test(agent, env, learned_policy)
env.close()