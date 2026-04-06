"""
Code sourced from "https://gymnasium.farama.org/introduction/train_agent/"
Followed along with initial example of Cart Pole Environment
"""

import gymnasium as gym

#cart pole environment
env = gym.make('CartPole-v1', render_mode="human")

#restarts the environment to start a new "episode"
observation, info = env.reset()
#observation: what the agent can see
#info: debugging

print(f"Starting observation: {observation}")
#Output: 
# Starting observation: [-0.00025287 -0.04458311 -0.01331567  0.02959813]
#cart_position, cart_velocity, pole_angle, poly_angular_velocity]

episode_over = False
total_reward = 0

while not episode_over:
    #choose an action: 0 = push cart left, 1= push cart right
    action = env.action_space.sample() # random action

    observation, reward, terminated, truncated, info = env.step(action)

    #reward: +1 for each step pole is upright
    #terminated: True if pole falls (failed)
    #truncated: true if time limit is hit

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode done. Total reward: {total_reward}")
env.close()