"""
The code in this file is sourced from "https://gymnasium.farama.org/introduction/train_agent/"
Followed along with Blackjack example and experimented with agent training
Wrote documentation for own understanding
"""

from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm  # Progress bar
from matplotlib import pyplot as plt


""" Using Q-Learning to solve Blackjack Environment
    
    -builds a Q-table to keep track of actions and their values
        -States: hand value, dealer cards, usable aces
        -Actions: hit or stand
        -Q-values: expected rewards for each action
        
    -process
        -try actions
        -update q-table
        -improve by filling table
        -exploit vs explore: weighing when to try something new vs using a known value
        
    -environment details
        -Observation: (player_sum, dealer_card, usable_ace)
        -Actions: 0=stand, 1=hit
        -rewards: +1 win, -1 loss, 0 draw
        -epsiode ends: stands or busts
        
    -epsilon greedy strategy for balancong
        -probablity = epsilon: choose random (Explore)
        -probability = 1-epsilon: choose best known (exploit)
        -want to start high and gradually decrease epsilon"""

class BlackjackAgent:
    def __init__(self, 
        env: gym.Env, 
        learning_rate: float, 
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        
        """Initializing BlackJack Q-Learning Agent
        
        Args:
            env: openaigym training environment
            learning_rate: how quickly to update Q-values (0-1)
            intial_epsilon: starting exploration rate (usually 1)
            epsilon_decay: how much to reduce epsilon each episode
            final_epsilon: minimum exploration rate (usually 0.1)
            discount_factor: how much to value future rewards (0-1)
        """

        self.env = env
        
        #q table maps (state, action) to expected reward
        #defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        #exploration params
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        #track learning
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon greedy strategy
            Returns action: 0 (stand) or 1 (hit)"""
        
        #explore
        if np.random.random() < self.epsilon:
                return self.env.action_space.sample()
        
        # exploit
        else:
             return int(np.argmax(self.q_values[obs]))
        
    def update(self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool]):
         
        """Update q value based on experience"""

        #best value from next state
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        #bellman equation for q-value
        target = reward + self.discount_factor * future_q_value

        #how wrong was the estimate?

        temporal_difference = target - self.q_values[obs][action]

        #update estimate towards error based on learning rate
        self.q_values[obs][action] = (
             self.q_values[obs][action] + self.lr * temporal_difference
        )

        #track learning process
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
         self.epsilon - max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    

        
#hyperparams
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

"""Note: tried varying with hyperparams to solve reward issue, did not work well. test seems to work fine regardless"""

env = gym.make("Blackjack-v1", sab = False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
     env = env, 
     learning_rate=learning_rate, 
     initial_epsilon=start_epsilon, 
     epsilon_decay=epsilon_decay, 
     final_epsilon=final_epsilon
)

for episode in tqdm(range(n_episodes)):
     #start a new hand
    obs, info = env.reset()
    done = False

    #play one hand
    while not done:
        action = agent.get_action(obs)

        next_obs, reward, terminated, trunacted, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)


        done = terminated or trunacted
        obs = next_obs
        
    agent.decay_epsilon()

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

'''Copy-pasted code for visualization'''
# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()


# Test your agent
def test_agent(agent, env, num_episodes = 1000):
    """testing the agent's performance without any additional learning"""

    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

# Test your agent
test_agent(agent, env)

env.close()