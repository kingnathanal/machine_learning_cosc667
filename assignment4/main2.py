import numpy as np
import matplotlib.pyplot as plt
import random

class Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.arm_probabilities = np.random.rand(n_arms)  # Random probabilities for each arm
        self.reset()

    def reset(self):
        # Reset counts and values for each new episode
        self.counts = np.zeros(self.n_arms)  # Counts of pulls for each arm
        self.values = np.zeros(self.n_arms)  # Estimated values of each arm

    def pull(self, arm):
        # Simulate pulling an arm
        reward = 1 if (random.random() < self.arm_probabilities[arm]) else 0
        return reward

    def update(self, arm, reward):
        # Update the estimates of the arm's value
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # New estimate is old estimate + (1/n) * (reward - old estimate)
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    def choose_arm(self, epsilon=0):
        # With probability epsilon choose a random arm, otherwise choose the best arm
        if random.random() < epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.values)  # Choose the arm with the highest estimated value

def run_bandit_simulation(strategy='greedy', n_arms=5, epsilon=0.1, steps_per_episode=100, max_episodes=10):
    bandit = Bandit(n_arms)
    overall_rewards = []

    for episode in range(max_episodes):
        episode_rewards = np.zeros(steps_per_episode)
        bandit.reset()  # Reset the bandit for each episode

        for i in range(steps_per_episode):
            if strategy == 'greedy':
                chosen_arm = bandit.choose_arm(epsilon=0)  # Greedy strategy
            elif strategy == 'epsilon-greedy':
                chosen_arm = bandit.choose_arm(epsilon=epsilon)  # Epsilon-greedy strategy
            
            reward = bandit.pull(chosen_arm)
            bandit.update(chosen_arm, reward)
            episode_rewards[i] = reward

        overall_rewards.append(np.mean(episode_rewards))
        print(f"Episode {episode + 1}: Average Reward = {overall_rewards[-1]}")

    return overall_rewards

    # Plotting results
    # plt.figure(figsize=(12, 6))
    # plt.plot(overall_rewards, label=f'{strategy} Strategy')
    # plt.xlabel('Episode')
    # plt.ylabel('Average Reward')
    # plt.title(f'Performance of {strategy} Strategy Over Episodes')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

# Parameters
n_arms = 5
epsilon = 0.4
steps_per_episode = 100
max_episodes = 200

# Run simulations for both strategies
print("Running Greedy Strategy Simulation")
greedy_rewards = run_bandit_simulation(strategy='greedy', n_arms=n_arms, steps_per_episode=steps_per_episode, max_episodes=max_episodes)

print("Running Epsilon-Greedy Strategy Simulation")
egreedy_rewards = run_bandit_simulation(strategy='epsilon-greedy', n_arms=n_arms, epsilon=0.1, steps_per_episode=steps_per_episode, max_episodes=max_episodes)

print("Running Epsilon-Greedy-Base Strategy Simulation")
egreedy_rewards_base = run_bandit_simulation(strategy='epsilon-greedy', n_arms=n_arms, epsilon=0.4, steps_per_episode=steps_per_episode, max_episodes=max_episodes)

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(greedy_rewards, label='greedy Strategy')
plt.plot(egreedy_rewards, label='epsilon-greedy Strategy')
plt.plot(egreedy_rewards_base, label='epsilon-greedy-base Strategy')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Performance of greedy Strategy Over Episodes')
plt.legend()
plt.grid(True)
plt.show()