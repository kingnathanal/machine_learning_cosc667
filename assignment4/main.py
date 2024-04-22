import numpy as np
import matplotlib.pyplot as plt
import random

class Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.arm_probabilities = np.random.rand(n_arms)  # Random probabilities for each arm
        self.counts = np.zeros(n_arms)  # Counts of pulls for each arm
        self.values = np.zeros(n_arms)
    
    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)  # Estimated values of each arm

    def pull(self, arm):
        # Reward is 1 with probability equal to the arm's probability, otherwise 0
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

def run_bandit_simulation(strategy='greedy', n_arms=5, epsilon=0.4, pulls=1000, max_episodes=200):
    bandit = Bandit(n_arms)
    overall_rewards = []

    for episode in range(max_episodes):
        rewards = np.zeros(pulls)
        bandit.reset()

        for i in range(pulls):
            if strategy == 'greedy':
                chosen_arm = bandit.choose_arm(epsilon=0)  # Greedy strategy
            elif strategy == 'epsilon-greedy':
                chosen_arm = bandit.choose_arm(epsilon=epsilon)  # Epsilon-greedy strategy
            
            reward = bandit.pull(chosen_arm)
            bandit.update(chosen_arm, reward)
            rewards[i] = reward

        overall_rewards.append(np.mean(rewards))
    return overall_rewards

n_arms = 5
pulls = 1000
epsilon = 0.4
# Run simulations
rewards_greedy = run_bandit_simulation(strategy='greedy', n_arms=n_arms, pulls=pulls)
rewards_epsilon_greedy = run_bandit_simulation(strategy='epsilon-greedy', n_arms=n_arms, epsilon=epsilon, pulls=pulls)
rewards_base = run_bandit_simulation(strategy='epsilon-greedy', n_arms=n_arms, epsilon=0.1, pulls=pulls)

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(rewards_greedy, label='Greedy (ε=0)')
plt.plot(rewards_epsilon_greedy, label=f'Epsilon-Greedy (ε={epsilon})', linestyle='dotted')
plt.plot(rewards_base, label='Epsilon-Greedy (ε=0.1)', linestyle='--')
plt.xlabel('Episodes')
plt.ylabel('Mean Average of Reward')
plt.title('Comparison of Greedy and Epsilon-Greedy Strategies')
plt.legend()
plt.grid(True)
plt.show()
