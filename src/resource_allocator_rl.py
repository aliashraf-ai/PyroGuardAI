"""
Reinforcement Learning Resource Allocator
Q-Learning agent for optimal wildfire response resource allocation
Learns to make strategic decisions: deploy drones, helicopters, firebreaks, evacuations
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict


class WildfireEnvironment:
    """Simulated wildfire environment for RL training"""

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.fire_intensity = None
        self.resources = None
        self.steps = 0
        self.max_steps = 50

    def reset(self):
        """Reset environment to initial state"""
        # Initialize random fire pattern
        self.fire_intensity = np.random.rand(self.grid_size, self.grid_size) * 0.3

        # Start fires in random locations
        n_initial_fires = np.random.randint(2, 5)
        for _ in range(n_initial_fires):
            x, y = np.random.randint(0, self.grid_size, 2)
            self.fire_intensity[x, y] = np.random.uniform(0.7, 1.0)

        # Available resources
        self.resources = {
            'drones': 8,
            'helicopters': 3,
            'ground_crews': 5,
            'firebreaks': 2
        }

        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """
        Get current state representation
        State = (total_fire, max_fire, fire_spread, resources_available)
        """
        total_fire = np.sum(self.fire_intensity)
        max_fire = np.max(self.fire_intensity)
        fire_spread = np.sum(self.fire_intensity > 0.5)
        resources_avail = sum(self.resources.values())

        # Discretize state for Q-table
        state = (
            min(int(total_fire), 20),
            min(int(max_fire * 10), 10),
            min(int(fire_spread), 15),
            min(resources_avail, 18)
        )

        return state

    def step(self, action):
        """
        Execute action and return (next_state, reward, done)

        Actions:
        0: Deploy drones (suppress small fires)
        1: Deploy helicopters (suppress large fires)
        2: Send ground crews (contain spread)
        3: Create firebreak (prevent spread)
        4: Evacuate zone (save lives, no suppression)
        """

        reward = 0
        done = False

        # Execute action
        if action == 0 and self.resources['drones'] > 0:
            # Drones suppress small fires effectively
            mask = self.fire_intensity < 0.5
            self.fire_intensity[mask] *= 0.6
            self.resources['drones'] -= 1
            reward += 5

        elif action == 1 and self.resources['helicopters'] > 0:
            # Helicopters tackle large fires
            mask = self.fire_intensity >= 0.5
            self.fire_intensity[mask] *= 0.4
            self.resources['helicopters'] -= 1
            reward += 10

        elif action == 2 and self.resources['ground_crews'] > 0:
            # Ground crews slow spread
            self.fire_intensity *= 0.9
            self.resources['ground_crews'] -= 1
            reward += 3

        elif action == 3 and self.resources['firebreaks'] > 0:
            # Firebreaks block spread in high-intensity areas
            mask = self.fire_intensity > 0.7
            self.fire_intensity[mask] *= 0.3
            self.resources['firebreaks'] -= 1
            reward += 8

        elif action == 4:
            # Evacuation - saves lives but doesn't suppress fire
            reward += 2

        else:
            # Invalid action (resource unavailable)
            reward -= 5

        # Fire spreads naturally
        self._spread_fire()

        # Calculate penalties
        total_damage = np.sum(self.fire_intensity)
        critical_fires = np.sum(self.fire_intensity > 0.8)

        reward -= total_damage * 2
        reward -= critical_fires * 5

        self.steps += 1

        # Episode ends if fire is controlled or max steps reached
        if np.max(self.fire_intensity) < 0.1:
            reward += 50  # Bonus for extinguishing
            done = True
        elif self.steps >= self.max_steps:
            reward -= 20  # Penalty for timeout
            done = True

        next_state = self._get_state()

        return next_state, reward, done

    def _spread_fire(self):
        """Simulate fire spread to adjacent cells"""
        new_intensity = self.fire_intensity.copy()

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.fire_intensity[i, j] > 0.3:
                    # Spread to neighbors
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            spread_amount = self.fire_intensity[i, j] * 0.15
                            new_intensity[ni, nj] = min(1.0, new_intensity[ni, nj] + spread_amount)

        self.fire_intensity = new_intensity


class QLearningAgent:
    """Q-Learning agent for resource allocation"""

    def __init__(self, n_actions=5, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        self.action_names = [
            'Deploy Drones',
            'Deploy Helicopters',
            'Send Ground Crews',
            'Create Firebreak',
            'Evacuate Zone'
        ]

        print("🤖 Q-Learning Agent initialized")
        print(f"   Actions: {n_actions}")
        print(f"   Learning rate: {self.lr}")
        print(f"   Discount factor: {self.gamma}")
        print(f"   Exploration rate: {self.epsilon}")

    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Q-learning update rule"""
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[state][action] += self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Reduce exploration over time"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(n_episodes=500):
    """Train the RL agent"""

    print("\n" + "=" * 60)
    print("🔥 TRAINING REINFORCEMENT LEARNING AGENT")
    print("=" * 60)

    env = WildfireEnvironment(grid_size=10)
    agent = QLearningAgent(n_actions=5)

    episode_rewards = []
    episode_lengths = []
    success_rate = []

    print(f"\n📊 Training for {n_episodes} episodes...")
    print("-" * 60)

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_lengths.append(env.steps)

        # Track success (fire controlled)
        success = np.max(env.fire_intensity) < 0.1
        success_rate.append(1 if success else 0)

        agent.decay_epsilon()

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            recent_success = np.mean(success_rate[-50:]) * 100

            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Steps: {avg_length:5.1f} | "
                  f"Success: {recent_success:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

    # Final statistics
    final_success = np.mean(success_rate[-100:]) * 100
    final_reward = np.mean(episode_rewards[-100:])

    print(f"\n📈 Final Performance (last 100 episodes):")
    print(f"   Success Rate: {final_success:.1f}%")
    print(f"   Average Reward: {final_reward:.2f}")
    print(f"   Q-Table Size: {len(agent.q_table)} states learned")

    return agent, episode_rewards, success_rate


def test_agent(agent, n_tests=10):
    """Test trained agent performance"""

    print("\n" + "=" * 60)
    print("🧪 TESTING TRAINED AGENT")
    print("=" * 60)

    env = WildfireEnvironment(grid_size=10)

    test_results = []

    for test in range(n_tests):
        state = env.reset()
        total_reward = 0
        done = False
        actions_taken = []

        print(f"\n🔥 Test {test + 1}:")
        print(f"   Initial fires: {np.sum(env.fire_intensity > 0.5)}")
        print(f"   Initial intensity: {np.sum(env.fire_intensity):.2f}")

        while not done:
            action = agent.choose_action(state, training=False)
            actions_taken.append(action)
            next_state, reward, done = env.step(action)

            state = next_state
            total_reward += reward

        success = np.max(env.fire_intensity) < 0.1
        final_intensity = np.sum(env.fire_intensity)

        print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Steps Taken: {env.steps}")
        print(f"   Final Intensity: {final_intensity:.2f}")
        print(f"   Most Used Action: {agent.action_names[max(set(actions_taken), key=actions_taken.count)]}")

        test_results.append({
            'success': success,
            'reward': total_reward,
            'steps': env.steps,
            'actions': actions_taken
        })

    # Summary
    success_count = sum(r['success'] for r in test_results)
    avg_reward = np.mean([r['reward'] for r in test_results])
    avg_steps = np.mean([r['steps'] for r in test_results])

    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY:")
    print(f"   Success Rate: {success_count}/{n_tests} ({success_count / n_tests * 100:.1f}%)")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print("=" * 60)

    return test_results


def visualize_training(episode_rewards, success_rate):
    """Visualize training progress"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Rewards over time
    ax1.plot(episode_rewards, alpha=0.3, color='blue', linewidth=0.5)

    # Moving average
    window = 50
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
    ax1.plot(range(window - 1, len(episode_rewards)), moving_avg,
             color='red', linewidth=2, label=f'{window}-episode moving average')

    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    ax1.set_title('📈 RL Agent Learning Progress - Reward Over Time',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Add improvement annotation
    initial_avg = np.mean(episode_rewards[:50])
    final_avg = np.mean(episode_rewards[-50:])
    improvement = ((final_avg - initial_avg) / abs(initial_avg)) * 100

    ax1.text(0.98, 0.05,
             f'Improvement: {improvement:+.1f}%\nInitial: {initial_avg:.1f}\nFinal: {final_avg:.1f}',
             transform=ax1.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen' if improvement > 0 else 'lightcoral',
                       edgecolor='darkgreen' if improvement > 0 else 'darkred', linewidth=2, alpha=0.9),
             fontsize=10, fontweight='bold', family='monospace')

    # Success rate over time
    window = 50
    success_moving_avg = np.convolve(success_rate, np.ones(window) / window, mode='valid') * 100

    ax2.plot(range(window - 1, len(success_rate)), success_moving_avg,
             color='green', linewidth=3, label='Success Rate')
    ax2.fill_between(range(window - 1, len(success_rate)), success_moving_avg,
                     alpha=0.3, color='green')

    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('🎯 Fire Control Success Rate Over Training',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 100)

    # Add final success rate
    final_success = np.mean(success_rate[-100:]) * 100
    ax2.axhline(y=final_success, color='red', linestyle='--', linewidth=2,
                label=f'Final: {final_success:.1f}%')
    ax2.legend(fontsize=10)

    plt.tight_layout()

    # Save
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/rl_training_progress.png', dpi=150, facecolor='white')
    print(f"\n📊 Training visualization saved to: results/rl_training_progress.png")
    plt.show()


def save_agent(agent):
    """Save trained agent"""
    os.makedirs('models', exist_ok=True)

    with open('models/rl_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)

    print(f"💾 Agent saved to: models/rl_agent.pkl")


def run_demo():
    """Complete RL training and testing demo"""

    print("\n" + "=" * 60)
    print("🎮 WILDFIRE RESOURCE ALLOCATOR - REINFORCEMENT LEARNING")
    print("=" * 60)

    # Train agent
    agent, episode_rewards, success_rate = train_agent(n_episodes=500)

    # Test agent
    test_results = test_agent(agent, n_tests=10)

    # Visualize training
    visualize_training(episode_rewards, success_rate)

    # Save agent
    save_agent(agent)

    print("\n✅ RL COMPONENT COMPLETE!")
    print("\n🎉 ALL 4 COMPONENTS FINISHED!")
    print("=" * 60)
    print("✅ DNN - Fire spread predictor (numerical data)")
    print("✅ CNN - Satellite image analyzer (image classification)")
    print("✅ PSO - Drone swarm coordinator (multi-objective optimization)")
    print("✅ RL  - Resource allocator (Q-learning decision making)")
    print("=" * 60)
    print("\n🚀 Next: Create integrated demo with all components!")


if __name__ == '__main__':
    run_demo()