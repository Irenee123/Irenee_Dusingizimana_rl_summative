import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from custom_env import SmartFarmingEnv
import argparse

# REINFORCE Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

def train_reinforce(learning_rate=1e-3, gamma=0.99, num_episodes=10000, save_model=True, experiment_id="default"):
    env = SmartFarmingEnv()
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = PolicyNetwork(obs_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            obs, reward, done, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

        # Compute loss
        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"REINFORCE Episode {episode}, Loss: {loss.item()}")

    # Save basic evaluation log for REINFORCE
    import numpy as np
    log_file = f"./logs/reinforce_{experiment_id}_results.txt"
    with open(log_file, 'w') as f:
        f.write(f"REINFORCE Experiment: {experiment_id}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Final Loss: {loss.item()}\n")
        f.write("REINFORCE completed\n")

    # Rename evaluation files to unique names
    import os
    if os.path.exists("./logs/evaluations.npz"):
        os.rename("./logs/evaluations.npz", f"./logs/{experiment_id}_evaluations.npz")

    # Save model
    if save_model:
        torch.save(policy.state_dict(), "./models/pg/reinforce_model.pth")
        print("REINFORCE model saved to ./models/pg/reinforce_model.pth")
    else:
        print("REINFORCE training completed (model not saved, logs preserved)")

# PPO and A2C imports
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback

def train_ppo(learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99, gae_lambda=0.95, total_timesteps=100000, save_model=True, experiment_id="default"):
    env = SmartFarmingEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        verbose=1
    )

    # Eval callback - save with unique filename
    eval_callback = EvalCallback(env, best_model_save_path="./models/pg/",
                                 log_path="./logs/", eval_freq=1000,
                                 deterministic=True, render=False)
    # Note: EvalCallback saves to fixed filenames, so we need to rename after training

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Rename evaluation files to unique names
    if os.path.exists("./logs/evaluations.npz"):
        os.rename("./logs/evaluations.npz", f"./logs/{experiment_id}_evaluations.npz")

    if save_model:
        model.save("./models/pg/ppo_model")
        print("PPO model saved to ./models/pg/ppo_model")
    else:
        print("PPO training completed (model not saved, logs preserved)")

def train_a2c(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0, total_timesteps=100000, save_model=True, experiment_id="default"):
    env = SmartFarmingEnv()

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        verbose=1
    )

    # Eval callback - save directly to logs folder
    eval_callback = EvalCallback(env, best_model_save_path="./models/pg/",
                                 log_path="./logs/", eval_freq=1000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Rename evaluation files to unique names
    if os.path.exists("./logs/evaluations.npz"):
        os.rename("./logs/evaluations.npz", f"./logs/{experiment_id}_evaluations.npz")

    if save_model:
        model.save("./models/pg/a2c_model")
        print("A2C model saved to ./models/pg/a2c_model")
    else:
        print("A2C training completed (model not saved, logs preserved)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Policy Gradient algorithms')
    parser.add_argument('--algorithm', type=str, required=True, choices=['reinforce', 'ppo', 'a2c'],
                       help='Algorithm to train: reinforce, ppo, or a2c')

    # Common args
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # REINFORCE specific
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes (REINFORCE)')

    # PPO/A2C specific
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps (PPO/A2C)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (PPO)')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda (PPO/A2C)')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total timesteps (PPO/A2C)')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save model after training')
    parser.add_argument('--no_save_model', action='store_false', dest='save_model', help='Do not save model after training')
    parser.add_argument('--experiment_id', type=str, default='default', help='Unique identifier for this experiment')

    args = parser.parse_args()

    if args.algorithm == 'reinforce':
        train_reinforce(learning_rate=args.learning_rate, gamma=args.gamma, num_episodes=args.num_episodes, save_model=args.save_model, experiment_id=args.experiment_id)
    elif args.algorithm == 'ppo':
        train_ppo(learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
                 gamma=args.gamma, gae_lambda=args.gae_lambda, total_timesteps=args.total_timesteps, save_model=args.save_model, experiment_id=args.experiment_id)
    elif args.algorithm == 'a2c':
        train_a2c(learning_rate=args.learning_rate, n_steps=args.n_steps, gamma=args.gamma,
                 gae_lambda=args.gae_lambda, total_timesteps=args.total_timesteps, save_model=args.save_model, experiment_id=args.experiment_id)