import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from custom_env import SmartFarmingEnv
import argparse

def train_dqn(learning_rate=1e-3, buffer_size=10000, batch_size=64, gamma=0.99, exploration_fraction=0.1, total_timesteps=100000, save_model=True, experiment_id="default"):
    env = SmartFarmingEnv()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        verbose=1
    )

    # Eval callback - save directly to logs folder with unique filename
    eval_callback = EvalCallback(env, best_model_save_path="./models/dqn/",
                                 log_path="./logs/", eval_freq=1000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    if save_model:
        model.save("./models/dqn/dqn_model")
        print("DQN model saved to ./models/dqn/dqn_model")
    else:
        print("DQN training completed (model not saved, logs preserved)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN on Smart Farming Environment')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--exploration_fraction', type=float, default=0.1, help='Exploration fraction')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total timesteps')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save model after training')
    parser.add_argument('--no_save_model', action='store_false', dest='save_model', help='Do not save model after training')
    parser.add_argument('--experiment_id', type=str, default='default', help='Unique identifier for this experiment')

    args = parser.parse_args()
    train_dqn(**vars(args))