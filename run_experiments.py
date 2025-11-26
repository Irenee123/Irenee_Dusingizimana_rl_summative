#!/usr/bin/env python3
"""
Automated Hyperparameter Tuning Experiments Runner

This script runs A2C hyperparameter tuning experiments:
- 10 A2C experiments

(DQN, REINFORCE, and PPO experiments already completed)

Results are saved to logs/ directory for analysis.
"""

import subprocess
import sys
import os
import time

def run_command(cmd, experiment_name):
    """Run a command and log the output"""
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())

        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}")

        if result.returncode == 0:
            print(f"[SUCCESS] {experiment_name} completed successfully")
        else:
            print(f"[FAILED] {experiment_name} failed with return code {result.returncode}")

        return result.returncode == 0

    except Exception as e:
        print(f"[ERROR] Error running {experiment_name}: {e}")
        return False

def main():
    print("Starting Automated Hyperparameter Tuning Experiments")
    print("This will run 10 A2C experiments")
    print("DQN, REINFORCE, and PPO experiments already completed. Each experiment may take several minutes to complete...")
    print()

    # DQN Experiments (10 runs) - NO MODEL SAVING for hyperparameter tuning
    dqn_commands = [
        "python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 50000 --no_save_model --experiment_id dqn_1",
        "python training/dqn_training.py --learning_rate 0.0001 --buffer_size 10000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 50000 --no_save_model --experiment_id dqn_2",
        "python training/dqn_training.py --learning_rate 0.001 --buffer_size 50000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 50000 --no_save_model --experiment_id dqn_3",
        "python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 128 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 50000 --no_save_model --experiment_id dqn_4",
        "python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 64 --gamma 0.95 --exploration_fraction 0.1 --total_timesteps 50000 --no_save_model --experiment_id dqn_5",
        "python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.2 --total_timesteps 50000 --no_save_model --experiment_id dqn_6",
        "python training/dqn_training.py --learning_rate 0.0005 --buffer_size 20000 --batch_size 32 --gamma 0.98 --exploration_fraction 0.15 --total_timesteps 50000 --no_save_model --experiment_id dqn_7",
        "python training/dqn_training.py --learning_rate 0.002 --buffer_size 5000 --batch_size 128 --gamma 0.99 --exploration_fraction 0.05 --total_timesteps 50000 --no_save_model --experiment_id dqn_8",
        "python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 50000 --no_save_model --experiment_id dqn_9",
        "python training/dqn_training.py --learning_rate 0.0001 --buffer_size 50000 --batch_size 128 --gamma 0.95 --exploration_fraction 0.2 --total_timesteps 50000 --no_save_model --experiment_id dqn_10",
    ]

    # REINFORCE Experiments (10 runs) - NO MODEL SAVING for hyperparameter tuning
    reinforce_commands = [
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.001 --gamma 0.99 --num_episodes 5000 --no_save_model --experiment_id reinforce_1",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.0001 --gamma 0.99 --num_episodes 5000 --no_save_model --experiment_id reinforce_2",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.001 --gamma 0.95 --num_episodes 5000 --no_save_model --experiment_id reinforce_3",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.0005 --gamma 0.98 --num_episodes 5000 --no_save_model --experiment_id reinforce_4",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.002 --gamma 0.99 --num_episodes 5000 --no_save_model --experiment_id reinforce_5",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.001 --gamma 0.9 --num_episodes 5000 --no_save_model --experiment_id reinforce_6",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.0001 --gamma 0.95 --num_episodes 5000 --no_save_model --experiment_id reinforce_7",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.0005 --gamma 0.99 --num_episodes 5000 --no_save_model --experiment_id reinforce_8",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.001 --gamma 0.99 --num_episodes 5000 --no_save_model --experiment_id reinforce_9",
        "python training/pg_training.py --algorithm reinforce --learning_rate 0.0002 --gamma 0.98 --num_episodes 5000 --no_save_model --experiment_id reinforce_10",
    ]

    # PPO Experiments (10 runs) - NO MODEL SAVING for hyperparameter tuning
    ppo_commands = [
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 2048 --batch_size 64 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id ppo_1",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0001 --n_steps 2048 --batch_size 64 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id ppo_2",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 1024 --batch_size 64 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id ppo_3",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 2048 --batch_size 128 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id ppo_4",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 2048 --batch_size 64 --gamma 0.95 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id ppo_5",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 2048 --batch_size 64 --gamma 0.99 --gae_lambda 0.9 --total_timesteps 50000 --no_save_model --experiment_id ppo_6",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0007 --n_steps 4096 --batch_size 32 --gamma 0.98 --gae_lambda 0.92 --total_timesteps 50000 --no_save_model --experiment_id ppo_7",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.001 --n_steps 1024 --batch_size 128 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id ppo_8",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 2048 --batch_size 64 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id ppo_9",
        "python training/pg_training.py --algorithm ppo --learning_rate 0.0005 --n_steps 2048 --batch_size 64 --gamma 0.97 --gae_lambda 0.93 --total_timesteps 50000 --no_save_model --experiment_id ppo_10",
    ]

    # A2C Experiments (10 runs) - NO MODEL SAVING for hyperparameter tuning
    a2c_commands = [
        "python training/pg_training.py --algorithm a2c --learning_rate 0.0007 --n_steps 5 --gamma 0.99 --gae_lambda 1.0 --total_timesteps 50000 --no_save_model --experiment_id a2c_1",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.0001 --n_steps 5 --gamma 0.99 --gae_lambda 1.0 --total_timesteps 50000 --no_save_model --experiment_id a2c_2",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.0007 --n_steps 10 --gamma 0.99 --gae_lambda 1.0 --total_timesteps 50000 --no_save_model --experiment_id a2c_3",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.0007 --n_steps 5 --gamma 0.95 --gae_lambda 1.0 --total_timesteps 50000 --no_save_model --experiment_id a2c_4",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.0007 --n_steps 5 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id a2c_5",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.001 --n_steps 5 --gamma 0.99 --gae_lambda 1.0 --total_timesteps 50000 --no_save_model --experiment_id a2c_6",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.0005 --n_steps 8 --gamma 0.98 --gae_lambda 0.97 --total_timesteps 50000 --no_save_model --experiment_id a2c_7",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.0007 --n_steps 5 --gamma 0.99 --gae_lambda 1.0 --total_timesteps 50000 --no_save_model --experiment_id a2c_8",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.0002 --n_steps 10 --gamma 0.95 --gae_lambda 0.9 --total_timesteps 50000 --no_save_model --experiment_id a2c_9",
        "python training/pg_training.py --algorithm a2c --learning_rate 0.001 --n_steps 5 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model --experiment_id a2c_10",
    ]

    # Run A2C experiments
    all_experiments = [
        ("A2C", a2c_commands),
    ]

    total_experiments = sum(len(commands) for _, commands in all_experiments)
    completed_experiments = 0

    print(f"Total experiments to run: {total_experiments}")
    print()

    start_time = time.time()

    for algorithm_name, commands in all_experiments:
        print(f"\nStarting {algorithm_name} experiments ({len(commands)} runs)")

        for i, cmd in enumerate(commands, 1):
            experiment_name = f"{algorithm_name} Run {i}"
            success = run_command(cmd, experiment_name)

            completed_experiments += 1
            elapsed_time = time.time() - start_time
            avg_time_per_experiment = elapsed_time / completed_experiments
            remaining_experiments = total_experiments - completed_experiments
            estimated_remaining_time = remaining_experiments * avg_time_per_experiment

            print(f"Progress: {completed_experiments}/{total_experiments} experiments completed")
            print(".1f")
            print(".1f")
            print()

            if not success:
                print(f"⚠️  {experiment_name} failed, but continuing with remaining experiments...")

    total_time = time.time() - start_time
    print(f"\nAll experiments completed!")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Results saved to logs/ directory")
    print(f"Models saved to models/ directory")

    print("\n📋 Next steps:")
    print("1. Analyze the logs/ directory for evaluation results")
    print("2. Compare performance across algorithms")
    print("3. Create your PDF report with the findings")

if __name__ == "__main__":
    main()