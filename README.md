# Irenee_Dusingizimana_rl_summative

## Smart Farming Irrigation System RL Project

This project implements a reinforcement learning agent for optimizing irrigation in a simulated farming environment. The agent learns to water crops efficiently based on soil conditions, plant health, and weather patterns.

## Environment Diagram

The simulated environment consists of a 5x5 grid representing individual crop fields. Each grid cell contains:
- Soil moisture level (0-10 scale)
- Plant health status (0-10 scale)

The agent (represented as a red circle) navigates the grid and performs actions:
- Movement: Up, Down, Left, Right
- Watering: Increases moisture at current position

Weather conditions (Sunny, Rainy, Cloudy, Stormy) dynamically affect:
- Watering efficiency
- Evaporation rates
- Overall environmental dynamics

Color coding in visualization:
- Green intensity: Plant health
- Blue intensity: Soil moisture
- Red circle: Agent position

The goal is to maximize crop health while minimizing water usage through intelligent irrigation decisions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Irenee_Dusingizimana_rl_summative.git
   cd Irenee_Dusingizimana_rl_summative
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models

Train individual models:
```bash
python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 50000
python training/pg_training.py --algorithm reinforce --learning_rate 0.001 --gamma 0.99 --num_episodes 5000
python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 2048 --batch_size 64 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 50000 --no_save_model
python training/pg_training.py --algorithm a2c --learning_rate 0.0007 --n_steps 5 --gamma 0.99 --gae_lambda 1.0 --total_timesteps 50000 --no_save_model
```

### Automated Hyperparameter Tuning

Run all 40 experiments (10 per algorithm) automatically:
```bash
python run_experiments.py
```

This script will:
- Run all hyperparameter combinations sequentially
- **Save ONLY evaluation logs to `logs/` directory** (no model files for tuning)
- Show progress and estimated completion time
- Take approximately 1-2 hours to complete all experiments

**Note:** Hyperparameter tuning experiments do not save model files to avoid disk space waste. Only the evaluation logs are preserved for analysis.

### Manual Hyperparameter Tuning

Individual hyperparameter tuning is done by running the training scripts with different command-line arguments as shown in the tables below.

**For hyperparameter tuning experiments, add `--no_save_model` to avoid saving model files:**
```bash
python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 100000 --no_save_model
```

### Running the Best Model

After training, run the best performing model (default: PPO):
```bash
python main.py
```

### Random Actions Demonstration

Generate a static image of random actions:
```bash
python random_demo.py
```

## Hyperparameter Tuning

For each algorithm, run at least 10 different hyperparameter combinations. Use the command line arguments to specify parameters.

### DQN Hyperparameter Combinations

| Run | Learning Rate | Buffer Size | Batch Size | Gamma | Exploration Fraction | Total Timesteps |
|-----|---------------|-------------|------------|-------|-----------------------|-----------------|
| 1   | 0.001         | 10000       | 64         | 0.99  | 0.1                   | 100000          |
| 2   | 0.0001        | 10000       | 64         | 0.99  | 0.1                   | 100000          |
| 3   | 0.001         | 50000       | 64         | 0.99  | 0.1                   | 100000          |
| 4   | 0.001         | 10000       | 128        | 0.99  | 0.1                   | 100000          |
| 5   | 0.001         | 10000       | 64         | 0.95  | 0.1                   | 100000          |
| 6   | 0.001         | 10000       | 64         | 0.99  | 0.2                   | 100000          |
| 7   | 0.0005        | 20000       | 32         | 0.98  | 0.15                  | 100000          |
| 8   | 0.002         | 5000        | 128        | 0.99  | 0.05                  | 100000          |
| 9   | 0.001         | 10000       | 64         | 0.99  | 0.1                   | 100000          |
| 10  | 0.0001        | 50000       | 128        | 0.95  | 0.2                   | 100000          |

Example command:
```bash
python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 50000 --no_save_model
```

### REINFORCE Hyperparameter Combinations

| Run | Learning Rate | Gamma | Num Episodes |
|-----|---------------|-------|--------------|
| 1   | 0.001         | 0.99  | 10000        |
| 2   | 0.0001        | 0.99  | 10000        |
| 3   | 0.001         | 0.95  | 10000        |
| 4   | 0.0005        | 0.98  | 10000        |
| 5   | 0.002         | 0.99  | 10000        |
| 6   | 0.001         | 0.9   | 10000        |
| 7   | 0.0001        | 0.95  | 10000        |
| 8   | 0.0005        | 0.99  | 10000        |
| 9   | 0.001         | 0.99  | 10000        |
| 10  | 0.0002        | 0.98  | 10000        |

Example command:
```bash
python training/pg_training.py --algorithm reinforce --learning_rate 0.001 --gamma 0.99 --num_episodes 5000 --no_save_model
```

### PPO Hyperparameter Combinations

| Run | Learning Rate | N Steps | Batch Size | Gamma | GAE Lambda | Total Timesteps |
|-----|---------------|---------|------------|-------|------------|-----------------|
| 1   | 0.0003        | 2048    | 64         | 0.99  | 0.95       | 100000          |
| 2   | 0.0001        | 2048    | 64         | 0.99  | 0.95       | 100000          |
| 3   | 0.0003        | 1024    | 64         | 0.99  | 0.95       | 100000          |
| 4   | 0.0003        | 2048    | 128        | 0.99  | 0.95       | 100000          |
| 5   | 0.0003        | 2048    | 64         | 0.95  | 0.95       | 100000          |
| 6   | 0.0003        | 2048    | 64         | 0.99  | 0.9        | 100000          |
| 7   | 0.0007        | 4096    | 32         | 0.98  | 0.92       | 100000          |
| 8   | 0.001         | 1024    | 128        | 0.99  | 0.95       | 100000          |
| 9   | 0.0003        | 2048    | 64         | 0.99  | 0.95       | 100000          |
| 10  | 0.0005        | 2048    | 64         | 0.97  | 0.93       | 100000          |

Example command:
```bash
python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 2048 --batch_size 64 --gamma 0.99 --gae_lambda 0.95 --total_timesteps 100000
```

### A2C Hyperparameter Combinations

| Run | Learning Rate | N Steps | Gamma | GAE Lambda | Total Timesteps |
|-----|---------------|---------|-------|------------|-----------------|
| 1   | 0.0007        | 5       | 0.99  | 1.0        | 100000          |
| 2   | 0.0001        | 5       | 0.99  | 1.0        | 100000          |
| 3   | 0.0007        | 10      | 0.99  | 1.0        | 100000          |
| 4   | 0.0007        | 5       | 0.95  | 1.0        | 100000          |
| 5   | 0.0007        | 5       | 0.99  | 0.95       | 100000          |
| 6   | 0.001         | 5       | 0.99  | 1.0        | 100000          |
| 7   | 0.0005        | 8       | 0.98  | 0.97       | 100000          |
| 8   | 0.0007        | 5       | 0.99  | 1.0        | 100000          |
| 9   | 0.0002        | 10      | 0.95  | 0.9        | 100000          |
| 10  | 0.001         | 5       | 0.99  | 0.95       | 100000          |

Example command:
```bash
python training/pg_training.py --algorithm a2c --learning_rate 0.0007 --n_steps 5 --gamma 0.99 --gae_lambda 1.0 --total_timesteps 100000
```

## Project Structure

- `environment/`: Custom Gymnasium environment and rendering components
- `training/`: Training scripts for different RL algorithms
- `models/`:
  - `dqn/`: Saved DQN models
  - `pg/`: Policy gradient models
    - `reinforce_model.pth`: REINFORCE model
    - `ppo_model.zip`: PPO model
    - `a2c_model.zip`: A2C model
- `main.py`: Entry point for running the best performing agent
- `random_demo.py`: Demonstration of random actions in the environment
- `requirements.txt`: Project dependencies