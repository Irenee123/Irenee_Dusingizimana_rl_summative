import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))

from stable_baselines3 import DQN
from custom_env import SmartFarmingEnv
from rendering import FarmingRenderer
import time
import pygame

def main():
    # Load the best performing model (using DQN - most stable)
    model_path = "./models/dqn/dqn_model"
    if not os.path.exists(model_path + ".zip"):
        print("Model not found. Please train a model first.")
        return

    print("Loading DQN model...")
    model = DQN.load(model_path)
    print("Model loaded successfully!")

    env = SmartFarmingEnv()
    print("Environment created!")

    try:
        renderer = FarmingRenderer(env)
        print("Renderer initialized - pygame window should appear!")
    except Exception as e:
        print(f"Error initializing renderer: {e}")
        print("This might be due to pygame display issues in terminal environment.")
        print("Try running from a Python IDE or ensure you have a display available.")
        return

    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("Starting visualization. Close the pygame window to stop.")

    while not done and steps < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        renderer.render()
        time.sleep(0.5)  # Slow down for visibility

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

    # Keep window open until user closes it
    print("Episode finished. Close the pygame window to exit.")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
                break
        time.sleep(0.1)

    pygame.quit()

    print(f"Total reward: {total_reward}, Steps: {steps}")
    env.close()
    renderer.close()

if __name__ == "__main__":
    main()