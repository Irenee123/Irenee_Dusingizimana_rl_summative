import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))

from custom_env import SmartFarmingEnv
from rendering import FarmingRenderer
import pygame
import numpy as np

def main():
    env = SmartFarmingEnv()
    renderer = FarmingRenderer(env)

    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < 50:  # Run for 50 steps
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        renderer.render()
        step += 1

    # Save screenshot
    pygame.image.save(renderer.screen, 'random_actions_demo.png')
    print("Saved random actions demo to random_actions_demo.png")
    print(f"Total water used: {env.total_water_used}")

    env.close()
    renderer.close()

if __name__ == "__main__":
    main()