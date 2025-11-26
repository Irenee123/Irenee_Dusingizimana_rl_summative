import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import sys

class SmartFarmingEnv(gym.Env):
    def __init__(self, grid_size=5, max_steps=100):
        super(SmartFarmingEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Actions: 0=up, 1=down, 2=left, 3=right, 4=water
        self.action_space = spaces.Discrete(5)

        # Observation: position (2), moisture grid (25), health grid (25), weather (4)
        obs_size = 2 + grid_size*grid_size * 2 + 4
        self.observation_space = spaces.Box(low=0, high=10, shape=(obs_size,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([self.grid_size//2, self.grid_size//2])
        self.moisture = np.random.uniform(3, 7, (self.grid_size, self.grid_size))
        self.health = np.random.uniform(4, 8, (self.grid_size, self.grid_size))
        self.weather = np.random.choice(4)  # 0=sunny, 1=rainy, 2=cloudy, 3=stormy
        self.steps = 0
        self.total_water_used = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.agent_pos.astype(np.float32)
        moisture_flat = self.moisture.flatten().astype(np.float32)
        health_flat = self.health.flatten().astype(np.float32)
        weather_onehot = np.zeros(4, dtype=np.float32)
        weather_onehot[self.weather] = 1.0
        return np.concatenate([pos, moisture_flat, health_flat, weather_onehot])

    def step(self, action):
        self.steps += 1
        reward = 0

        # Movement
        if action < 4:
            if action == 0:  # up
                self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
            elif action == 1:  # down
                self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
            elif action == 2:  # left
                self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            elif action == 3:  # right
                self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        else:  # water
            x, y = self.agent_pos
            water_amount = 2.0
            if self.weather == 1:  # rainy
                water_amount *= 0.5
            elif self.weather == 3:  # stormy
                water_amount *= 1.5
            self.moisture[x, y] = min(10, self.moisture[x, y] + water_amount)
            self.total_water_used += water_amount
            reward -= 0.1 * water_amount  # cost of water

        # Update environment
        self._update_environment()

        # Calculate reward
        healthy_plants = np.sum(self.health > 7)
        reward += healthy_plants * 0.1

        overwatered = np.sum(self.moisture > 9)
        reward -= overwatered * 0.5

        # Terminal
        done = self.steps >= self.max_steps or np.all(self.health > 7)
        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    def _update_environment(self):
        # Evaporation based on weather
        evap_rate = [0.3, 0.1, 0.2, 0.5][self.weather]  # sunny, rainy, cloudy, stormy
        self.moisture = np.maximum(0, self.moisture - evap_rate)

        # Health update based on moisture
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if 4 <= self.moisture[i,j] <= 8:
                    self.health[i,j] = min(10, self.health[i,j] + 0.1)
                elif self.moisture[i,j] < 2:
                    self.health[i,j] = max(0, self.health[i,j] - 0.2)
                elif self.moisture[i,j] > 9:
                    self.health[i,j] = max(0, self.health[i,j] - 0.1)

        # Random weather change occasionally
        if np.random.rand() < 0.1:
            self.weather = np.random.choice(4)

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((400, 400))
            pygame.display.set_caption("Smart Farming Irrigation")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        cell_size = 400 // self.grid_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x, y = i * cell_size, j * cell_size
                moisture = self.moisture[i, j]
                health = self.health[i, j]

                # Color based on health and moisture
                color = (int(255 * (1 - health/10)), int(255 * health/10), int(255 * moisture/10))
                pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))

                # Draw grid lines
                pygame.draw.rect(self.screen, (0,0,0), (x, y, cell_size, cell_size), 1)

        # Draw agent
        ax, ay = self.agent_pos
        pygame.draw.circle(self.screen, (255, 0, 0), (ax * cell_size + cell_size//2, ay * cell_size + cell_size//2), cell_size//4)

        pygame.display.flip()
        self.clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()