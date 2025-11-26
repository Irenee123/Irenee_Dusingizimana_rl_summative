import pygame
import numpy as np

class FarmingRenderer:
    def __init__(self, env):
        self.env = env
        pygame.init()

        # Check if display is available
        if not pygame.display.get_init():
            raise Exception("Pygame display not available. Make sure you have a display environment.")

        try:
            self.screen = pygame.display.set_mode((600, 400))
            pygame.display.set_caption("Smart Farming Irrigation System")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            print("Pygame window created successfully!")
        except pygame.error as e:
            pygame.quit()
            raise Exception(f"Failed to create pygame window: {e}")

    def render(self):
        self.screen.fill((255, 255, 255))

        # Render grid
        grid_size = self.env.grid_size
        cell_size = 400 // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                x, y = i * cell_size, j * cell_size
                moisture = self.env.moisture[i, j]
                health = self.env.health[i, j]

                # Color: green for health, blue for moisture
                color = (int(255 * (1 - health/10)), int(255 * health/10), int(255 * moisture/10))
                pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))
                pygame.draw.rect(self.screen, (0,0,0), (x, y, cell_size, cell_size), 1)

        # Draw agent
        ax, ay = self.env.agent_pos
        pygame.draw.circle(self.screen, (255, 0, 0), (ax * cell_size + cell_size//2, ay * cell_size + cell_size//2), cell_size//4)

        # Display stats
        stats_x = 420
        weather_names = ['Sunny', 'Rainy', 'Cloudy', 'Stormy']
        weather = weather_names[self.env.weather]
        total_water = self.env.total_water_used
        healthy = np.sum(self.env.health > 7)
        steps = self.env.steps

        self.screen.blit(self.font.render(f"Weather: {weather}", True, (0,0,0)), (stats_x, 20))
        self.screen.blit(self.font.render(f"Water Used: {total_water:.1f}", True, (0,0,0)), (stats_x, 50))
        self.screen.blit(self.font.render(f"Healthy Plants: {healthy}", True, (0,0,0)), (stats_x, 80))
        self.screen.blit(self.font.render(f"Steps: {steps}", True, (0,0,0)), (stats_x, 110))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()