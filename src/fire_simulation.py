import numpy as np


class FireSim:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Grid: 0 = Safe, 1-100 = Fire Intensity
        self.grid = np.zeros((height, width))
        self.wind_speed = 0.5
        self.wind_dir = (1, 0)  # Blowing Right

    def ignite_random(self):
        """Start a random fire for the demo"""
        cx, cy = np.random.randint(20, self.width - 20), np.random.randint(20, self.height - 20)
        # Create a fire blob
        for y in range(cy - 5, cy + 5):
            for x in range(cx - 5, cx + 5):
                self.grid[y, x] = np.random.randint(50, 100)

    def update(self):
        """Spread fire logic (Cellular Automata)"""
        new_grid = self.grid.copy()

        # Iterate through burning cells
        # (Optimization: In a real heavy sim we would use convolution, but loops work for small grids)
        rows, cols = np.where(self.grid > 10)

        for y, x in zip(rows, cols):
            # Burn out slowly
            new_grid[y, x] -= 1

            # Spread to neighbors with randomness
            if np.random.random() < 0.2:  # Spread chance
                # Wind bias (spread right more often)
                nx = x + np.random.choice([-1, 0, 1, 2])
                ny = y + np.random.choice([-1, 0, 1])

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Ignite neighbor
                    if new_grid[ny, nx] < 100:
                        new_grid[ny, nx] += 5

        self.grid = np.clip(new_grid, 0, 100)

    def apply_suppression(self, x, y, radius=10):
        """Water Tanker drops water here"""
        y_min = max(0, int(y - radius))
        y_max = min(self.height, int(y + radius))
        x_min = max(0, int(x - radius))
        x_max = min(self.width, int(x + radius))

        # Douse fire
        self.grid[y_min:y_max, x_min:x_max] *= 0.5  # Reduce by 50%