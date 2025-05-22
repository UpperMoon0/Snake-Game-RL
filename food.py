import random

class Food:
    def __init__(self, grid_width=30, grid_height=20): # Added default fallbacks
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.position = self.randomize_position([])

    def randomize_position(self, snake_body):
        while True:
            pos = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if pos not in snake_body:
                return pos

    def get_position(self):
        return self.position
