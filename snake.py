import random

# Screen dimensions
WIDTH = 600
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Snake directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Reward/Penalty Constants
PENALTY_SELF_WALL_COLLISION = -100

class Snake:
    def __init__(self, snake_id=0):
        self.id = snake_id
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.grow = False
        self.is_alive = True

    def move(self): # Returns penalty if collision, 0 otherwise
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = ((head_x + dir_x) % GRID_WIDTH, (head_y + dir_y) % GRID_HEIGHT)

        # Check wall collision
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            self.is_alive = False
            return PENALTY_SELF_WALL_COLLISION

        # Check for collision with itself
        if new_head in self.body[1:]:
            self.is_alive = False
            return PENALTY_SELF_WALL_COLLISION

        self.body.insert(0, new_head)

        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
        return 0 # No collision from this move

    def change_direction(self, new_direction):
        # Prevent snake from reversing directly
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            return
        self.direction = new_direction

    def grow_snake(self):
        self.grow = True

    def get_head_position(self):
        return self.body[0]

    def get_body(self):
        return self.body
