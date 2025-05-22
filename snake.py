import random

# Snake directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Reward/Penalty Constants
PENALTY_SELF_WALL_COLLISION = -100

class Snake:
    def __init__(self, snake_id=0, grid_width=30, grid_height=20, grid_size=20): # Added default fallbacks
        self.id = snake_id
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_size = grid_size 
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        self.body = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.grow = False
        self.is_alive = True

    def move(self): # Returns penalty if collision, 0 otherwise
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        
        # Calculate new head position based on current direction
        new_head_x = head_x + dir_x
        new_head_y = head_y + dir_y

        # Wall collision check (wrapping)
        if new_head_x >= self.grid_width:
            new_head_x = 0
        elif new_head_x < 0:
            new_head_x = self.grid_width - 1
        
        if new_head_y >= self.grid_height:
            new_head_y = 0
        elif new_head_y < 0:
            new_head_y = self.grid_height - 1
        
        new_head = (new_head_x, new_head_y)

        # Check for collision with itself
        if new_head in self.body[1:]: # Check against body excluding the current head before move
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
