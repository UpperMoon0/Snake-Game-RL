from matplotlib import pyplot as plt
import pygame
import numpy as np
import random
import time

# Screen dimensions
WIDTH = 600
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Snake directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.grow = False

    def move(self):
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = ((head_x + dir_x) % GRID_WIDTH, (head_y + dir_y) % GRID_HEIGHT)

        if new_head in self.body[1:]: # Check for collision with itself
            return True # Game Over

        self.body.insert(0, new_head)

        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
        return False # No collision

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

class Food:
    def __init__(self):
        self.position = self.randomize_position([])

    def randomize_position(self, snake_body):
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in snake_body:
                return pos

    def get_position(self):
        return self.position

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01):
        self.q_table = {}  # (state) -> [action_values]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay_rate
        self.min_epsilon = min_exploration_rate
        self.actions = [UP, DOWN, LEFT, RIGHT] # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

    def get_state(self, snake, food):
        head_x, head_y = snake.get_head_position()
        food_x, food_y = food.get_position()

        # Relative position of food
        food_dx = food_x - head_x
        food_dy = food_y - head_y

        # Normalize food_dx and food_dy to be -1, 0, or 1
        norm_food_dx = np.sign(food_dx)
        norm_food_dy = np.sign(food_dy)


        # Danger ahead, left, right (relative to snake's current direction)
        # 0: No danger, 1: Danger (wall or self)
        danger_straight = self._check_danger(snake, snake.direction)
        danger_left = self._check_danger(snake, self._get_relative_left(snake.direction))
        danger_right = self._check_danger(snake, self._get_relative_right(snake.direction))


        state = (
            # Food location
            norm_food_dx, norm_food_dy,

            # Snake direction (one-hot encoded)
            snake.direction == UP,
            snake.direction == DOWN,
            snake.direction == LEFT,
            snake.direction == RIGHT,

            # Danger
            danger_straight,
            danger_left,
            danger_right,

            # Tail nearby (experimental) - check if tail is in adjacent cells
            self._is_tail_near(snake, (head_x + 1, head_y)), # Right
            self._is_tail_near(snake, (head_x - 1, head_y)), # Left
            self._is_tail_near(snake, (head_x, head_y + 1)), # Down
            self._is_tail_near(snake, (head_x, head_y - 1)), # Up
        )
        return state

    def _is_tail_near(self, snake, position):
        # Check if the given position is occupied by the snake's tail (excluding the head)
        # And also ensure it's not the cell immediately after the head if the snake is short
        if len(snake.body) < 2:
            return False
        return position in snake.body[1:]


    def _get_relative_left(self, direction):
        if direction == UP: return LEFT
        if direction == DOWN: return RIGHT
        if direction == LEFT: return DOWN
        if direction == RIGHT: return UP
        return direction # Should not happen

    def _get_relative_right(self, direction):
        if direction == UP: return RIGHT
        if direction == DOWN: return LEFT
        if direction == LEFT: return UP
        if direction == RIGHT: return DOWN
        return direction # Should not happen


    def _check_danger(self, snake, direction_vector):
        head_x, head_y = snake.get_head_position()
        next_x = (head_x + direction_vector[0])
        next_y = (head_y + direction_vector[1])

        # Check wall collision
        if not (0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT):
            return 1 # Danger: wall

        # Check self collision
        if (next_x, next_y) in snake.body:
            return 1 # Danger: self
        return 0 # No danger

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            # Exploit: choose the action with the highest Q-value
            # Ensure state is in q_table, initialize if not
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(self.actions))
            q_values = self.q_table.get(state, np.zeros(len(self.actions)))
            return self.actions[np.argmax(q_values)] # Exploit

    def update_q_table(self, state, action, reward, next_state):
        # Ensure states are in q_table, initialize if not
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        action_index = self.actions.index(action)
        old_value = self.q_table[state][action_index]
        next_max = np.max(self.q_table[next_state])

        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action_index] = new_value

    def decay_exploration(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def draw_grid(surface):
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(surface, WHITE, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(surface, WHITE, (0, y), (WIDTH, y))

def draw_snake(surface, snake):
    for segment in snake.get_body():
        pygame.draw.rect(surface, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def draw_food(surface, food):
    pygame.draw.rect(surface, RED, (food.get_position()[0] * GRID_SIZE, food.get_position()[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def game_loop(screen, clock, agent, num_episodes=1000, training_mode=True, display_game=True):
    episode_rewards = []
    max_score = 0

    for episode in range(num_episodes):
        snake = Snake()
        food = Food()
        food.position = food.randomize_position(snake.get_body()) # Ensure food is not on snake initially
        game_over = False
        current_reward = 0
        score = 0
        steps_since_last_food = 0
        max_steps_without_food = GRID_WIDTH * GRID_HEIGHT * 2 # Heuristic for ending stuck episodes

        start_time = time.time()

        while not game_over:
            state = agent.get_state(snake, food)
            action = agent.choose_action(state)

            # For human play or direct control (not used in RL training directly here)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return episode_rewards, agent.q_table # Or save q_table
                # Manual controls (optional, can be removed for pure RL)
                # if event.type == pygame.KEYDOWN:
                #     if event.key == pygame.K_UP: snake.change_direction(UP)
                #     elif event.key == pygame.K_DOWN: snake.change_direction(DOWN)
                #     elif event.key == pygame.K_LEFT: snake.change_direction(LEFT)
                #     elif event.key == pygame.K_RIGHT: snake.change_direction(RIGHT)

            snake.change_direction(action) # AI controls the snake
            collision = snake.move()
            steps_since_last_food += 1
            reward = 0 # Default reward

            if collision:
                reward = -100  # Penalty for collision
                game_over = True
            elif snake.get_head_position() == food.get_position():
                reward = 50  # Reward for eating food
                snake.grow_snake()
                food.position = food.randomize_position(snake.get_body())
                score += 1
                steps_since_last_food = 0
            else:
                # Small penalty for each step to encourage efficiency
                # reward = -0.1
                # Reward for getting closer to food, penalty for moving away
                old_dist_to_food = abs(state[0]) + abs(state[1]) # Using normalized distance from state
                new_head_pos = snake.get_head_position()
                new_dist_to_food = abs(np.sign(food.get_position()[0] - new_head_pos[0])) + \
                                   abs(np.sign(food.get_position()[1] - new_head_pos[1]))

                if new_dist_to_food < old_dist_to_food:
                    reward = 1 # Small reward for moving closer
                elif new_dist_to_food > old_dist_to_food:
                    reward = -1.5 # Penalty for moving further
                else:
                    reward = -0.5 # Penalty for not moving closer or further (e.g. parallel)


            # End episode if snake is stuck in a loop or takes too long
            if steps_since_last_food > max_steps_without_food:
                reward = -75 # Penalty for being stuck
                game_over = True


            current_reward += reward
            next_state = agent.get_state(snake, food) # Get new state after action

            if training_mode:
                agent.update_q_table(state, action, reward, next_state)

            if display_game or not training_mode: # Display if not training or if specifically told to
                screen.fill(BLACK)
                # draw_grid(screen) # Optional: draw grid lines
                draw_snake(screen, snake)
                draw_food(screen, food)
                pygame.display.flip()
                clock.tick(15 if training_mode else 10) # Faster for training, slower for viewing

            if game_over and training_mode:
                agent.decay_exploration()


        episode_rewards.append(current_reward)
        if score > max_score:
            max_score = score

        if training_mode and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} - Score: {score}, Max Score: {max_score}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Time: {time.time() - start_time:.2f}s")
            # Save Q-table periodically
            if (episode + 1) % 500 == 0:
                 np.save("q_table.npy", agent.q_table)
                 print(f"Q-table saved at episode {episode+1}")


        if not training_mode and game_over:
            print(f"Game Over! Score: {score}")
            # Wait a bit before restarting or closing
            time.sleep(2)


    if training_mode:
        print("Training finished.")
        np.save("q_table_final.npy", agent.q_table)
        print("Final Q-table saved as q_table_final.npy")
    return episode_rewards, agent.q_table


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake RL")
    clock = pygame.time.Clock()

    agent = QLearningAgent()

    # --- Model Loading/Training Choice ---
    q_table_path = "q_table_final.npy"
    training_needed = False
    rewards = [] # Initialize rewards

    # Check if a pre-trained model exists to offer the choice
    can_load_model = False
    try:
        with open(q_table_path, 'rb') as f: # Try opening to check existence
            can_load_model = True
    except FileNotFoundError:
        pass # Model doesn't exist, will train new

    if can_load_model:
        load_model_choice = input(f"Load pre-trained model from '{q_table_path}'? (yes/no): ").strip().lower()
        if load_model_choice == 'yes':
            try:
                agent.q_table = np.load(q_table_path, allow_pickle=True).item()
                agent.epsilon = agent.min_epsilon # Start with low exploration if loaded
                print(f"Loaded Q-table from {q_table_path}")
                training_needed = False
            except Exception as e:
                print(f"Error loading Q-table: {e}. Training a new model.")
                training_needed = True # Ensure training if loading failed
        else:
            print("Opted to train a new model.")
            training_needed = True
    else:
        print(f"No pre-trained model found at '{q_table_path}'. Training a new model.")
        training_needed = True


    if training_needed:
        # --- Training Phase ---
        print("Starting Training...")
        # To load a partially trained Q-table (e.g., q_table.npy) if you want to resume:
        # try:
        #     agent.q_table = np.load("q_table.npy", allow_pickle=True).item()
        #     print("Resuming training from q_table.npy")
        # except FileNotFoundError:
        #     print("No q_table.npy found, starting fresh training.")

        training_episodes = 2000 # Adjust as needed
        # Set display_game to False during long training sessions for speed.
        # rewards list is populated by game_loop
        rewards, q_table = game_loop(screen, clock, agent, num_episodes=training_episodes, training_mode=True, display_game=False)
        # Saving of q_table_final.npy is handled within game_loop at the end of training
    # else:
    # rewards list remains empty if no training occurred

    # --- Testing/Demonstration Phase ---
    # Ensure q_table is loaded if not trained
    if not training_needed and not agent.q_table: # Should be loaded if training_needed is False
        print("Error: Model was supposed to be loaded but Q-table is empty. Check loading logic.")
        # As a fallback, try loading again or exit
        try:
            agent.q_table = np.load(q_table_path, allow_pickle=True).item()
            agent.epsilon = agent.min_epsilon
            print(f"Re-loaded Q-table from {q_table_path} for demonstration.")
        except Exception as e:
            print(f"Could not load Q-table for demonstration: {e}. Exiting.")
            pygame.quit()
            exit()


    print("\\nStarting Demonstration (using learned Q-table)...")
    agent.epsilon = 0 # Turn off exploration for demonstration
    game_loop(screen, clock, agent, num_episodes=5, training_mode=False, display_game=True) # Run a few games to demonstrate

    pygame.quit()
    print("Game closed.")

    # Optional: Plot rewards if you have matplotlib
    # Only plot if training occurred and rewards list is not empty
    if training_needed and rewards:
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Rewards over Time")
        # Calculate and plot a moving average
        moving_avg_window = 100
        if len(rewards) >= moving_avg_window:
            moving_avg = np.convolve(rewards, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            plt.plot(np.arange(moving_avg_window -1, len(rewards)), moving_avg, label=f'Moving Average ({moving_avg_window} episodes)')
        plt.legend()
        plt.show()
