import argparse 
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

            snake.change_direction(action) # AI controls snake if not manual
            collision = snake.move()
            steps_since_last_food += 1
            reward = 0

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
            next_state = agent.get_state(snake, food)

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

def play_mode_game_loop(screen, clock, ai_q_learning_agent, num_ai_snakes=1):
    player_snake = Snake()
    food = Food()

    # Initialize AI snakes
    ai_snakes = []
    current_all_snake_segments = set(player_snake.get_body())

    for _ in range(num_ai_snakes):
        ai_snake_instance = Snake()
        while True:
            # Generate a random head position for the AI snake's first segment
            potential_head_pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            # For a new snake, its body is just its head. Check if this spot is free.
            if potential_head_pos not in current_all_snake_segments:
                ai_snake_instance.body = [potential_head_pos]
                ai_snake_instance.direction = random.choice([UP, DOWN, LEFT, RIGHT])
                
                for segment in ai_snake_instance.get_body(): # Should be just one segment initially
                    current_all_snake_segments.add(segment)
                
                ai_snakes.append(ai_snake_instance)
                break
    
    food.position = food.randomize_position(list(current_all_snake_segments))

    game_over = False
    player_score = 0

    font = pygame.font.Font(None, 36)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # Save Q-table if necessary or exit, for now just return
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w: player_snake.change_direction(UP)
                elif event.key == pygame.K_s: player_snake.change_direction(DOWN)
                elif event.key == pygame.K_a: player_snake.change_direction(LEFT)
                elif event.key == pygame.K_d: player_snake.change_direction(RIGHT)

        if player_snake.move():
            game_over = True

        if game_over:
            break

        # Check food collision for player
        if player_snake.get_head_position() == food.get_position():
            player_snake.grow_snake()
            player_score += 1
            
            temp_occupied_cells = set(player_snake.get_body())
            for s_ai in ai_snakes:
                for seg in s_ai.get_body():
                    temp_occupied_cells.add(seg)
            food.position = food.randomize_position(list(temp_occupied_cells))

        # Move AI snakes & check collisions
        active_ai_snakes = []
        for ai_snake in ai_snakes:
            if ai_q_learning_agent and ai_q_learning_agent.q_table and len(ai_q_learning_agent.q_table) > 0:
                state = ai_q_learning_agent.get_state(ai_snake, food)
                action = ai_q_learning_agent.choose_action(state) # Agent is in exploitation mode
                ai_snake.change_direction(action)
            else: # Fallback to random movement if no Q-table or agent
                if random.random() < 0.2: # 20% chance to change direction
                    ai_snake.change_direction(random.choice([UP, DOWN, LEFT, RIGHT]))
            
            ai_snake_collided_self_or_wall = ai_snake.move()

            if not ai_snake_collided_self_or_wall:
                active_ai_snakes.append(ai_snake)

                # Check collisions between this live AI and player
                if player_snake.get_head_position() in ai_snake.get_body(): # Player head vs AI body
                    game_over = True
                    break 
                if ai_snake.get_head_position() in player_snake.get_body(): # AI head vs player body
                    game_over = True
                    break
            # If ai_snake_collided_self_or_wall is True, the AI snake is not added to active_ai_snakes,
            # effectively removing it from the game.
        
        ai_snakes = active_ai_snakes
        if game_over:
            break

        # Draw everything
        screen.fill(BLACK)
        draw_snake(screen, player_snake)
        for ai_snake_to_draw in ai_snakes:
            draw_snake(screen, ai_snake_to_draw)
        draw_food(screen, food)
        
        # Display score
        score_text_surface = font.render(f"Score: {player_score}", True, WHITE)
        screen.blit(score_text_surface, (10, 10))
        
        pygame.display.flip()
        clock.tick(10)

    print(f"Game Over! Your score: {player_score}")
    time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake RL Agent")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from the last saved model (q_table.npy or q_table_final.npy if the former is not found).")
    parser.add_argument("--demo", action="store_true", help="Run in demonstration mode using q_table_final.npy. If combined with training, demo runs after training.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes for training.")
    parser.add_argument("--play", action="store_true", help="Play the game against AI snakes.")
    parser.add_argument("--ais", type=int, default=1, help="Number of AI snakes when in --play mode.")

    args = parser.parse_args()

    if args.play and (args.continue_training):
        print("Error: --play mode cannot be used with --continue-training simultaneously.")
        exit()
    if args.play and args.episodes != parser.get_default("episodes"):
        print("Warning: --episodes is ignored in --play mode.")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake RL")
    clock = pygame.time.Clock()

    agent = QLearningAgent()
    ai_agent_for_play_mode = None

    q_table_final_path = "q_table_final.npy"
    q_table_periodic_path = "q_table.npy"
    rewards = []
    model_loaded_for_training_continuation = False

    # --- Determine Action: Train, Continue, or Prepare for Demo ---
    needs_training = False

    if args.play:
        print(f"--- Player vs. AI Mode ({args.ais} AI(s)) ---")
        ai_agent_for_play_mode = QLearningAgent(exploration_rate=0.0, min_exploration_rate=0.0) # Ensure exploitation
        try:
            print(f"Loading model for AI snakes from {q_table_final_path}...")
            ai_agent_for_play_mode.q_table = np.load(q_table_final_path, allow_pickle=True).item()
            if not ai_agent_for_play_mode.q_table:
                 print(f"WARNING: Loaded Q-table from {q_table_final_path} is empty. AI snakes will use random behavior.")
            else:
                print("Model loaded successfully for AI snakes.")
        except FileNotFoundError:
            print(f"WARNING: {q_table_final_path} not found. AI snakes will use random behavior.")
            ai_agent_for_play_mode.q_table = {} 
        except Exception as e:
            print(f"ERROR: Could not load {q_table_final_path} for AI snakes: {e}. AI snakes will use random behavior.")
            ai_agent_for_play_mode.q_table = {}
        
        play_mode_game_loop(screen, clock, ai_agent_for_play_mode, num_ai_snakes=args.ais)
    elif args.continue_training:
        print("Attempting to continue training...")
        loaded_from_periodic = False
        try:
            agent.q_table = np.load(q_table_periodic_path, allow_pickle=True).item()
            print(f"Loaded Q-table from periodic save: {q_table_periodic_path} to continue training.")
            model_loaded_for_training_continuation = True
            loaded_from_periodic = True
        except FileNotFoundError:
            print(f"Periodic save {q_table_periodic_path} not found.")
        except Exception as e:
            print(f"Error loading periodic Q-table {q_table_periodic_path}: {e}.")

        if not model_loaded_for_training_continuation:
            try:
                agent.q_table = np.load(q_table_final_path, allow_pickle=True).item()
                # If continuing from a final model, epsilon might need adjustment if it was at min_epsilon
                if agent.epsilon <= agent.min_epsilon: # Give it a bit more exploration if it was fully trained
                    agent.epsilon = 0.1
                print(f"Loaded Q-table from final save: {q_table_final_path} to continue training.")
                model_loaded_for_training_continuation = True
                needs_training = True
            except FileNotFoundError:
                print(f"Final save {q_table_final_path} not found for continuation.")
            except Exception as e:
                print(f"Error loading final Q-table {q_table_final_path} for continuation: {e}.")

        if not model_loaded_for_training_continuation:
            print("No existing model found to continue. Starting new training session.")
            agent.q_table = {}
            agent.epsilon = 1.0
            needs_training = True
        else:
            print("Continuing training with loaded model.")

    elif not args.demo: # If not continuing and not demo-only, it's a new training session
        print("Starting new training session.")
        agent.q_table = {}
        agent.epsilon = 1.0
        needs_training = True
    
    # If only --demo is specified, needs_training remains False here.

    if needs_training:
        print(f"--- Training Phase (UI Disabled) for {args.episodes} episodes ---")
        # Training always has graphics disabled
        current_rewards, q_table_from_training = game_loop(screen, clock, agent, num_episodes=args.episodes, training_mode=True, display_game=False)
        rewards.extend(current_rewards) # Accumulate rewards if multiple training sessions happen (though current logic implies one)
        print("Training phase complete.")
        # Q-table is saved within game_loop (q_table_final.npy at end, q_table.npy periodically)
        # agent.q_table is updated by game_loop

    # --- Demonstration Phase ---
    if args.demo:
        print("\n--- Demonstration Phase (UI Enabled) ---")
        model_available_for_demo = False
        if needs_training: # True if training was run in this session
            if agent.q_table and len(agent.q_table) > 0:
                print("Using model from current training session for demonstration.")
                model_available_for_demo = True
            else:
                print("Warning: Training was run, but no Q-table is available in the agent from this session. Attempting to load from file.")
        
        if not model_available_for_demo:
            try:
                print(f"Loading model for demonstration from {q_table_final_path}...")
                agent.q_table = np.load(q_table_final_path, allow_pickle=True).item()
                if not agent.q_table or len(agent.q_table) == 0:
                    print(f"ERROR: Loaded Q-table from {q_table_final_path} is empty.")
                else:
                    print("Model loaded successfully for demonstration.")
                    model_available_for_demo = True
            except FileNotFoundError:
                print(f"ERROR: {q_table_final_path} not found. Cannot run demonstration without a trained model.")
            except Exception as e:
                print(f"ERROR: Could not load {q_table_final_path} for demonstration: {e}")
        
        if model_available_for_demo and agent.q_table and len(agent.q_table) > 0:
            agent.epsilon = 0 # No exploration for demo
            game_loop(screen, clock, agent, num_episodes=5, training_mode=False, display_game=True)
        else:
            print("No model available to run demonstration.")
    elif not needs_training and not args.demo and not args.play:
        print("No action specified (e.g., --train, --continue-training, --demo, --play). Exiting.")

    pygame.quit()
    print("Game closed.")

    # Optional: Plot rewards if you have matplotlib
    if rewards:
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
