import argparse 
from matplotlib import pyplot as plt
import pygame
import numpy as np
import random
import time
from snake import Snake  # Import Snake class
from food import Food    # Import Food class

# Screen dimensions
WIDTH = 600
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Make these global so they can be updated if the screen is resized
global global_WIDTH, global_HEIGHT, global_GRID_WIDTH, global_GRID_HEIGHT, global_GRID_SIZE
global_WIDTH = WIDTH
global_HEIGHT = HEIGHT
global_GRID_WIDTH = GRID_WIDTH
global_GRID_HEIGHT = GRID_HEIGHT
global_GRID_SIZE = GRID_SIZE

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

# Reward/Penalty Constants
PENALTY_SELF_WALL_COLLISION = -100
REWARD_EAT_FOOD = 50
PENALTY_STUCK = -75
REWARD_MOVE_CLOSER_TO_FOOD = 1
PENALTY_MOVE_AWAY_FROM_FOOD = -1.5 # More severe penalty for moving away
PENALTY_MOVE_PARALLEL_FOOD = -0.5 # Small penalty for not making progress towards food
PENALTY_HEAD_HITS_OTHER_BODY = -150 # Increased penalty
REWARD_OTHER_HEAD_HITS_MY_BODY = 100  # Increased reward
PENALTY_HEAD_TO_HEAD_COLLISION = -200 # Increased penalty

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01):
        self.q_table = {}  # (state) -> [action_values]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay_rate
        self.min_epsilon = min_exploration_rate
        self.actions = [UP, DOWN, LEFT, RIGHT] # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

    def get_state(self, snake, food, all_snakes):
        head_x, head_y = snake.get_head_position()
        food_x, food_y = food.get_position()


        # Relative position of food
        food_dx = food_x - head_x
        food_dy = food_y - head_y

        # Normalize food_dx and food_dy to be -1, 0, or 1
        norm_food_dx = np.sign(food_dx)
        norm_food_dy = np.sign(food_dy)

        # Danger from self/wall
        # 0: No danger, 1: Danger (wall or self)
        danger_straight_self_wall = self._check_danger_self_wall(snake, snake.direction)
        danger_left_self_wall = self._check_danger_self_wall(snake, self._get_relative_left(snake.direction))
        danger_right_self_wall = self._check_danger_self_wall(snake, self._get_relative_right(snake.direction))

        # Danger from other snakes
        danger_straight_other = self._check_danger_other_snake(snake, snake.direction, all_snakes)
        danger_left_other = self._check_danger_other_snake(snake, self._get_relative_left(snake.direction), all_snakes)
        danger_right_other = self._check_danger_other_snake(snake, self._get_relative_right(snake.direction), all_snakes)

        state = (
            # Food location
            norm_food_dx, norm_food_dy,

            # Snake direction (one-hot encoded)
            snake.direction == UP,
            snake.direction == DOWN,
            snake.direction == LEFT,
            snake.direction == RIGHT,

            # Danger
            danger_straight_self_wall,
            danger_left_self_wall,
            danger_right_self_wall,

            # Danger from other snakes
            danger_straight_other,
            danger_left_other,
            danger_right_other,

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

    def _check_danger_self_wall(self, snake, direction_vector):
        head_x, head_y = snake.get_head_position()
        next_x = (head_x + direction_vector[0])
        next_y = (head_y + direction_vector[1])

        # Check wall collision
        if not (0 <= next_x < global_GRID_WIDTH and 0 <= next_y < global_GRID_HEIGHT):
            return 1 # Danger: wall

        # Check self collision
        if (next_x, next_y) in snake.body:
            return 1 # Danger: self
        return 0 # No danger
    
    def _check_danger_other_snake(self, current_snake, direction_vector, all_snakes):
        head_x, head_y = current_snake.get_head_position()
        next_x = (head_x + direction_vector[0])
        next_y = (head_y + direction_vector[1])
        # No need to check wall here, as _check_danger_self_wall handles it.
        # We only care if the next step is into another snake.

        for other_s in all_snakes:
            if other_s.id == current_snake.id or not other_s.is_alive:
                continue
            if (next_x, next_y) in other_s.get_body(): # Check collision with any part of other snake
                return 1 # Danger: other snake
        return 0 # No danger from other snakes in this direction

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

def game_loop(screen, clock, agent, num_episodes=1000, training_mode=True, display_game=True, num_ai_snakes_to_train=1):
    total_rewards_all_episodes = []
    max_avg_score_overall = 0 # Track max average score across episodes

    for episode in range(num_episodes):
        snakes = [Snake(snake_id=i) for i in range(num_ai_snakes_to_train)]
        food = Food()

        occupied_starts = set()
        for s_obj in snakes:
            while True:
                start_pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                if start_pos not in occupied_starts:
                    s_obj.body = [start_pos]
                    s_obj.direction = random.choice([UP, DOWN, LEFT, RIGHT])
                    occupied_starts.add(start_pos)
                    break
        
        all_initial_bodies = []
        for s_obj in snakes:
            all_initial_bodies.extend(s_obj.get_body())
        food.position = food.randomize_position(all_initial_bodies)

        current_episode_rewards_per_snake = [0] * num_ai_snakes_to_train
        scores_per_snake = [0] * num_ai_snakes_to_train
        steps_since_last_food_per_snake = [0] * num_ai_snakes_to_train
        
        last_state_action = {} 

        start_time = time.time()
        episode_steps = 0
        # Heuristic timeout, scales with number of snakes and grid size
        max_episode_steps = (GRID_WIDTH * GRID_HEIGHT * num_ai_snakes_to_train) * 1.5 
        if num_ai_snakes_to_train == 1: # Longer for single snake
             max_episode_steps = GRID_WIDTH * GRID_HEIGHT * 2.5

        max_steps_without_food = GRID_WIDTH * GRID_HEIGHT # Per snake

        while True: # Inner loop for steps in an episode
            episode_steps += 1
            active_snakes_this_step = [s for s in snakes if s.is_alive]

            # Episode termination conditions
            if not active_snakes_this_step: # All snakes died
                break
            if num_ai_snakes_to_train > 1 and len(active_snakes_this_step) <= 1: # Winner found or all but one died
                break
            if episode_steps > max_episode_steps: # Timeout
                # Apply a penalty to any snakes still alive if timeout is considered bad
                # For now, just end.
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    # Save Q-table if training_mode?
                    if training_mode: np.save("q_table_emergency_exit.npy", agent.q_table)
                    return total_rewards_all_episodes, agent.q_table

            # 1. Get actions for all live snakes
            old_head_positions = {} # For food distance reward calculation
            for s_obj in active_snakes_this_step:
                state = agent.get_state(s_obj, food, snakes)
                action = agent.choose_action(state)
                last_state_action[s_obj.id] = (state, action)
                s_obj.change_direction(action)
                old_head_positions[s_obj.id] = s_obj.get_head_position()

            # 2. Move snakes and collect self/wall collision penalties
            step_rewards = {s.id: 0 for s in snakes}
            for s_obj in active_snakes_this_step:
                penalty_from_move = s_obj.move() # move() now sets is_alive and returns penalty
                if penalty_from_move != 0:
                    step_rewards[s_obj.id] += penalty_from_move

            # 3. Calculate rewards and handle consequences
            food_eaten_this_step = False
            # Check for food eaten (only live snakes after move)
            for s_obj in snakes:
                if not s_obj.is_alive: continue
                if s_obj.get_head_position() == food.get_position():
                    step_rewards[s_obj.id] += REWARD_EAT_FOOD
                    s_obj.grow_snake()
                    scores_per_snake[s_obj.id] += 1
                    steps_since_last_food_per_snake[s_obj.id] = 0
                    food_eaten_this_step = True
                    
                    all_current_bodies = []
                    for s_k_food_avoid in snakes: # Use all snakes (alive or not) to avoid spawning food on them
                        all_current_bodies.extend(s_k_food_avoid.get_body())
                    food.position = food.randomize_position(all_current_bodies)
                    break 

            # Inter-snake collisions (only among snakes still alive)
            live_snakes_for_combat = [s for s in snakes if s.is_alive]

            for i in range(len(live_snakes_for_combat)):
                s1_obj = live_snakes_for_combat[i]
                if not s1_obj.is_alive: continue # Already died (e.g. self-collision)

                s1_head = s1_obj.get_head_position()
                s1_body_set = set(s1_obj.get_body()[1:])

                for j in range(i + 1, len(live_snakes_for_combat)):
                    s2_obj = live_snakes_for_combat[j]
                    if not s2_obj.is_alive: continue

                    s2_head = s2_obj.get_head_position()
                    s2_body_set = set(s2_obj.get_body()[1:])

                    # Head-to-head
                    if s1_head == s2_head:
                        step_rewards[s1_obj.id] += PENALTY_HEAD_TO_HEAD_COLLISION
                        step_rewards[s2_obj.id] += PENALTY_HEAD_TO_HEAD_COLLISION
                        s1_obj.is_alive = False
                        s2_obj.is_alive = False
                        # Both are out, continue checking other pairs if any
                        break # s1_obj is done for this inner loop

                    # S1 head hits S2 body
                    if s1_head in s2_body_set:
                        step_rewards[s1_obj.id] += PENALTY_HEAD_HITS_OTHER_BODY
                        step_rewards[s2_obj.id] += REWARD_OTHER_HEAD_HITS_MY_BODY 
                        s1_obj.is_alive = False
                        # s1 is out, break from inner loop for s1, continue outer loop
                        break 

                    # S2 head hits S1 body
                    if s2_head in s1_body_set:
                        step_rewards[s2_obj.id] += PENALTY_HEAD_HITS_OTHER_BODY
                        step_rewards[s1_obj.id] += REWARD_OTHER_HEAD_HITS_MY_BODY
                        s2_obj.is_alive = False
                        # s2 is out, s1 might still be alive for other collisions in this step
                        # No break here, s1 might collide with s3, etc.
            
            # Step penalties/rewards for distance to food & stuck penalty (for snakes still alive)
            for s_obj in snakes:
                if not s_obj.is_alive: continue

                if not food_eaten_this_step and s_obj.id in old_head_positions:
                    old_head_x, old_head_y = old_head_positions[s_obj.id]
                    food_x, food_y = food.get_position()
                    
                    old_dist_to_food = abs(food_x - old_head_x) + abs(food_y - old_head_y)
                    new_head_pos = s_obj.get_head_position()
                    new_dist_to_food = abs(food_x - new_head_pos[0]) + abs(food_y - new_head_pos[1])

                    if new_dist_to_food < old_dist_to_food:
                        step_rewards[s_obj.id] += REWARD_MOVE_CLOSER_TO_FOOD
                    elif new_dist_to_food > old_dist_to_food:
                        step_rewards[s_obj.id] += PENALTY_MOVE_AWAY_FROM_FOOD
                    else:
                        step_rewards[s_obj.id] += PENALTY_MOVE_PARALLEL_FOOD
                
                steps_since_last_food_per_snake[s_obj.id] += 1
                if steps_since_last_food_per_snake[s_obj.id] > max_steps_without_food:
                    step_rewards[s_obj.id] += PENALTY_STUCK
                    s_obj.is_alive = False

            # 4. Update Q-tables
            if training_mode:
                for s_obj in snakes: # Iterate all original snakes for this episode
                    if s_obj.id in last_state_action: # If snake took an action this step
                        state, action = last_state_action[s_obj.id]
                        reward_for_q_update = step_rewards.get(s_obj.id, 0)
                        
                        next_state = agent.get_state(s_obj, food, snakes) # Get state even if dead
                        agent.update_q_table(state, action, reward_for_q_update, next_state)
                        current_episode_rewards_per_snake[s_obj.id] += reward_for_q_update
            
            # 5. Display
            if display_game or not training_mode:
                screen.fill(BLACK)
                for s_disp_obj in snakes:
                    if s_disp_obj.is_alive: 
                        draw_snake(screen, s_disp_obj)
                draw_food(screen, food)
                pygame.display.flip()
                clock.tick(15 if training_mode else 10)

        # Episode finished
        if training_mode:
            agent.decay_exploration()
        
        total_reward_this_episode = sum(current_episode_rewards_per_snake)
        total_rewards_all_episodes.append(total_reward_this_episode)
        
        current_avg_score = np.mean(scores_per_snake) if scores_per_snake else 0
        if current_avg_score > max_avg_score_overall:
            max_avg_score_overall = current_avg_score

        if training_mode and (episode + 1) % 100 == 0:
            avg_reward_last_100 = np.mean(total_rewards_all_episodes[-100:])
            # Calculate average score over the last 100 episodes (need to store scores per episode)
            # For simplicity, just print current episode's average score and max overall.
            num_alive_at_end = len([s for s in snakes if s.is_alive])
            print(f"Episode {episode + 1}/{num_episodes} - Avg Score: {current_avg_score:.2f}, Max Avg Score: {max_avg_score_overall:.2f}, Sum Reward: {total_reward_this_episode:.2f}, Avg Reward (last 100): {avg_reward_last_100:.2f}, Epsilon: {agent.epsilon:.4f}, Snakes Alive: {num_alive_at_end}, Time: {time.time() - start_time:.2f}s")
            if (episode + 1) % 500 == 0:
                 np.save("q_table.npy", agent.q_table)
                 print(f"Q-table saved at episode {episode+1}")

        if not training_mode: # Demonstration mode episode end
            if num_ai_snakes_to_train == 1:
                 print(f"Game Over! Score: {scores_per_snake[0]}")
            else:
                 print(f"Demo Episode Over! Scores: {scores_per_snake}, Snakes Alive: {len([s for s in snakes if s.is_alive])}")
            time.sleep(1) # Shorter sleep for demo

    if training_mode:
        print("Training finished.")
        np.save("q_table_final.npy", agent.q_table)
        print("Final Q-table saved as q_table_final.npy")
    return total_rewards_all_episodes, agent.q_table

def play_mode_game_loop(screen, clock, ai_q_learning_agent, num_ai_snakes=1):
    player_snake = Snake(snake_id="player") # Give player a unique ID
    food = Food()

    # Initialize AI snakes
    ai_snakes = []
    current_all_snake_segments = set(player_snake.get_body())

    for i in range(num_ai_snakes):
        ai_snake_instance = Snake(snake_id=f"ai_{i}")
        while True:
            potential_head_pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if potential_head_pos not in current_all_snake_segments:
                ai_snake_instance.body = [potential_head_pos]
                ai_snake_instance.direction = random.choice([UP, DOWN, LEFT, RIGHT])
                current_all_snake_segments.add(potential_head_pos) # Add only head initially
                ai_snakes.append(ai_snake_instance)
                break
    
    food.position = food.randomize_position(list(current_all_snake_segments))

    game_over = False
    player_score = 0

    font = pygame.font.Font(None, 36)

    while not game_over:
        # --- Player Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w or event.key == pygame.K_UP: player_snake.change_direction(UP)
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN: player_snake.change_direction(DOWN)
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT: player_snake.change_direction(LEFT)
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT: player_snake.change_direction(RIGHT)

        # --- Player Movement & Self/Wall Collision ---
        if player_snake.move() != 0: # move() returns penalty on collision, 0 otherwise
            game_over = True
            break

        # --- Player Eats Food ---
        if player_snake.get_head_position() == food.get_position():
            player_snake.grow_snake()
            player_score += 1
            temp_occupied_cells = set(player_snake.get_body())
            for s_ai in ai_snakes:
                if s_ai.is_alive: temp_occupied_cells.update(s_ai.get_body())
            food.position = food.randomize_position(list(temp_occupied_cells))

        # --- AI Snakes Movement & AI Self/Wall Collision ---
        active_ai_snakes_after_move = []
        for ai_snake in ai_snakes:
            if not ai_snake.is_alive: continue

            if ai_q_learning_agent and ai_q_learning_agent.q_table and len(ai_q_learning_agent.q_table) > 0:
                # Construct `all_snakes_for_ai_state` including player and other AIs
                all_snakes_for_ai_state = [player_snake] + ai_snakes
                state = ai_q_learning_agent.get_state(ai_snake, food, all_snakes_for_ai_state)
                action = ai_q_learning_agent.choose_action(state) 
                ai_snake.change_direction(action)
            else: 
                if random.random() < 0.2: 
                    ai_snake.change_direction(random.choice([UP, DOWN, LEFT, RIGHT]))
            
            if ai_snake.move() == 0: # No self/wall collision for this AI
                active_ai_snakes_after_move.append(ai_snake)
            # If ai_snake.move() != 0, it sets its own is_alive to False and is excluded
        ai_snakes = active_ai_snakes_after_move

        # --- Inter-Snake Collisions (Player vs AI, AI vs AI) ---
        # Player head vs AI body
        player_head = player_snake.get_head_position()
        for ai_s in ai_snakes: # Iterate only live AIs
            if not ai_s.is_alive: continue
            if player_head in ai_s.get_body(): # Player head hits AI body/head
                game_over = True; break
        if game_over: break

        # AI head vs Player body
        player_body_set = set(player_snake.get_body()) # Includes player head too
        for ai_s in ai_snakes:
            if not ai_s.is_alive: continue
            if ai_s.get_head_position() in player_body_set:
                game_over = True; break
        if game_over: break

        # AI vs AI collisions (simplified for play mode: head vs any part of other AI)
        # More complex logic from training loop could be used if desired
        temp_live_ais = [s for s in ai_snakes if s.is_alive]
        for i in range(len(temp_live_ais)):
            ai1 = temp_live_ais[i]
            ai1_head = ai1.get_head_position()
            for j in range(i + 1, len(temp_live_ais)):
                ai2 = temp_live_ais[j]
                ai2_head = ai2.get_head_position()
                if ai1_head == ai2_head: # Head-to-head
                    ai1.is_alive = False; ai2.is_alive = False; break
                if ai1_head in ai2.get_body(): ai1.is_alive = False; break
                if ai2_head in ai1.get_body(): ai2.is_alive = False
            if not ai1.is_alive: break # if ai1 died, re-evaluate from next ai1
        ai_snakes = [s for s in ai_snakes if s.is_alive] # Filter out AIs that died in AI vs AI

        # Draw everything
        screen.fill(BLACK)
        if player_snake.is_alive: draw_snake(screen, player_snake)
        for ai_snake_to_draw in ai_snakes:
            if ai_snake_to_draw.is_alive: draw_snake(screen, ai_snake_to_draw)
        draw_food(screen, food)
        
        score_text_surface = font.render(f"Score: {player_score}", True, WHITE)
        screen.blit(score_text_surface, (10, 10))
        
        pygame.display.flip()
        clock.tick(10)

    # Game over screen / message
    screen.fill(BLACK)
    final_message = f"Game Over! Your score: {player_score}"
    msg_surface = font.render(final_message, True, WHITE)
    msg_rect = msg_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(msg_surface, msg_rect)
    pygame.display.flip()
    time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake RL Agent")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from the last saved model (q_table.npy or q_table_final.npy if the former is not found).")
    parser.add_argument("--demo", action="store_true", help="Run in demonstration mode using q_table_final.npy. If combined with training, demo runs after training.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes for training.")
    parser.add_argument("--play", action="store_true", help="Play the game against AI snakes.")
    parser.add_argument("--ais", type=int, default=0, help="Number of AI snakes. In --play mode, this is opponents. In training mode (if >0), this is the number of agents to train simultaneously. If 0 in training/demo, defaults to 1.")

    args = parser.parse_args()

    # Combined mode check - play mode is exclusive of training execution paths
    if args.play and (args.continue_training or args.episodes != parser.get_default("episodes")):
        print("Warning: --continue-training and --episodes are ignored in --play mode. Player vs AI mode will use q_table_final.npy for AI opponents.")
        # Reset training-related flags if play is dominant
        if args.continue_training: args.continue_training = False
        # args.episodes default is fine for this warning

    pygame.init()
    infoObject = pygame.display.Info()
    screen_width = infoObject.current_w
    screen_height = infoObject.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    
    # Update global variables for screen dimensions
    global_WIDTH = screen_width
    global_HEIGHT = screen_height
    # Assuming GRID_SIZE remains constant, recalculate GRID_WIDTH and GRID_HEIGHT
    global_GRID_WIDTH = global_WIDTH // global_GRID_SIZE
    global_GRID_HEIGHT = global_HEIGHT // global_GRID_SIZE
    
    pygame.display.set_caption("Snake RL")
    clock = pygame.time.Clock()

    agent = QLearningAgent() # This agent is used for training or single-agent demo
    ai_agent_for_play_mode = None # This agent is for AI opponents in play mode

    q_table_final_path = "q_table_final.npy"
    q_table_periodic_path = "q_table.npy"
    rewards = [] # For plotting training rewards

    if args.play:
        print(f"--- Player vs. AI Mode ({args.ais if args.ais > 0 else 1} AI(s)) ---")
        num_opponents = args.ais if args.ais > 0 else 1 # Default to 1 AI opponent if --ais is 0 or not set in play mode
        ai_agent_for_play_mode = QLearningAgent(exploration_rate=0.0, min_exploration_rate=0.0) # Ensure exploitation for opponents
        try:
            print(f"Loading model for AI snakes from {q_table_final_path}...")
            ai_agent_for_play_mode.q_table = np.load(q_table_final_path, allow_pickle=True).item()
            print(f"Successfully loaded {q_table_final_path} for AI opponents.")
        except FileNotFoundError:
            print(f"Warning: {q_table_final_path} not found for AI opponents. AI will move randomly or with an empty Q-table.")
            ai_agent_for_play_mode.q_table = {} # Ensure q_table exists
        except Exception as e:
            print(f"Error loading Q-table for AI opponents: {e}. AI will move randomly or with an empty Q-table.")
            ai_agent_for_play_mode.q_table = {} # Ensure q_table exists
        
        play_mode_game_loop(screen, clock, ai_agent_for_play_mode, num_ai_snakes=num_opponents)
    
    else: # Training or Demo mode
        num_snakes_for_operation = args.ais if args.ais > 0 else 1

        training_performed_this_run = False
        # Determine if training should occur
        should_train = args.continue_training or not args.demo

        if should_train:
            if args.continue_training:
                try:
                    # Try loading periodic save first, then final save
                    try:
                        agent.q_table = np.load(q_table_periodic_path, allow_pickle=True).item()
                        print(f"Continuing training from {q_table_periodic_path} with {num_snakes_for_operation} snake(s).")
                    except FileNotFoundError:
                        agent.q_table = np.load(q_table_final_path, allow_pickle=True).item()
                        print(f"Continuing training from {q_table_final_path} with {num_snakes_for_operation} snake(s) ({q_table_periodic_path} not found).")
                    # Optionally adjust epsilon when continuing, e.g., agent.epsilon = max(agent.min_epsilon, agent.epsilon * 0.8)
                except FileNotFoundError:
                    print(f"No saved Q-table found ({q_table_periodic_path} or {q_table_final_path}). Starting fresh training with {num_snakes_for_operation} snake(s).")
                except Exception as e:
                    print(f"Error loading Q-table for continuation: {e}. Starting fresh training with {num_snakes_for_operation} snake(s).")
            else: # Fresh training (not continue_training, and not demo-only implies training)
                print(f"Starting new training session with {num_snakes_for_operation} snake(s).")

            print(f"Training for {args.episodes} episodes...")
            rewards, agent.q_table = game_loop(
                screen, clock, agent,
                num_episodes=args.episodes,
                training_mode=True,
                display_game=False, # Set to True to watch training (slower)
                num_ai_snakes_to_train=num_snakes_for_operation
            )
            training_performed_this_run = True
            
            if rewards: # Plotting rewards
                plt.figure() # Ensure a new figure for the plot
                plt.plot(rewards)
                plt.title(f'Total Rewards per Episode ({num_snakes_for_operation} snakes)')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plot_filename = f'rewards_plot_{num_snakes_for_operation}_snakes.png'
                plt.savefig(plot_filename)
                print(f"Rewards plot saved to {plot_filename}")
                # plt.show() # Uncomment to display plot, but it can be blocking

        # Handle demo phase
        if args.demo:
            print(f"--- Demo Mode ({num_snakes_for_operation} AI(s)) ---")
            if not training_performed_this_run: # Load model only if no training was done in this run
                try:
                    agent.q_table = np.load(q_table_final_path, allow_pickle=True).item()
                    print(f"Loaded Q-table from {q_table_final_path} for demo.")
                except FileNotFoundError:
                    print(f"Error: {q_table_final_path} not found for demo. Cannot run demo without a trained model.")
                    pygame.quit()
                    exit() # Exit if demo cannot run
                except Exception as e:
                    print(f"Error loading Q-table for demo: {e}")
                    pygame.quit()
                    exit() # Exit if demo cannot run

            agent.epsilon = 0.0  # Ensure exploitation for demo
            agent.min_epsilon = 0.0 # Ensure exploitation for demo

            print(f"Running demo with {num_snakes_for_operation} snake(s)...")
            game_loop(
                screen, clock, agent,
                num_episodes=5,  # Number of demo episodes
                training_mode=False,
                display_game=True,
                num_ai_snakes_to_train=num_snakes_for_operation
            )

    pygame.quit()
    # import sys # If using sys.exit()
    # sys.exit()
