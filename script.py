"""
# Training an AI to play 2048 using Deep Q-Learning with PyTorch

## Installation Instructions
# Required dependencies:
# 1. Install PyTorch (for neural network)
# 2. Install NumPy (for game logic and state representation)
# 3. Install Matplotlib (for visualization)
# 4. Install Pygame (for game visualization)

# Run these commands in your terminal:
# pip install torch numpy matplotlib pygame
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import copy
import time
import pygame
import sys

# Colors for the pygame visualization
GRID_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (237, 190, 30),
    8192: (237, 185, 15)
}
TEXT_COLORS = {
    0: (205, 193, 180),
    2: (119, 110, 101),
    4: (119, 110, 101),
    8: (249, 246, 242),
    16: (249, 246, 242),
    32: (249, 246, 242),
    64: (249, 246, 242),
    128: (249, 246, 242),
    256: (249, 246, 242),
    512: (249, 246, 242),
    1024: (249, 246, 242),
    2048: (249, 246, 242),
    4096: (249, 246, 242),
    8192: (249, 246, 242)
}

# Define the 2048 game environment
class Game2048:
    def __init__(self, grid_size=4, visualize=False):
        self.grid_size = grid_size
        self.visualize = visualize  # Set this BEFORE calling reset
        self.window = None
        self.actions = [0, 1, 2, 3]  # 0: up, 1: right, 2: down, 3: left

        # Initialize pygame if visualization is enabled
        if visualize:
            if not pygame.get_init():
                pygame.init()
            self.window_size = 400
            self.cell_size = self.window_size // self.grid_size
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption('2048 AI')
            self.font = pygame.font.SysFont('Arial', 30)

        # Now call reset AFTER initializing visualize attribute
        self.reset()

    def __del__(self):
        # We'll let the training function handle pygame cleanup
        pass

    def reset(self):
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        # Add two initial tiles
        self.add_random_tile()
        self.add_random_tile()
        self.score = 0
        self.done = False
        self.moves_without_merge = 0  # Track moves without merges for better rewards

        # Draw initial state if visualization is enabled
        if self.visualize and self.window:
            self._draw_game()

        return self.get_state()

    def add_random_tile(self):
        # Add a new tile (2 or 4) to a random empty cell
        if np.any(self.grid == 0):
            empty_cells = list(zip(*np.where(self.grid == 0)))
            i, j = random.choice(empty_cells)
            self.grid[i, j] = 2 if random.random() < 0.9 else 4

    def get_state(self):
        # Convert grid to a state representation
        # We'll use a one-hot encoding approach for each position
        state = np.zeros((self.grid_size, self.grid_size, 16), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] > 0:
                    # Get the power of 2 (1 for 2, 2 for 4, 3 for 8, etc.)
                    power = int(np.log2(self.grid[i, j]))
                    if power <= 15:  # Limit to 2^15 = 32768
                        state[i, j, power-1] = 1
        # Convert to PyTorch tensor and add batch dimension
        return torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)  # [1, 16, 4, 4]

    def get_valid_moves(self):
        # Check which moves are valid
        valid_moves = []

        # For each direction, check if the move would change the grid
        for action in self.actions:
            grid_copy = copy.deepcopy(self.grid)
            _, moved, _ = self._make_move(grid_copy, action)
            if moved:
                valid_moves.append(action)

        return valid_moves

    def _make_move(self, grid, action):
        moved = False
        merge_score = 0

        # Rotate the grid based on the action to simplify processing
        if action == 0:  # up
            grid = grid  # No rotation needed
        elif action == 1:  # right
            grid = np.rot90(grid, 1)
        elif action == 2:  # down
            grid = np.rot90(grid, 2)
        elif action == 3:  # left
            grid = np.rot90(grid, 3)

        # Process the grid column by column
        for i in range(self.grid_size):
            col = grid[:, i].copy()
            # Remove zeros (compact)
            col = col[col != 0]

            # Merge tiles
            if len(col) > 1:
                for j in range(len(col) - 1):
                    if j < len(col) - 1 and col[j] == col[j + 1]:
                        col[j] *= 2
                        merge_score += col[j]
                        col[j + 1] = 0
                # Remove zeros again after merging
                col = col[col != 0]

            # Pad with zeros
            new_col = np.zeros(self.grid_size, dtype=np.int32)
            new_col[:len(col)] = col

            # Check if the column changed
            if not np.array_equal(grid[:, i], new_col):
                moved = True

            grid[:, i] = new_col

        # Rotate back
        if action == 0:  # up
            pass  # No rotation needed
        elif action == 1:  # right
            grid = np.rot90(grid, 3)
        elif action == 2:  # down
            grid = np.rot90(grid, 2)
        elif action == 3:  # left
            grid = np.rot90(grid, 1)

        return grid, moved, merge_score

    def step(self, action):
        # Save old state for comparison
        old_grid = self.grid.copy()
        old_max = np.max(old_grid)
        old_empty_cells = np.count_nonzero(old_grid == 0)

        valid_moves = self.get_valid_moves()

        # Check if move is valid
        if action in valid_moves:
            # Make the move
            self.grid, moved, merge_score = self._make_move(self.grid, action)

            if moved:
                # Add a new tile
                self.add_random_tile()
                # Update score
                self.score += merge_score

                #######################
                # REWARD CALCULATION
                #######################

                # Base reward starts at zero - will be positive or negative
                reward = 0

                # Track state changes
                new_max = np.max(self.grid)
                new_empty_cells = np.count_nonzero(self.grid == 0)

                # 1. PRIMARY OBJECTIVE: Create higher value tiles
                # Exponential reward for new max tile (2^n creates n^2 reward)
                if new_max > old_max:
                    # Log base 2 of new_max gives the exponent (e.g., 1024 -> 10)
                    exponent = np.log2(new_max)
                    reward += exponent ** 2 * 10  # Quadratic scaling

                    # Extra reward if max is in corner
                    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
                    max_pos = np.unravel_index(np.argmax(self.grid), self.grid.shape)
                    if max_pos in corners:
                        reward += exponent ** 2 * 5  # Additional bonus for corner

                # 2. MERGE QUALITY: Reward merges, bigger merges are better
                if merge_score > 0:
                    # Square root scaling - bigger merges much more valuable
                    reward += np.sqrt(merge_score) * 5
                else:
                    # Small penalty for not merging anything
                    reward -= 10

                # 3. BOARD OPENNESS: Critical to have empty cells
                # Exponential penalty when board fills up
                empty_ratio = new_empty_cells / (self.grid_size * self.grid_size)
                if empty_ratio < 0.25:  # Critical threshold - 75% full
                    # Severe exponential penalty when board gets too full
                    reward -= (1 - empty_ratio) ** 3 * 500
                else:
                    # Modest reward for keeping board open
                    reward += new_empty_cells * 2

                # 4. SNAKE PATTERN: Using the most effective 2048 strategy
                snake_score = self._calculate_snake_pattern()
                reward += snake_score * 20

                # 5. SMOOTHNESS: Adjacent tiles should have similar values
                smoothness = self._calculate_smoothness()
                reward += smoothness * 10

                # 6. GAME OVER: Huge penalty for losing
                if not self.get_valid_moves():
                    self.done = True
                    # Bigger penalty for early game overs (low max tile)
                    exponent = np.log2(new_max)
                    reward -= 3000 - (exponent * 100)  # Less penalty for higher max tiles
            else:
                # Invalid move (should never happen with proper valid_moves check)
                reward = -100
        else:
            # Invalid move (should never happen with proper valid_moves check)
            reward = -100

        # Update visualization if enabled
        if self.visualize and self.window:
            self._draw_game()
            pygame.display.update()

        return self.get_state(), reward, self.done, {"score": self.score, "highest": np.max(self.grid)}

    def _calculate_snake_pattern(self):
        """
        Calculate how well the board follows the optimal snake pattern.
        This is the most effective 2048 strategy - build in a specific zigzag pattern.

        The snake pattern with values decreasing along the path:
        [1][2][3][4]
        [8][7][6][5]
        [9][10][11][12]
        [16][15][14][13]

        Where 1 is the highest value, 2 is second highest, etc.
        """
        # Create the weight matrix for the snake pattern
        # Higher values = more important positions
        weights = np.array([
            [15, 14, 13, 12],
            [8, 9, 10, 11],
            [7, 6, 5, 4],
            [0, 1, 2, 3]
        ])

        # Alternative for top-right corner (adjust based on your preference)
        # weights = np.array([
        #     [12, 13, 14, 15],
        #     [11, 10, 9,  8],
        #     [4,  5,  6,  7],
        #     [3,  2,  1,  0]
        # ])

        # Create a ranking of tile values (excluding zeros)
        flat_grid = self.grid.flatten()
        non_zero_tiles = flat_grid[flat_grid > 0]
        ranked_tiles = np.zeros_like(self.grid)

        if len(non_zero_tiles) > 0:
            # Sort unique non-zero values in descending order
            unique_values = np.unique(non_zero_tiles)
            sorted_values = np.sort(unique_values)[::-1]

            # Assign ranks to tiles (1 = highest value, 2 = second highest, etc.)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.grid[i, j] > 0:
                        # Find position in sorted array (smaller index = higher rank)
                        rank = np.where(sorted_values == self.grid[i, j])[0][0]
                        ranked_tiles[i, j] = rank

        # Calculate score based on how well ranks match the snake pattern
        score = 0
        max_possible_score = 0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] > 0:
                    # Higher weights and lower ranks (higher values) should align
                    # Perfect alignment: highest value (rank 0) at weight 15
                    alignment_quality = (15 - weights[i, j]) * ranked_tiles[i, j]
                    # Invert so higher is better
                    score += 15 - min(15, alignment_quality)
                    max_possible_score += 15

        # Normalize the score (0 to 1)
        normalized_score = score / max(1, max_possible_score)

        # Scale to a more useful range
        return normalized_score * 100

    def _calculate_smoothness(self):
        """
        Calculate how smooth the board is (adjacent tiles have similar values).
        Smoothness makes it easier to merge tiles.
        """
        smoothness = 0

        # Check horizontal adjacency
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if self.grid[i, j] > 0 and self.grid[i, j + 1] > 0:
                    # Calculate difference in terms of powers of 2
                    diff = abs(np.log2(self.grid[i, j]) - np.log2(self.grid[i, j + 1]))
                    # Reward smaller differences (smoothness)
                    if diff == 0:
                        smoothness += 2  # Same values (ready to merge)
                    elif diff == 1:
                        smoothness += 1  # Close values (almost ready to merge)

        # Check vertical adjacency
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                if self.grid[i, j] > 0 and self.grid[i + 1, j] > 0:
                    # Calculate difference in terms of powers of 2
                    diff = abs(np.log2(self.grid[i, j]) - np.log2(self.grid[i + 1, j]))
                    # Reward smaller differences (smoothness)
                    if diff == 0:
                        smoothness += 2  # Same values (ready to merge)
                    elif diff == 1:
                        smoothness += 1  # Close values (almost ready to merge)

        return smoothness

    def _draw_game(self):
        """Draw the game state using pygame"""
        # Draw background
        self.window.fill(GRID_COLOR)

        # Draw cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.grid[i, j]

                # Draw cell background
                pygame.draw.rect(
                    self.window,
                    TILE_COLORS.get(value, (237, 194, 46)),  # Default to 2048 color
                    (j * self.cell_size + 10, i * self.cell_size + 10,
                     self.cell_size - 20, self.cell_size - 20),
                    border_radius=5
                )

                # Draw cell value
                if value != 0:
                    text_color = TEXT_COLORS.get(value, (249, 246, 242))

                    # Adjust font size based on number of digits
                    font_size = 30
                    if value > 999:
                        font_size = 22
                    elif value > 99:
                        font_size = 25

                    font = pygame.font.SysFont('Arial', font_size, bold=True)
                    text = font.render(str(value), True, text_color)
                    text_rect = text.get_rect(center=(
                        j * self.cell_size + self.cell_size // 2,
                        i * self.cell_size + self.cell_size // 2
                    ))
                    self.window.blit(text, text_rect)

        # Draw score
        score_font = pygame.font.SysFont('Arial', 20)
        score_text = score_font.render(f"Score: {self.score}", True, (119, 110, 101))
        self.window.blit(score_text, (10, 10))

        # Update display
        pygame.display.update()

    def render(self):
        """Print the current grid state to console"""
        print(f"Score: {self.score}")
        print(self.grid)
        print("\n")

        # If visualization is enabled, render the game
        if self.visualize and self.window:
            self._draw_game()
            pygame.time.delay(100)  # Small delay to make the visualization visible


# Define the DQN Network using PyTorch - with an improved architecture
class DQNetwork(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQNetwork, self).__init__()

        # Convolutional layers to process the board
        self.conv1 = nn.Conv2d(input_shape[0], 128, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(128)  # Added batch normalization
        self.conv2 = nn.Conv2d(128, 256, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(256)  # Added batch normalization
        self.flatten = nn.Flatten()

        # Calculate flattened size
        self.feature_size = self._get_conv_output(input_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)  # Increased width
        self.dropout = nn.Dropout(0.2)  # Added dropout for regularization
        self.fc2 = nn.Linear(512, 256)  # Added another layer for depth
        self.fc3 = nn.Linear(256, action_size)

    def _get_conv_output(self, shape):
        # Helper function to calculate conv output dimensions
        input = torch.rand(1, *shape)
        output = self.conv1(input)
        output = F.relu(output)
        output = self.bn1(output)
        output = self.conv2(output)
        output = F.relu(output)
        output = self.bn2(output)
        return int(np.prod(output.size()))

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Define the DQN Agent with improved learning
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Increased memory size
        self.gamma = 0.99  # Increased discount factor for better long-term planning
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # Slightly higher min epsilon for exploration
        self.epsilon_decay = 0.9999  # Slower decay for better exploration
        self.learning_rate = 0.0001  # Reduced learning rate for stability
        self.batch_size = 256  # Increased batch size
        self.update_frequency = 4  # Only update network every few steps
        self.target_update_freq = 2000
        self.steps = 0

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS (Metal Performance Shaders)")
        else:
            self.device = torch.device("cpu")
            print("MPS not available. Using CPU")

        # Initialize Q-networks
        self.model = DQNetwork(state_shape, action_size).to(self.device)
        self.target_model = DQNetwork(state_shape, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        self.criterion = nn.HuberLoss()  # Use Huber loss for stability instead of MSE

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves=None):
        # Act using epsilon-greedy policy
        if valid_moves is None or len(valid_moves) == 0:
            valid_moves = [0, 1, 2, 3]

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)

        state = state.to(self.device)
        with torch.no_grad():
            act_values = self.model(state)

        # Filter out invalid moves
        act_values = act_values.cpu().numpy()[0]
        valid_values = np.copy(act_values)
        for i in range(self.action_size):
            if i not in valid_moves:
                valid_values[i] = -float('inf')

        return np.argmax(valid_values)

    def replay(self):
        # Train the agent with experiences from memory
        if len(self.memory) < self.batch_size:
            return

        self.steps += 1

        # Only update every few steps for stability
        if self.steps % self.update_frequency != 0:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.cat([s1 for s1, _, _, _, _ in minibatch]).to(self.device)
        next_states = torch.cat([s2 for _, _, _, s2, _ in minibatch]).to(self.device)

        # Predict Q-values
        all_q_values = self.model(states)
        all_next_q_values = self.target_model(next_states).detach()

        # Prepare target values using Double DQN approach
        target_q_values = all_q_values.clone()

        # Update Q-values for actions taken using Double DQN
        for idx, (_, action, reward, _, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                # Double DQN: use online network to select action, target network to evaluate
                with torch.no_grad():
                    online_q_values = self.model(next_states[idx:idx+1])
                    best_action = torch.argmax(online_q_values, dim=1).item()
                    target = reward + self.gamma * all_next_q_values[idx, best_action]

            target_q_values[idx, action] = target

        # Optimize the model
        self.optimizer.zero_grad()
        loss = self.criterion(all_q_values, target_q_values)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.eval()

    def save(self, name):
        torch.save(self.model.state_dict(), name)


# Training function - optimized with proper pygame handling
def train_agent(episodes=1000, update_target_every=1000, save_every=100, visualize_every=100):
    env = Game2048(visualize=False)  # Non-visualized environment for most episodes
    state_shape = (16, 4, 4)  # channels first for PyTorch
    action_size = 4  # number of possible actions

    agent = DQNAgent(state_shape, action_size)

    scores = []
    max_tiles = []
    episodes_since_last_high = 0
    best_score = 0

    for e in range(episodes):
        # Only visualize on specific episodes to save memory
        visualize = (e % visualize_every == 0)

        current_env = env  # Default: use non-visualized environment

        # If visualizing, we need to make sure pygame is properly initialized
        if visualize:
            print(f"\nEpisode {e+1}: Visualizing gameplay...")
            # Make sure pygame is initialized
            if not pygame.get_init():
                pygame.init()
            # Create a fresh visualization environment
            current_env = Game2048(visualize=True)

        state = current_env.reset()
        total_reward = 0

        while not current_env.done:
            valid_moves = current_env.get_valid_moves()
            action = agent.act(state, valid_moves)
            next_state, reward, done, info = current_env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {info['score']}, " +
                      f"Highest Tile: {info['highest']}, Epsilon: {agent.epsilon:.4}")
                scores.append(info['score'])
                max_tiles.append(info['highest'])

                # Track best performance
                if info['score'] > best_score:
                    best_score = info['score']
                    agent.save("2048_best_model.pt")
                    episodes_since_last_high = 0
                    print(f"New best score: {best_score}! Model saved.")
                else:
                    episodes_since_last_high += 1
                break

            agent.replay()

            # Only add delay if visualizing
            if visualize:
                pygame.time.delay(30)  # Shorter delay for less waiting

                # Handle pygame events to prevent freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return agent  # Early return if user closes window

        # Clean up pygame resources after visualization
        if visualize:
            pygame.time.delay(500)  # Show final state briefly
            pygame.quit()  # Explicitly quit pygame to free resources

        # Update target model periodically (based on steps, not episodes)
        if e % update_target_every == 0:
            agent.update_target_model()
            print("Target network updated.")

        # Save model periodically
        if e % save_every == 0 and e > 0:
            agent.save(f"2048_model_ep{e}.pt")

    # Plot training results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('Game Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.subplot(1, 3, 2)
    plt.plot(max_tiles)
    plt.title('Highest Tiles')
    plt.xlabel('Episode')
    plt.ylabel('Highest Tile Value')

    # Plot moving average
    window_size = 50
    if len(scores) >= window_size:
        moving_avg = [np.mean(scores[i:i+window_size]) for i in range(len(scores)-window_size+1)]
        plt.subplot(1, 3, 3)
        plt.plot(range(window_size-1, len(scores)), moving_avg)
        plt.title(f'Moving Average (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Avg Score')

    plt.tight_layout()
    plt.savefig('2048_training_results.png')
    plt.show()

    return agent

# Evaluation function with better pygame handling
def evaluate_agent(agent, games=5, render=True):
    # Initialize pygame if rendering
    if render and not pygame.get_init():
        pygame.init()

    env = Game2048(visualize=render)
    scores = []
    highest_tiles = []

    for i in range(games):
        state = env.reset()
        total_reward = 0
        moves = 0

        print(f"\nGame {i+1}/{games}")
        while not env.done:
            valid_moves = env.get_valid_moves()
            action = agent.act(state, valid_moves)

            # Display action being taken
            action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
            print(f"Move {moves+1}: {action_names[action]}")

            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            moves += 1

            if render:
                pygame.time.delay(200)  # Add delay for better visualization

                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            if done:
                scores.append(info['score'])
                highest_tiles.append(info['highest'])
                print(f"Game {i+1} finished - Score: {info['score']}, Highest Tile: {info['highest']}")
                if render:
                    pygame.time.delay(1000)  # Show final state for a second
                break

    # Clean up pygame when done
    if render and pygame.get_init():
        pygame.quit()

    print(f"\nEvaluation over {games} games:")
    print(f"Average Score: {np.mean(scores)}")
    print(f"Average Highest Tile: {np.mean(highest_tiles)}")
    print(f"Max Score: {max(scores)}")
    print(f"Max Highest Tile: {max(highest_tiles)}")

    return np.mean(scores), np.mean(highest_tiles)

# Main execution function
if __name__ == "__main__":
    # Set longer training with less frequent visualization
    print("Starting training...")
    trained_agent = train_agent(episodes=50000, visualize_every=1000, update_target_every=100)

    trained_agent.save("2048_final_model.pt")

    # Make sure pygame is closed before evaluation
    if pygame.get_init():
        pygame.quit()

    # Evaluate the agent with visualization
    print("\nVisualizing trained agent's gameplay...")
    evaluate_agent(trained_agent, games=5, render=True)
