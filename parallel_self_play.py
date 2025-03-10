import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Concatenate, BatchNormalization, Dropout, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam
import os
import json
import time
import threading
import pygame
from collections import deque
import random
import platform
from chess_env import ChessEnv
from self_play import SelfPlayAgent, calculate_reward
import copy
import matplotlib.pyplot as plt
import argparse
import sys

# Global configuration for TensorFlow with CUDA
def configure_tensorflow_for_cuda():
    """Configure TensorFlow for optimal CUDA 12.8 performance."""
    # Enable memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            # Set memory limit to 10GB (assuming 12GB card)
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
            )
            print(f"GPU found: {gpu}")
    
    # Enable XLA JIT compilation
    tf.config.optimizer.set_jit(True)
    
    # Configure for CUDA 12.8 performance optimizations
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "remapping": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "loop_optimization": True,
        "function_optimization": True,
        "debug_stripper": True,
        "disable_model_pruning": False,
        "scoped_allocator_optimization": True,
        "pin_to_host_optimization": True,
        "implementation_selector": True,
        "auto_mixed_precision": True,
        "disable_meta_optimizer": False,
    })
    
    # Disable eager execution for better performance
    tf.compat.v1.disable_eager_execution()
    
    # Use mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print("TensorFlow configured for CUDA 12.8 with optimizations enabled")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU devices: {len(gpus)}")

# Call the function to configure TensorFlow
configure_tensorflow_for_cuda()

class MultiGameVisualizer:
    """Visualizer for multiple chess games running in parallel"""
    
    def __init__(self, num_games, board_size=200, window_width=None, window_height=None):
        """
        Initialize the multi-game visualizer.
        
        Args:
            num_games: Number of games to visualize
            board_size: Size of each chess board in pixels
            window_width: Width of the window (calculated automatically if None)
            window_height: Height of the window (calculated automatically if None)
        """
        pygame.init()
        
        self.num_games = num_games
        self.board_size = board_size
        self.square_size = board_size // 8
        
        # Calculate grid layout
        self.cols = min(4, num_games)  # Maximum 4 games per row
        self.rows = (num_games + self.cols - 1) // self.cols
        
        # Calculate window size
        padding = 20  # Padding between boards
        self.window_width = window_width or (self.cols * board_size + (self.cols + 1) * padding)
        self.window_height = window_height or (self.rows * board_size + (self.rows + 1) * padding + 40)  # Extra space for info
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Multi-Game Chess Training Visualizer")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.LIGHT_SQUARE = (240, 217, 181)  # Light brown
        self.DARK_SQUARE = (181, 136, 99)    # Dark brown
        self.HIGHLIGHT = (255, 255, 0, 100)  # Yellow highlight with transparency
        self.BACKGROUND = (50, 50, 50)       # Dark gray
        self.TEXT_COLOR = (220, 220, 220)    # Light gray
        
        # Game state
        self.boards = [None] * num_games
        self.last_moves = [None] * num_games
        self.game_statuses = ["In Progress"] * num_games
        self.move_counts = [0] * num_games
        self.running = True
        self.clock = pygame.time.Clock()
        
        # Load piece images
        self.pieces = {}
        self.load_pieces()
        
        # Font for text
        self.font = pygame.font.SysFont('Arial', 14)
        self.title_font = pygame.font.SysFont('Arial', 20, bold=True)
        
    def load_pieces(self):
        """Load chess piece images"""
        piece_mapping = {
            1: 'wp',  # white pawn
            2: 'wn',  # white knight
            3: 'wb',  # white bishop
            4: 'wr',  # white rook
            5: 'wq',  # white queen
            6: 'wk',  # white king
            -1: 'bp', # black pawn
            -2: 'bn', # black knight
            -3: 'bb', # black bishop
            -4: 'br', # black rook
            -5: 'bq', # black queen
            -6: 'bk'  # black king
        }
        
        # Create a pieces directory if it doesn't exist
        if not os.path.exists('pieces'):
            os.makedirs('pieces')
            print("Created 'pieces' directory. Please add chess piece images.")
            
        # Try to load pieces, use colored rectangles as fallback
        for piece_value, piece_code in piece_mapping.items():
            piece_path = os.path.join('pieces', f'{piece_code}.png')
            try:
                if os.path.exists(piece_path):
                    img = pygame.image.load(piece_path)
                    self.pieces[piece_value] = pygame.transform.scale(img, (self.square_size, self.square_size))
                else:
                    # Create a colored rectangle as a fallback
                    color = self.WHITE if piece_value > 0 else self.BLACK
                    piece_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                    pygame.draw.rect(piece_surface, (*color, 180), (0, 0, self.square_size, self.square_size))
                    
                    # Add a letter to identify the piece
                    font = pygame.font.SysFont('Arial', self.square_size // 2)
                    piece_letter = piece_code[1].upper()
                    text = font.render(piece_letter, True, self.WHITE if piece_value < 0 else self.BLACK)
                    text_rect = text.get_rect(center=(self.square_size // 2, self.square_size // 2))
                    piece_surface.blit(text, text_rect)
                    
                    self.pieces[piece_value] = piece_surface
            except Exception as e:
                print(f"Error loading piece image {piece_code}: {e}")
                # Create a colored rectangle as a fallback
                color = self.WHITE if piece_value > 0 else self.BLACK
                piece_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                pygame.draw.rect(piece_surface, (*color, 180), (0, 0, self.square_size, self.square_size))
                
                # Add a letter to identify the piece
                font = pygame.font.SysFont('Arial', self.square_size // 2)
                piece_letter = piece_code[1].upper()
                text = font.render(piece_letter, True, self.WHITE if piece_value < 0 else self.BLACK)
                text_rect = text.get_rect(center=(self.square_size // 2, self.square_size // 2))
                piece_surface.blit(text, text_rect)
                
                self.pieces[piece_value] = piece_surface
    
    def draw_board(self, board_idx):
        """Draw a chess board at the specified index"""
        if self.boards[board_idx] is None:
            return
            
        # Calculate position
        row = board_idx // self.cols
        col = board_idx % self.cols
        padding = 20
        x_offset = col * (self.board_size + padding) + padding
        y_offset = row * (self.board_size + padding) + padding
        
        # Draw board background
        pygame.draw.rect(
            self.screen, 
            self.BACKGROUND, 
            (x_offset - 5, y_offset - 5, self.board_size + 10, self.board_size + 30)
        )
        
        # Draw game number and status
        game_text = f"Game {board_idx + 1}: {self.game_statuses[board_idx]} (Moves: {self.move_counts[board_idx]})"
        text_surface = self.font.render(game_text, True, self.TEXT_COLOR)
        self.screen.blit(text_surface, (x_offset, y_offset + self.board_size + 5))
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                color = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    (x_offset + col * self.square_size, y_offset + row * self.square_size, 
                     self.square_size, self.square_size)
                )
        
        # Draw pieces
        for row in range(8):
            for col in range(8):
                piece = self.boards[board_idx][row][col]
                if piece != 0:  # If there's a piece on this square
                    self.screen.blit(
                        self.pieces[piece], 
                        (x_offset + col * self.square_size, y_offset + row * self.square_size)
                    )
        
        # Highlight last move
        if self.last_moves[board_idx]:
            start, end = self.last_moves[board_idx]
            
            # Create a transparent surface for highlighting
            highlight_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, (255, 255, 0, 100), highlight_surface.get_rect())
            
            # Highlight the start and end squares
            start_col, start_row = start[1], start[0]  # Convert from (row, col) to (col, row) for drawing
            end_col, end_row = end[1], end[0]
            
            self.screen.blit(
                highlight_surface, 
                (x_offset + start_col * self.square_size, y_offset + start_row * self.square_size)
            )
            self.screen.blit(
                highlight_surface, 
                (x_offset + end_col * self.square_size, y_offset + end_row * self.square_size)
            )
    
    def update(self, boards, last_moves=None, game_statuses=None, move_counts=None):
        """Update the visualizer with the current board states"""
        # Update game states
        for i in range(self.num_games):
            if i < len(boards) and boards[i] is not None:
                self.boards[i] = boards[i]
            
            if last_moves and i < len(last_moves) and last_moves[i] is not None:
                self.last_moves[i] = last_moves[i]
                
            if game_statuses and i < len(game_statuses):
                self.game_statuses[i] = game_statuses[i]
                
            if move_counts and i < len(move_counts):
                self.move_counts[i] = move_counts[i]
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        
        # Clear screen
        self.screen.fill(self.BACKGROUND)
        
        # Draw title
        title = self.title_font.render("Parallel Chess Training", True, self.TEXT_COLOR)
        self.screen.blit(title, (self.window_width // 2 - title.get_width() // 2, 5))
        
        # Draw all boards
        for i in range(self.num_games):
            self.draw_board(i)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS
        return self.running
    
    def close(self):
        """Close the visualizer"""
        self.running = False
        pygame.quit()


class AdvancedChessModel:
    """Advanced deep neural network model for chess with CUDA 12.8 optimizations"""
    
    def __init__(self, state_shape, action_size):
        """
        Initialize the model with CUDA 12.8 optimizations.
        
        Args:
            state_shape: Shape of the state input
            action_size: Size of the action space (64*64=4096 for chess)
        """
        print("Initializing AdvancedChessModel with CUDA 12.8 optimizations...")
        self.state_shape = state_shape
        self.action_size = action_size
        
        # Build the model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Create compiled prediction functions for GPU acceleration
        self._create_predict_single()
        self._create_predict_batch()
        self._create_target_predict_single()
        self._create_target_predict_batch()
        
        print("Model compilation complete. Using CUDA 12.8 optimized functions.")
    
    def _residual_block(self, x, filters, kernel_size=3):
        """Residual block for the neural network"""
        y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
        y = tf.keras.layers.BatchNormalization()(y)
        out = tf.keras.layers.Add()([x, y])
        out = tf.keras.layers.LeakyReLU()(out)
        return out
    
    def _create_predict_single(self):
        """Create a compiled function for predicting on a single state"""
        input_signature = [tf.TensorSpec(shape=(None, *self.state_shape), dtype=tf.float32)]
        
        @tf.function(input_signature=input_signature, jit_compile=True)
        def predict_single(x):
            return self.model(x)
        
        self._predict_single_fn = predict_single
    
    def _create_predict_batch(self):
        """Create a compiled function for predicting on a batch of states"""
        input_signature = [tf.TensorSpec(shape=(None, *self.state_shape), dtype=tf.float32)]
        
        @tf.function(input_signature=input_signature, jit_compile=True)
        def predict_batch(x):
            return self.model(x)
        
        self._predict_batch_fn = predict_batch
    
    def _create_target_predict_single(self):
        """Create a compiled function for target network prediction on a single state"""
        input_signature = [tf.TensorSpec(shape=(None, *self.state_shape), dtype=tf.float32)]
        
        @tf.function(input_signature=input_signature, jit_compile=True)
        def target_predict_single(x):
            return self.target_model(x)
        
        self._target_predict_single_fn = target_predict_single
    
    def _create_target_predict_batch(self):
        """Create a compiled function for target network prediction on a batch of states"""
        input_signature = [tf.TensorSpec(shape=(None, *self.state_shape), dtype=tf.float32)]
        
        @tf.function(input_signature=input_signature, jit_compile=True)
        def target_predict_batch(x):
            return self.target_model(x)
        
        self._target_predict_batch_fn = target_predict_batch
    
    def _build_model(self):
        """Build the chess model with CUDA optimizations"""
        # Input layer
        input_layer = tf.keras.Input(shape=self.state_shape)
        
        # Convolutional layers with residual connections for feature extraction
        x = tf.keras.layers.Conv2D(128, 3, padding='same')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        # Add residual blocks for deep feature extraction
        for _ in range(5):  # Increased from 3 to 5 for better performance
            x = self._residual_block(x, 128)
        
        # Policy head (action probabilities)
        policy = tf.keras.layers.Conv2D(64, 1, activation='relu')(x)
        policy = tf.keras.layers.BatchNormalization()(policy)
        policy = tf.keras.layers.Flatten()(policy)
        policy = tf.keras.layers.Dense(self.action_size, activation='softmax', name='policy_output')(policy)
        
        # Value head (state evaluation)
        value = tf.keras.layers.Conv2D(32, 1, activation='relu')(x)
        value = tf.keras.layers.BatchNormalization()(value)
        value = tf.keras.layers.Flatten()(value)
        value = tf.keras.layers.Dense(256, activation='relu')(value)
        value = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(value)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=[policy, value])
        
        # Compile the model with mixed precision for CUDA optimization
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Loss functions
        model.compile(
            optimizer=optimizer,
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mean_squared_error'
            },
            metrics={
                'policy_output': 'accuracy',
                'value_output': 'mean_squared_error'
            }
        )
        
        return model
    
    def update_target_model(self):
        """Update target model with weights from the main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def predict(self, state, batch_size=1):
        """
        Predict action probabilities and value for a given state using CUDA acceleration.
        
        Args:
            state: The state to predict for
            batch_size: Number of states to predict for (1 for single state)
            
        Returns:
            Tuple of (policy probabilities, state value)
        """
        # Reshape state if it's a single state
        if state.ndim == 3:  # Single state
            state = np.expand_dims(state, axis=0)
        
        try:
            # Use the compiled function for better performance
            if batch_size == 1:
                return self._predict_single_fn(state)
            else:
                return self._predict_batch_fn(state)
        except Exception as e:
            # Fall back to direct model call if there's an error
            print(f"CUDA prediction error: {e}. Using fallback prediction.")
            return self.model(state)
    
    def target_predict(self, state, batch_size=1):
        """
        Predict action probabilities and value for a given state using the target network.
        
        Args:
            state: The state to predict for
            batch_size: Number of states to predict for (1 for single state)
            
        Returns:
            Tuple of (policy probabilities, state value)
        """
        # Reshape state if it's a single state
        if state.ndim == 3:  # Single state
            state = np.expand_dims(state, axis=0)
        
        try:
            # Use the compiled function for better performance
            if batch_size == 1:
                return self._target_predict_single_fn(state)
            else:
                return self._target_predict_batch_fn(state)
        except Exception as e:
            # Fall back to direct model call if there's an error
            print(f"CUDA target prediction error: {e}. Using fallback prediction.")
            return self.target_model(state)
    
    @tf.function(jit_compile=True)
    def _train_step(self, states, policy_targets, value_targets):
        """Optimized CUDA training step using TensorFlow's gradient tape"""
        with tf.GradientTape() as tape:
            # Forward pass
            policy_pred, value_pred = self.model(states, training=True)
            
            # Calculate losses
            policy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(policy_targets, policy_pred))
            value_loss = tf.reduce_mean(tf.square(value_targets - tf.squeeze(value_pred, axis=-1)))
            
            # Combined loss
            total_loss = policy_loss + value_loss
        
        # Get gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Apply gradients
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return policy_loss, value_loss, total_loss
    
    def train(self, states, actions, rewards, next_states, dones, batch_size=64):
        """
        Train the model on a batch of experiences using CUDA optimizations.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            batch_size: Size of the batch
            
        Returns:
            Training loss
        """
        # Ensure we have the right batch size
        states = np.reshape(states, (-1, *self.state_shape))
        next_states = np.reshape(next_states, (-1, *self.state_shape))
        batch_size = states.shape[0]
        
        try:
            # Get next state values from target model
            _, next_values = self.target_predict(next_states, batch_size)
            next_values = next_values.numpy().flatten()
            
            # Calculate target values (separate rewards array since they might have different shapes)
            target_values = np.array(rewards) + 0.99 * next_values * (1 - np.array(dones))
            
            # One-hot encode actions for policy targets
            policy_targets = np.zeros((batch_size, self.action_size))
            for i, action in enumerate(actions):
                policy_targets[i, action] = 1
            
            # Train using the optimized training step
            policy_loss, value_loss, total_loss = self._train_step(
                tf.convert_to_tensor(states, dtype=tf.float32),
                tf.convert_to_tensor(policy_targets, dtype=tf.float32),
                tf.convert_to_tensor(target_values, dtype=tf.float32)
            )
            
            return float(total_loss.numpy())
        
        except Exception as e:
            print(f"CUDA training error: {e}. Using fallback training.")
            
            # Fallback to using model.fit
            history = self.model.fit(
                states,
                {
                    'policy_output': policy_targets,
                    'value_output': target_values
                },
                batch_size=batch_size,
                verbose=0
            )
            
            return history.history['loss'][0]
    
    def save(self, filepath):
        """Save model weights to file"""
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        """Load model weights from file"""
        self.model.load_weights(filepath)


class AdvancedSelfPlayAgent:
    """Advanced agent for self-play chess training"""
    
    def __init__(self, state_shape, action_size, memory_size=100000, batch_size=64):
        """
        Initialize the advanced self-play agent.
        
        Args:
            state_shape: Shape of the state input (board representation)
            action_size: Size of the action space
            memory_size: Size of the replay memory
            batch_size: Batch size for training
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
        # Discount factor
        self.gamma = 0.99
        
        # How often to update target network
        self.update_target_every = 5
        
        # Create model
        self.model = AdvancedChessModel(state_shape, action_size)
        
        # Temperature for move selection (higher = more exploration)
        self.temperature = 1.0
        self.temperature_min = 0.1
        self.temperature_decay = 0.9999
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        """Choose an action based on the current state"""
        if not valid_actions:
            return None
            
        # With probability epsilon, choose a random valid action
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Reshape state for prediction
        state = np.reshape(state, (1,) + self.state_shape)
        
        # Get action values from model
        action_values, _ = self.model.predict(state)
        
        # Apply temperature to action values (higher temperature = more exploration)
        if self.temperature > 0:
            action_values = action_values / self.temperature
        
        # Create a mask for valid actions
        valid_mask = np.zeros(self.action_size)
        valid_mask[valid_actions] = 1
        
        # Apply mask to action values (set invalid actions to very negative values)
        masked_values = action_values[0] * valid_mask - 1e9 * (1 - valid_mask)
        
        # Choose the action with the highest value
        return np.argmax(masked_values)
    
    def replay(self, batch_size):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        # Sample a batch of experiences
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract components
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Train the model
        loss = self.model.train(states, actions, rewards, next_states, dones, batch_size)
        
        return loss
    
    def update_target_model(self):
        """Update the target model with the current model weights"""
        self.model.update_target_model()
    
    def save(self, filepath):
        """Save the model weights"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load model weights"""
        self.model.load(filepath)
        
    def decay_exploration(self):
        """Decay exploration parameters"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay


class ParallelSelfPlay:
    """Class for training chess AI using parallelized self-play with GPU acceleration"""
    
    def __init__(self, num_games=8, visualize=False, model_dir="models", log_file="parallel_training_log.json", batch_size=256):
        """
        Initialize parallel self-play training with CUDA optimization.
        
        Args:
            num_games: Number of parallel games to run
            visualize: Whether to visualize the games
            model_dir: Directory to save models
            log_file: File to save training log
            batch_size: Batch size for training (optimized for CUDA)
        """
        print("Initializing Parallel Self-Play with CUDA 12.8 optimizations")
        self.num_games = num_games
        self.visualize = visualize
        self.model_dir = model_dir
        self.log_file = log_file
        self.batch_size = batch_size
        
        # Game environments
        self.envs = [ChessEnv(visualize=False) for _ in range(num_games)]
        
        # Get state and action shapes from the environment
        self.state_shape = self.envs[0]._get_observation().shape
        self.action_size = 64 * 64  # All possible combinations of start and end squares
        
        # Create directory for models if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Initialize visualizer if needed
        self.visualizer = None
        if visualize:
            self.visualizer = MultiGameVisualizer(num_games)
            
        # Initialize agents
        print(f"Creating main models with CUDA optimizations...")
        self.main_agent = AdvancedSelfPlayAgent(self.state_shape, self.action_size, batch_size=batch_size)
        
        # Separate models for white and black
        self.white_main_agent = AdvancedSelfPlayAgent(self.state_shape, self.action_size, batch_size=batch_size)
        self.black_main_agent = AdvancedSelfPlayAgent(self.state_shape, self.action_size, batch_size=batch_size)
        
        # Agents for each game
        self.white_agents = [None] * num_games
        self.black_agents = [None] * num_games
        
        # Initialize training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "win_rates": {
                "white": [],
                "black": [],
                "draw": []
            },
            "white_loss": [],
            "black_loss": []
        }
        
        # Load the latest models if available
        self.episode_start = 0
        self._load_latest_model()
        
        # Load or initialize the training log
        self.training_log = self._load_training_log()
        
        # Setup GPU for optimal performance
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Configure GPU for optimal CUDA 12.8 performance."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s). Optimizing for CUDA 12.8...")
            
            try:
                # Optimize TensorFlow operations for CUDA
                tf.config.experimental.set_memory_growth(gpus[0], True)
                
                # Set GPU options for TensorFlow
                gpu_options = tf.compat.v1.GPUOptions(
                    per_process_gpu_memory_fraction=0.9,  # Use 90% of GPU memory
                    allow_growth=True
                )
                
                # Log device placement for debugging
                tf.debugging.set_log_device_placement(False)
                
                print("GPU setup complete for CUDA 12.8")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU found. Training will be significantly slower.")
            
        # Optional: Print TensorFlow configuration
        print(f"TensorFlow executing eagerly: {tf.executing_eagerly()}")
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    
    def _load_latest_model(self):
        """Load the latest model if available"""
        latest_model = None
        latest_white_model = None
        latest_black_model = None
        episode_start = 0
        
        # Look for the latest models
        for file in os.listdir(self.model_dir):
            if file.endswith(".weights.h5"):
                if "white" in file:
                    latest_white_model = file
                elif "black" in file:
                    latest_black_model = file
                else:
                    latest_model = file
        
        # Load white model if found
        if latest_white_model:
            model_path = os.path.join(self.model_dir, latest_white_model)
            print(f"Loading existing white model: {model_path}")
            self.white_main_agent.load(model_path)
            
            # Extract episode number from filename
            try:
                episode_num = int(latest_white_model.split("_")[-1].split(".")[0])
                if episode_num > episode_start:
                    episode_start = episode_num
                    print(f"White model episode: {episode_start}")
            except:
                pass
        
        # Load black model if found
        if latest_black_model:
            model_path = os.path.join(self.model_dir, latest_black_model)
            print(f"Loading existing black model: {model_path}")
            self.black_main_agent.load(model_path)
            
            # Extract episode number from filename
            try:
                episode_num = int(latest_black_model.split("_")[-1].split(".")[0])
                if episode_num > episode_start:
                    episode_start = episode_num
                    print(f"Black model episode: {episode_start}")
            except:
                pass
        
        # Load combined model or copy weights if not found
        if latest_model:
            model_path = os.path.join(self.model_dir, latest_model)
            print(f"Loading existing combined model: {model_path}")
            self.main_agent.load(model_path)
            
            # Extract episode number from filename
            try:
                episode_num = int(latest_model.split("_")[-1].split(".")[0])
                if episode_num > episode_start:
                    episode_start = episode_num
                    print(f"Combined model episode: {episode_start}")
            except:
                pass
        else:
            # If no combined model but we have white/black, average them
            if latest_white_model or latest_black_model:
                if latest_white_model and latest_black_model:
                    print("Creating combined model from white and black models")
                    # Average the weights
                    white_weights = self.white_main_agent.model.model.get_weights()
                    black_weights = self.black_main_agent.model.model.get_weights()
                    avg_weights = []
                    
                    for w_layer, b_layer in zip(white_weights, black_weights):
                        avg_weights.append((w_layer + b_layer) / 2)
                    
                    self.main_agent.model.model.set_weights(avg_weights)
                    self.main_agent.update_target_model()
                elif latest_white_model:
                    self.main_agent.model.model.set_weights(self.white_main_agent.model.model.get_weights())
                    self.main_agent.update_target_model()
                else:
                    self.main_agent.model.model.set_weights(self.black_main_agent.model.model.get_weights())
                    self.main_agent.update_target_model()
        
        # If we have a starting episode, share knowledge among all models
        if episode_start > 0:
            print(f"Continuing training from episode {episode_start}")
            
            # If we only loaded one type of model, share knowledge
            if latest_white_model and not latest_black_model:
                self.black_main_agent.model.model.set_weights(self.white_main_agent.model.model.get_weights())
                self.black_main_agent.update_target_model()
            elif latest_black_model and not latest_white_model:
                self.white_main_agent.model.model.set_weights(self.black_main_agent.model.model.get_weights())
                self.white_main_agent.update_target_model()
        
        return episode_start
    
    def _load_training_log(self):
        """Load the training log if it exists"""
        # Initialize with a consistent structure that includes both formats
        training_log = {
            # For use in train method
            "episode_rewards": [], 
            "episode_lengths": [], 
            "win_rates": {"white": [], "black": [], "draw": []},
            
            # For visualize_training.py
            "episodes": [],
            "rewards": [],
            "epsilons": []
        }
        
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    loaded_log = json.load(f)
                    # Ensure all required keys exist in the loaded log
                    for key in training_log.keys():
                        if key in loaded_log:
                            training_log[key] = loaded_log[key]
                print(f"Loaded training log from {self.log_file}")
            except Exception as e:
                print(f"Could not load training log from {self.log_file}, starting fresh: {e}")
        
        return training_log
    
    def _update_visualizer(self, states, last_moves, game_statuses, move_counts):
        """Update the visualizer with the current game states"""
        if not self.visualize or self.visualizer is None:
            return True
        
        # Extract boards from environments
        boards = [env.engine.board.copy() for env in self.envs]  # Use copy to avoid reference issues
        
        # Update the visualizer
        return self.visualizer.update(
            boards=boards,
            last_moves=last_moves,
            game_statuses=game_statuses,
            move_counts=move_counts
        )
    
    def _verify_board_state(self, env_index):
        """Verify that the board state is valid for the given environment"""
        env = self.envs[env_index]
        board = env.engine.board
        
        # Count pieces by type
        piece_counts = {}
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != 0:
                    if piece not in piece_counts:
                        piece_counts[piece] = 0
                    piece_counts[piece] += 1
        
        # Check for correct number of pieces
        expected_counts = {
            1: 8,   # white pawns
            2: 2,   # white knights
            3: 2,   # white bishops
            4: 2,   # white rooks
            5: 1,   # white queen
            6: 1,   # white king
            -1: 8,  # black pawns
            -2: 2,  # black knights
            -3: 2,  # black bishops
            -4: 2,  # black rooks
            -5: 1,  # black queen
            -6: 1   # black king
        }
        
        is_valid = True
        for piece, expected in expected_counts.items():
            actual = piece_counts.get(piece, 0)
            if actual != expected:
                print(f"Warning: Game {env_index} has {actual} {piece} pieces, expected {expected}")
                is_valid = False
                
                # If king is missing, confirm that it's properly detected as game over
                if (piece == 6 or piece == -6) and actual == 0:
                    # Ensure the game is recognized as over due to missing king
                    env.engine.check_game_state()
                    if not env.engine.checkmate:
                        print(f"Warning: King is missing but game not marked as checkmate")
        
        return is_valid
    
    def _initialize_game_agents(self):
        """Initialize separate agent instances for each game and side with GPU optimizations"""
        print("Initializing game agents with NVIDIA GPU optimizations...")
        
        for i in range(self.num_games):
            # Create agents with optimized parameters for RTX 4070 Super
            self.white_agents[i] = AdvancedSelfPlayAgent(
                self.state_shape, 
                self.action_size,
                memory_size=100000,  # Smaller memory per agent to save system RAM
                batch_size=self.batch_size  # Use larger batch size
            )
            self.black_agents[i] = AdvancedSelfPlayAgent(
                self.state_shape, 
                self.action_size,
                memory_size=100000,  # Smaller memory per agent to save system RAM
                batch_size=self.batch_size  # Use larger batch size
            )
            
            # Copy weights from the appropriate main models
            self.white_agents[i].model.model.set_weights(self.white_main_agent.model.model.get_weights())
            self.black_agents[i].model.model.set_weights(self.black_main_agent.model.model.get_weights())
            
            # Apply small random perturbations to create different variants
            # This ensures diversity in play styles while starting from the same base knowledge
            white_weights = self.white_agents[i].model.model.get_weights()
            black_weights = self.black_agents[i].model.model.get_weights()
            
            # Apply small random noise (0.1% variation)
            for j in range(len(white_weights)):
                if len(white_weights[j].shape) > 1:  # Only perturb weight matrices, not biases
                    white_weights[j] += np.random.normal(0, 0.001, white_weights[j].shape)
                    black_weights[j] += np.random.normal(0, 0.001, black_weights[j].shape)
            
            # Set the perturbed weights
            self.white_agents[i].model.model.set_weights(white_weights)
            self.black_agents[i].model.model.set_weights(black_weights)
            
            # Initialize target models
            self.white_agents[i].update_target_model()
            self.black_agents[i].update_target_model()
        
        print("Game agents initialized successfully")
    
    def _consolidate_knowledge(self):
        """
        Consolidate knowledge from game agents back to the main models,
        keeping white and black strategies separate
        """
        # Get all weights from white agents
        white_weights = []
        for i in range(self.num_games):
            if self.white_agents[i] is not None:
                white_weights.append(self.white_agents[i].model.model.get_weights())
        
        # Get all weights from black agents
        black_weights = []
        for i in range(self.num_games):
            if self.black_agents[i] is not None:
                black_weights.append(self.black_agents[i].model.model.get_weights())
        
        # Process white agents if we have any
        if white_weights:
            # Average the white weights efficiently
            avg_white_weights = [np.zeros_like(layer) for layer in white_weights[0]]
            for layer_idx in range(len(avg_white_weights)):
                layer_weights = np.array([weights[layer_idx] for weights in white_weights])
                avg_white_weights[layer_idx] = np.mean(layer_weights, axis=0)
            
            # Update the white main agent
            self.white_main_agent.model.model.set_weights(avg_white_weights)
            self.white_main_agent.update_target_model()
            print(f"Knowledge from {len(white_weights)} white agents consolidated to white main model")
        
        # Process black agents if we have any
        if black_weights:
            # Average the black weights efficiently
            avg_black_weights = [np.zeros_like(layer) for layer in black_weights[0]]
            for layer_idx in range(len(avg_black_weights)):
                layer_weights = np.array([weights[layer_idx] for weights in black_weights])
                avg_black_weights[layer_idx] = np.mean(layer_weights, axis=0)
            
            # Update the black main agent
            self.black_main_agent.model.model.set_weights(avg_black_weights)
            self.black_main_agent.update_target_model()
            print(f"Knowledge from {len(black_weights)} black agents consolidated to black main model")
        
        # Now create a combined model for evaluation (60% white knowledge, 40% black knowledge)
        # White gets slightly more weight as it has first-move advantage in chess
        if white_weights and black_weights:
            white_main_weights = self.white_main_agent.model.model.get_weights()
            black_main_weights = self.black_main_agent.model.model.get_weights()
            combined_weights = []
            
            # Weighted average of white and black models
            for w_layer, b_layer in zip(white_main_weights, black_main_weights):
                combined_weights.append(0.6 * w_layer + 0.4 * b_layer)
            
            # Update the combined main agent
            self.main_agent.model.model.set_weights(combined_weights)
            self.main_agent.update_target_model()
            
            print("White and black knowledge combined into main model")
        # If we only have one color, use that for the main agent too
        elif white_weights:
            self.main_agent.model.model.set_weights(self.white_main_agent.model.model.get_weights())
            self.main_agent.update_target_model()
        elif black_weights:
            self.main_agent.model.model.set_weights(self.black_main_agent.model.model.get_weights())
            self.main_agent.update_target_model()

    def _save_models(self, episode):
        """
        Save the current white, black, and combined models.
        
        Args:
            episode: Current episode number for naming
        """
        # Save white model
        white_model_path = os.path.join(self.model_dir, f"advanced_chess_model_white_episode_{episode}.weights.h5")
        self.white_main_agent.save(white_model_path)
        print(f"White model saved to {white_model_path}")
        
        # Save black model
        black_model_path = os.path.join(self.model_dir, f"advanced_chess_model_black_episode_{episode}.weights.h5")
        self.black_main_agent.save(black_model_path)
        print(f"Black model saved to {black_model_path}")
        
        # Save combined model (for compatibility with evaluation scripts)
        combined_model_path = os.path.join(self.model_dir, f"advanced_chess_model_episode_{episode}.weights.h5")
        self.main_agent.save(combined_model_path)
        print(f"Combined model saved to {combined_model_path}")
        
        # Save checkpoint file with metadata
        checkpoint_info = {
            "episode": episode,
            "timestamp": time.time(),
            "white_model": white_model_path,
            "black_model": black_model_path,
            "combined_model": combined_model_path,
            "training_config": {
                "batch_size": self.batch_size,
                "num_games": self.num_games
            }
        }
        
        checkpoint_path = os.path.join(self.model_dir, f"checkpoint_{episode}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_path}")

    def _evaluate_models(self, num_games=10):
        """
        Evaluate the current models by playing games with minimal exploration.
        
        Args:
            num_games: Number of games to play for evaluation
        """
        print(f"\nEvaluating models on {num_games} games...")
        
        # Create separate environments for evaluation
        eval_envs = [ChessEnv(visualize=False) for _ in range(num_games)]
        
        # Track metrics
        white_wins = 0
        black_wins = 0
        draws = 0
        total_moves = 0
        game_results = []
        
        # Create evaluation agents with minimal exploration
        white_eval_agent = copy.deepcopy(self.white_main_agent)
        black_eval_agent = copy.deepcopy(self.black_main_agent)
        
        # Set minimal exploration for evaluation
        white_eval_agent.epsilon = 0.05
        white_eval_agent.temperature = 0.1
        black_eval_agent.epsilon = 0.05
        black_eval_agent.temperature = 0.1
        
        # Play evaluation games
        for game in range(num_games):
            state = eval_envs[game].reset()
            done = False
            moves = 0
            white_to_move = True
            
            while not done and moves < 200:
                # Get the current agent
                current_agent = white_eval_agent if white_to_move else black_eval_agent
                
                # Get action
                action, _ = current_agent.act(state)
                
                # Execute the action
                state, _, done, status_info = eval_envs[game].step(action)
                
                # Update trackers
                moves += 1
                white_to_move = not white_to_move
                
                # Check for early termination
                if done:
                    break
            
            # Record result
            total_moves += moves
            game_status = status_info.get('status', 'Draw')
            
            if "checkmate" in game_status.lower():
                if "white wins" in game_status.lower():
                    white_wins += 1
                    game_results.append("White Win")
                else:
                    black_wins += 1
                    game_results.append("Black Win")
            else:
                draws += 1
                game_results.append("Draw")
        
        # Calculate rates
        win_rate = white_wins / num_games
        loss_rate = black_wins / num_games
        draw_rate = draws / num_games
        avg_moves = total_moves / num_games
        
        # Print evaluation results
        print(f"Evaluation results: White Wins: {white_wins}, Black Wins: {black_wins}, Draws: {draws}")
        print(f"White Win rate: {win_rate:.2f}, Black Win rate: {loss_rate:.2f}, Draw rate: {draw_rate:.2f}")
        print(f"Average game length: {avg_moves:.1f} moves\n")
        
        # Save evaluation results to log
        if "evaluations" not in self.training_log:
            self.training_log["evaluations"] = []
        
        self.training_log["evaluations"].append({
            "episode": self.episode_start + len(self.training_log["episode_rewards"]),
            "white_wins": white_wins,
            "black_wins": black_wins,
            "draws": draws,
            "white_win_rate": win_rate,
            "black_win_rate": loss_rate,
            "draw_rate": draw_rate,
            "avg_moves": avg_moves,
            "game_results": game_results
        })
        
        # Save updated log
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        return win_rate, draw_rate, avg_moves

    def log_training_progress(self, episode_rewards, episode, move_counts):
        """
        Log training progress and save to file.
        
        Args:
            episode_rewards: List of rewards for each game
            episode: Current episode number
            move_counts: List of move counts for each game
        """
        # Calculate average metrics
        avg_reward = sum(episode_rewards) / self.num_games
        avg_moves = sum(move_counts) / self.num_games
        
        # Count game outcomes
        white_wins = 0
        black_wins = 0
        draws = 0
        
        for i in range(self.num_games):
            game_status = self.envs[i].get_game_status()
            if "white wins" in game_status.lower():
                white_wins += 1
            elif "black wins" in game_status.lower():
                black_wins += 1
            else:
                draws += 1
        
        # Calculate win rates
        white_win_rate = white_wins / self.num_games
        black_win_rate = black_wins / self.num_games
        draw_rate = draws / self.num_games
        
        # Update metrics tracking
        self.training_metrics["episode_rewards"].append(float(avg_reward))
        self.training_metrics["episode_lengths"].append(float(avg_moves))
        self.training_metrics["win_rates"]["white"].append(white_win_rate)
        self.training_metrics["win_rates"]["black"].append(black_win_rate)
        self.training_metrics["win_rates"]["draw"].append(draw_rate)
        
        # Update training log
        current_episode = self.episode_start + len(self.training_log["episodes"])
        self.training_log["episodes"].append(current_episode)
        self.training_log["rewards"].append(float(avg_reward))
        self.training_log["epsilons"].append(float(self.white_main_agent.epsilon))  # Using white's epsilon
        self.training_log["win_rates"]["white"].append(white_win_rate)
        self.training_log["win_rates"]["black"].append(black_win_rate)
        self.training_log["win_rates"]["draw"].append(draw_rate)
        
        # Save training log
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        # Print progress
        print(f"Episode {episode+1}: Avg moves={avg_moves:.1f}, Avg reward={avg_reward:.2f}")
        print(f"Epsilon: {self.white_main_agent.epsilon:.4f}")
        print(f"Results: White wins={white_wins}, Black wins={black_wins}, Draws={draws}")
        print(f"Win rates: White={white_win_rate:.2f}, Black={black_win_rate:.2f}, Draw={draw_rate:.2f}")

    def train(self, episodes=1000, save_freq=100, max_moves=200, eval_freq=50, eval_games=10):
        """
        Train the agent using GPU-accelerated parallel self-play with multiple model instances.
        
        Args:
            episodes: Number of episodes (iterations) to train for
            save_freq: How often to save the model
            max_moves: Maximum number of moves per game
            eval_freq: How often to evaluate the model
            eval_games: Number of games to play during evaluation
        """
        print(f"Starting NVIDIA GPU-accelerated parallel self-play training with {self.num_games} simultaneous games")
        print(f"Batch size: {self.batch_size}, GPU Memory: ~10GB reserved")
        
        # Main training loop
        for episode in range(self.episode_start, self.episode_start + episodes):
            print(f"Episode {episode+1}/{self.episode_start + episodes}")
            
            # Explicitly recreate environments to ensure clean state
            self.envs = [ChessEnv(visualize=False) for _ in range(self.num_games)]
            
            # Initialize separate agents for each game with GPU optimizations
            self._initialize_game_agents()
            
            # Reset all environments
            states = [env.reset() for env in self.envs]
            dones = [False] * self.num_games
            episode_rewards = [0] * self.num_games
            move_counts = [0] * self.num_games
            last_moves = [None] * self.num_games
            game_statuses = ["In Progress"] * self.num_games
            
            # Track which color is to move in each game (start with white)
            white_to_move = [True] * self.num_games
            
            # Verify initial board state
            for i in range(self.num_games):
                self._verify_board_state(i)
            
            # Update visualizer with initial states
            if not self._update_visualizer(states, last_moves, game_statuses, move_counts):
                print("Visualizer closed. Stopping training.")
                break
            
            # Main game loop - continue until all games are done
            while not all(dones):
                # Process each game that is not done
                for i in range(self.num_games):
                    if dones[i]:
                        continue
                    
                    # Get the current side's agent
                    current_agent = self.white_agents[i] if white_to_move[i] else self.black_agents[i]
                    
                    # Get action from the current agent
                    action, action_values = current_agent.act(states[i])
                    
                    # Execute the action
                    next_state, reward, done, status_info = self.envs[i].step(action)
                    
                    # Update the last move
                    last_moves[i] = status_info.get('last_move', last_moves[i])
                    
                    # Store the transition in the current agent's memory
                    current_agent.remember(states[i], action, reward, next_state, done)
                    
                    # Update state and trackers
                    states[i] = next_state
                    episode_rewards[i] += reward
                    move_counts[i] += 1
                    game_statuses[i] = status_info.get('status', 'In Progress')
                    
                    # Check for draw or max moves
                    if move_counts[i] >= max_moves:
                        dones[i] = True
                        game_statuses[i] = "Draw (Max Moves)"
                    elif done:
                        dones[i] = True
                
                # Update visualizer with current states
                if not self._update_visualizer(states, last_moves, game_statuses, move_counts):
                    print("Visualizer closed. Stopping training.")
                    break
            
            # After all games complete, consolidate knowledge
            self._consolidate_knowledge()
            
            # Log training progress
            self.log_training_progress(episode_rewards, episode, move_counts)
            
            # Save models periodically
            if (episode + 1) % save_freq == 0:
                self._save_models(episode + 1)
            
            # Evaluate performance periodically
            if (episode + 1) % eval_freq == 0:
                self._evaluate_models(eval_games)
        
        # Save final models
        self._save_models(self.episode_start + episodes)
        
        print("Training complete")
        return self.training_log

    def _batch_train_agents(self, agents, side_name):
        """
        Perform batched training on a group of agents to improve GPU utilization.
        
        Args:
            agents: List of agent instances to train
            side_name: Name of the side (White/Black) for logging
        """
        # Calculate how many samples we need per agent to reach our optimal batch size
        total_agents = len(agents)
        samples_per_agent = max(32, self.batch_size // total_agents)
        
        # Check if we have enough samples across all agents
        total_samples = sum(len(agent.memory) for agent in agents)
        if total_samples < self.batch_size:
            return  # Not enough samples to train effectively
        
        # Train each agent with a portion of the batch
        for agent in agents:
            if len(agent.memory) > samples_per_agent * 4:  # Ensure we have enough samples
                for _ in range(1):  # Perform 1 training step
                    loss = agent.replay(samples_per_agent)
                    if loss is not None:
                        self.training_metrics[f"{side_name}_loss"].append(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel chess self-play training")
    parser.add_argument("--num_games", type=int, default=8, help="Number of games to run in parallel")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to train for")
    parser.add_argument("--save_freq", type=int, default=100, help="How often to save the model")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_file", type=str, default="parallel_training_log.json", help="File to log training progress")
    parser.add_argument("--visualize", action="store_true", help="Visualize the training")
    parser.add_argument("--eval_freq", type=int, default=50, help="How often to evaluate the model")
    parser.add_argument("--eval_games", type=int, default=10, help="Number of games to play during evaluation")
    
    args = parser.parse_args()
    
    # Create and run the parallel self-play trainer
    trainer = ParallelSelfPlay(
        num_games=args.num_games,
        visualize=args.visualize,
        model_dir=args.model_dir,
        log_file=args.log_file
    )
    
    trainer.train(
        episodes=args.episodes,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_games=args.eval_games
    ) 



# Run with CUDA 12.8 optimizations: 
# python parallel_self_play.py --num_games 8 --episodes 2000 --save_freq 100 --eval_freq 50 --batch_size 256