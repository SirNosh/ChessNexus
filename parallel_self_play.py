import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
import os
import json
import time
import threading
import pygame
from collections import deque
import random
from chess_env import ChessEnv
from self_play import SelfPlayAgent, calculate_reward
from visualize_training import log_training_progress

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


class ParallelSelfPlay:
    """Trains a chess agent using parallel self-play across multiple games"""
    
    def __init__(self, num_games=4, visualize=False, model_dir="models", log_file="parallel_training_log.json"):
        """
        Initialize the parallel self-play trainer.
        
        Args:
            num_games: Number of games to run in parallel
            visualize: Whether to visualize the games
            model_dir: Directory to save models
            log_file: File to log training progress
        """
        self.num_games = num_games
        self.visualize = visualize
        self.model_dir = model_dir
        self.log_file = log_file
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize environments
        self.envs = [ChessEnv(visualize=False) for _ in range(num_games)]
        
        # Get state shape and action size from first environment
        self.state_shape = self.envs[0].observation_space_shape
        self.action_size = self.envs[0].action_space_size
        
        # Initialize agent
        self.agent = SelfPlayAgent(self.state_shape, self.action_size)
        
        # Initialize visualizer if needed
        self.visualizer = None
        if visualize:
            self.visualizer = MultiGameVisualizer(num_games)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = {"white": [], "black": [], "draw": []}
        
        # Load existing model if available
        self.episode_start = self._load_latest_model()
        
        # Load training log if it exists
        self.training_log = self._load_training_log()
    
    def _load_latest_model(self):
        """Load the latest model if available"""
        latest_model = None
        episode_start = 0
        
        for file in os.listdir(self.model_dir):
            if file.endswith(".weights.h5"):
                latest_model = file
        
        if latest_model:
            model_path = os.path.join(self.model_dir, latest_model)
            print(f"Loading existing model: {model_path}")
            self.agent.load(model_path)
            
            # Extract episode number from filename
            try:
                episode_start = int(latest_model.split("_")[-1].split(".")[0])
                print(f"Continuing training from episode {episode_start}")
            except:
                episode_start = 0
        
        return episode_start
    
    def _load_training_log(self):
        """Load the training log if it exists"""
        training_log = {
            "episode_rewards": [], 
            "episode_lengths": [], 
            "win_rates": {"white": [], "black": [], "draw": []}
        }
        
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    training_log = json.load(f)
                print(f"Loaded training log from {self.log_file}")
            except:
                print(f"Could not load training log from {self.log_file}, starting fresh")
        
        return training_log
    
    def _update_visualizer(self, states, last_moves, game_statuses, move_counts):
        """Update the visualizer with the current game states"""
        if not self.visualize or self.visualizer is None:
            return True
        
        # Extract boards from environments
        boards = [env.engine.board for env in self.envs]
        
        # Update the visualizer
        return self.visualizer.update(
            boards=boards,
            last_moves=last_moves,
            game_statuses=game_statuses,
            move_counts=move_counts
        )
    
    def train(self, episodes=1000, save_freq=100, max_moves=200):
        """
        Train the agent using parallel self-play.
        
        Args:
            episodes: Number of episodes (iterations) to train for
            save_freq: How often to save the model
            max_moves: Maximum number of moves per game
        """
        print(f"Starting parallel self-play training with {self.num_games} simultaneous games")
        
        # Main training loop
        for episode in range(self.episode_start, self.episode_start + episodes):
            print(f"Episode {episode+1}/{self.episode_start + episodes}")
            
            # Reset all environments
            states = [env.reset() for env in self.envs]
            dones = [False] * self.num_games
            episode_rewards = [0] * self.num_games
            move_counts = [0] * self.num_games
            last_moves = [None] * self.num_games
            game_statuses = ["In Progress"] * self.num_games
            
            # Update visualizer with initial states
            if not self._update_visualizer(states, last_moves, game_statuses, move_counts):
                print("Visualizer closed. Stopping training.")
                break
            
            # Play all games until they're done
            all_done = False
            while not all_done:
                # Process each game
                for i in range(self.num_games):
                    # Skip if this game is already done
                    if dones[i] or move_counts[i] >= max_moves:
                        continue
                    
                    # Get valid actions for this game
                    valid_actions = self.envs[i].get_valid_actions()
                    
                    if not valid_actions:
                        dones[i] = True
                        game_statuses[i] = "No valid moves"
                        continue
                    
                    # Choose action
                    action = self.agent.act(states[i], valid_actions)
                    
                    # Decode the action to get the move
                    start_square_idx = action // 64
                    end_square_idx = action % 64
                    
                    start_row, start_col = start_square_idx // 8, start_square_idx % 8
                    end_row, end_col = end_square_idx // 8, end_square_idx % 8
                    
                    start = (start_row, start_col)
                    end = (end_row, end_col)
                    
                    # Take action
                    next_state, reward, done, info = self.envs[i].step(action)
                    
                    # Remember the experience
                    self.agent.remember(states[i], action, reward, next_state, done)
                    
                    # Update state and counters
                    states[i] = next_state
                    episode_rewards[i] += reward
                    move_counts[i] += 1
                    last_moves[i] = (start, end)
                    
                    # Update game status
                    if done:
                        dones[i] = True
                        if self.envs[i].engine.checkmate:
                            winner = "Black" if self.envs[i].engine.white_to_move else "White"
                            game_statuses[i] = f"{winner} won"
                        elif self.envs[i].engine.stalemate:
                            game_statuses[i] = "Stalemate"
                        else:
                            game_statuses[i] = "Game Over"
                    elif move_counts[i] >= max_moves:
                        dones[i] = True
                        game_statuses[i] = "Max moves reached"
                
                # Train the agent on collected experiences
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.replay()
                
                # Update visualizer
                if not self._update_visualizer(states, last_moves, game_statuses, move_counts):
                    print("Visualizer closed. Stopping training.")
                    break
                
                # Check if all games are done
                all_done = all(dones)
                
                # Add a small delay to make visualization viewable
                if self.visualize:
                    time.sleep(0.05)
            
            # Update target network periodically
            if episode % self.agent.update_target_every == 0:
                self.agent.update_target_model()
            
            # Decay epsilon
            if self.agent.epsilon > self.agent.epsilon_min:
                self.agent.epsilon *= self.agent.epsilon_decay
            
            # Record metrics
            avg_reward = sum(episode_rewards) / self.num_games
            avg_moves = sum(move_counts) / self.num_games
            self.episode_rewards.append(avg_reward)
            self.episode_lengths.append(avg_moves)
            
            # Count wins
            white_wins = 0
            black_wins = 0
            draws = 0
            
            for i in range(self.num_games):
                if "White won" in game_statuses[i]:
                    white_wins += 1
                elif "Black won" in game_statuses[i]:
                    black_wins += 1
                else:
                    draws += 1
            
            # Calculate win rates
            self.win_rates["white"].append(white_wins / self.num_games)
            self.win_rates["black"].append(black_wins / self.num_games)
            self.win_rates["draw"].append(draws / self.num_games)
            
            # Update training log
            self.training_log["episode_rewards"].append(float(avg_reward))
            self.training_log["episode_lengths"].append(float(avg_moves))
            self.training_log["win_rates"]["white"].append(self.win_rates["white"][-1])
            self.training_log["win_rates"]["black"].append(self.win_rates["black"][-1])
            self.training_log["win_rates"]["draw"].append(self.win_rates["draw"][-1])
            
            # Save training log
            with open(self.log_file, 'w') as f:
                json.dump(self.training_log, f)
            
            # Print progress
            print(f"Episode {episode+1}: Avg moves={avg_moves:.1f}, Avg reward={avg_reward:.2f}, epsilon={self.agent.epsilon:.4f}")
            print(f"Results: White wins={white_wins}, Black wins={black_wins}, Draws={draws}")
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                model_path = os.path.join(self.model_dir, f"parallel_chess_model_episode_{episode+1}.weights.h5")
                self.agent.save(model_path)
                print(f"Model saved to {model_path}")
                
                # Log training progress
                log_training_progress(self.log_file)
        
        # Save final model
        model_path = os.path.join(self.model_dir, f"parallel_chess_model_episode_{self.episode_start + episodes}.weights.h5")
        self.agent.save(model_path)
        print(f"Final model saved to {model_path}")
        
        # Log final training progress
        log_training_progress(self.log_file)
        
        # Close visualizer if open
        if self.visualize and self.visualizer:
            self.visualizer.close()
        
        return self.agent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel chess self-play training")
    parser.add_argument("--num_games", type=int, default=4, help="Number of games to run in parallel")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--save_freq", type=int, default=100, help="How often to save the model")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_file", type=str, default="parallel_training_log.json", help="File to log training progress")
    parser.add_argument("--visualize", action="store_true", help="Visualize the training")
    
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
        save_freq=args.save_freq
    ) 