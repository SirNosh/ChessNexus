import numpy as np
from chess_engine import ChessEngine
from chess_visualizer import ChessVisualizer

class ChessEnv:
    """
    A reinforcement learning environment for chess.
    This environment follows a similar API to OpenAI Gym environments.
    """
    
    def __init__(self, visualize=False):
        self.engine = ChessEngine()
        self.action_space_size = 64 * 64  # All possible moves from any square to any square
        self.observation_space_shape = (8, 8, 12)  # 8x8 board with 12 channels (6 piece types x 2 colors)
        self.visualize = visualize
        self.visualizer = None
        if visualize:
            self.visualizer = ChessVisualizer()
        self.reset()
    
    def reset(self):
        """Reset the environment to the initial state"""
        self.engine.reset_board()
        if self.visualize and self.visualizer:
            self.visualizer.update(self.engine.board)
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: An integer in [0, 4095] representing a move from one square to another.
                   The action is encoded as (start_row * 8 + start_col) * 64 + (end_row * 8 + end_col)
        
        Returns:
            observation: The current state of the board
            reward: The reward for the action
            done: Whether the game is over
            info: Additional information
        """
        # Decode the action
        action_idx = action
        start_square_idx = action_idx // 64
        end_square_idx = action_idx % 64
        
        start_row, start_col = start_square_idx // 8, start_square_idx % 8
        end_row, end_col = end_square_idx // 8, end_square_idx % 8
        
        start = (start_row, start_col)
        end = (end_row, end_col)
        
        # Check if the move is valid
        valid_moves = self.engine.get_valid_moves()
        move_is_valid = False
        
        for move in valid_moves:
            if move[0] == start and move[1] == end:
                move_is_valid = True
                break
        
        if not move_is_valid:
            # If the move is invalid, return the current state with a negative reward
            return self._get_observation(), -10, False, {"valid_move": False}
        
        # Make the move
        captured_piece = self.engine.board[end[0]][end[1]]
        self.engine.make_move(start, end)
        
        # Check if the game is over
        self.engine.check_game_state()
        done = self.engine.checkmate or self.engine.stalemate
        
        # Calculate reward
        reward = 0
        
        # Reward for capturing pieces
        if captured_piece != 0:
            # Piece values: pawn=1, knight/bishop=3, rook=5, queen=9
            piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
            piece_type = abs(captured_piece)
            reward += piece_values.get(piece_type, 0)
        
        # Reward for checkmate
        if self.engine.checkmate:
            reward += 100
        
        # Update the visualizer if enabled
        if self.visualize and self.visualizer:
            self.visualizer.update(self.engine.board, last_move=(start, end))
        
        return self._get_observation(), reward, done, {"valid_move": True}
    
    def _get_observation(self):
        """Convert the board state to a neural network friendly format"""
        # Create a 8x8x12 observation (6 piece types x 2 colors)
        observation = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Map piece values to channel indices
        piece_to_channel = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,  # White pieces
            -1: 6, -2: 7, -3: 8, -4: 9, -5: 10, -6: 11  # Black pieces
        }
        
        # Fill the observation
        for row in range(8):
            for col in range(8):
                piece = self.engine.board[row][col]
                if piece != 0:
                    channel = piece_to_channel[piece]
                    observation[row, col, channel] = 1
        
        return observation
    
    def render(self, mode='human'):
        """Render the current state of the environment"""
        if mode == 'human':
            if self.visualize and self.visualizer:
                self.visualizer.update(self.engine.board)
            else:
                self.engine.print_board()
        else:
            return self._get_observation()
    
    def get_valid_actions(self):
        """Get a list of valid actions in the current state"""
        valid_moves = self.engine.get_valid_moves()
        valid_actions = []
        
        for move in valid_moves:
            start_row, start_col = move[0]
            end_row, end_col = move[1]
            
            start_idx = start_row * 8 + start_col
            end_idx = end_row * 8 + end_col
            
            action = start_idx * 64 + end_idx
            valid_actions.append(action)
        
        return valid_actions
    
    def close(self):
        """Close the environment"""
        if self.visualize and self.visualizer:
            self.visualizer.close()


# Example usage
if __name__ == "__main__":
    env = ChessEnv()
    obs = env.reset()
    
    # Play a random game
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            print("No valid actions!")
            break
        
        # Choose a random action
        action = np.random.choice(valid_actions)
        
        # Take the action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Reward: {reward}, Total: {total_reward}")
        
        if done:
            print("Game over!")
            env.render()
            break 