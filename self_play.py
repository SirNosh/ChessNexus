import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os
import json
from chess_env import ChessEnv
from visualize_training import log_training_progress

class SelfPlayAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.batch_size = 64
        self.update_target_every = 5  # Update target network every 5 episodes

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.state_shape))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions, training=True):
        # Epsilon-greedy action selection
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Reshape state for the model
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        
        # Filter for valid actions only
        valid_act_values = {action: act_values[0][action] for action in valid_actions}
        
        # If no valid actions with positive values, choose randomly
        if not valid_act_values or max(valid_act_values.values()) <= 0:
            return random.choice(valid_actions)
            
        return max(valid_act_values, key=valid_act_values.get)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size,) + self.state_shape)
        targets = np.zeros((self.batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            if done:
                target = reward
            else:
                # Double DQN: select action using model, evaluate using target model
                next_state_expanded = np.expand_dims(next_state, axis=0)
                next_action = np.argmax(self.model.predict(next_state_expanded, verbose=0)[0])
                target = reward + self.gamma * self.target_model.predict(next_state_expanded, verbose=0)[0][next_action]
            
            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            target_f[0][action] = target
            targets[i] = target_f[0]
        
        # Train the model on the batch
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        # Ensure filename ends with .weights.h5
        if not name.endswith('.weights.h5'):
            if name.endswith('.h5'):
                name = name.replace('.h5', '.weights.h5')
            else:
                name = name + '.weights.h5'
        self.model.save_weights(name)


def calculate_reward(env, move, captured_piece, move_count):
    """Calculate a comprehensive reward based on chess principles"""
    reward = 0
    board = env.engine.board
    
    # Determine game phase (opening, middlegame, endgame)
    total_pieces = np.count_nonzero(board)
    if move_count < 10:
        phase = "opening"
    elif total_pieces < 10:
        phase = "endgame"
    else:
        phase = "middlegame"
    
    # 1. Material rewards
    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # Pawn, Knight, Bishop, Rook, Queen, King
    if captured_piece != 0:
        reward += piece_values[abs(captured_piece)]
    
    # 2. Center control (reward for controlling or occupying center squares)
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    center_control = 0
    for row, col in center_squares:
        piece = board[row][col]
        if (env.engine.white_to_move and piece > 0) or (not env.engine.white_to_move and piece < 0):
            center_control += 0.3
    reward += center_control
    
    # 3. Piece development (reward for moving pieces off their starting positions)
    if phase == "opening":
        # Reward for developing knights and bishops early
        if move[0] in [(7, 1), (7, 6)] and board[move[0]] in [2, -2]:  # Knight development
            reward += 0.5
        elif move[0] in [(7, 2), (7, 5)] and board[move[0]] in [3, -3]:  # Bishop development
            reward += 0.5
        # Small penalty for moving the queen too early
        elif move[0] == (7, 3) and board[move[0]] in [5, -5] and move_count < 6:
            reward -= 0.3
    
    # 4. Mobility (reward for moves that increase piece mobility)
    start_row, start_col = move[0]
    end_row, end_col = move[1]
    
    # Reward for moving to squares that control more of the board
    if abs(board[start_row][start_col]) in [2, 3, 5]:  # Knight, Bishop, Queen
        # More central positions generally control more squares
        central_distance = abs(end_row - 3.5) + abs(end_col - 3.5)
        mobility_reward = max(0, (4 - central_distance) * 0.1)
        reward += mobility_reward
    
    # 5. King safety
    if phase != "endgame":
        # Penalize moving the king in opening/middlegame (except castling)
        if abs(board[start_row][start_col]) == 6:  # King
            # Check if this is a castling move
            is_castling = abs(start_col - end_col) > 1
            if is_castling:
                reward += 1.0  # Good reward for castling
            else:
                reward -= 0.5  # Penalty for moving king otherwise
    else:
        # In endgame, reward king activity
        if abs(board[start_row][start_col]) == 6:  # King
            # Reward king moving toward center in endgame
            central_distance = abs(end_row - 3.5) + abs(end_col - 3.5)
            reward += max(0, (4 - central_distance) * 0.2)
    
    # 6. Pawn structure
    if abs(board[start_row][start_col]) == 1:  # Pawn
        # Reward for advancing pawns in middlegame/endgame
        if phase != "opening":
            direction = -1 if env.engine.white_to_move else 1
            if (end_row - start_row) * direction < 0:  # Moving forward
                reward += 0.1 * abs(end_row - start_row)
        
        # Penalty for isolated pawns
        has_neighbor = False
        for col_offset in [-1, 1]:
            if 0 <= end_col + col_offset < 8:
                for row in range(8):
                    if board[row][end_col + col_offset] == (1 if env.engine.white_to_move else -1):
                        has_neighbor = True
                        break
        if not has_neighbor:
            reward -= 0.2
    
    # 7. Phase-specific priorities
    if phase == "opening":
        # In opening, development and center control are key
        reward *= 1.2  # Amplify the rewards for good opening play
    elif phase == "middlegame":
        # In middlegame, tactical awareness matters more
        if env.engine.is_in_check(not env.engine.white_to_move):
            reward += 0.7  # Reward for putting opponent in check
    else:  # endgame
        # In endgame, pawn promotion and king activity matter more
        if abs(board[start_row][start_col]) == 1:  # Pawn
            # Reward pawns advancing toward promotion
            promotion_progress = 7 - end_row if env.engine.white_to_move else end_row
            reward += promotion_progress * 0.15
    
    # 8. Checkmate and stalemate
    if env.engine.checkmate:
        reward += 100  # Big reward for checkmate
    elif env.engine.stalemate:
        reward -= 50   # Penalty for stalemate
    
    return reward


def self_play_training(episodes=1000, save_freq=100, model_dir="models", log_file="training_log.json", visualize=False):
    """
    Train a chess agent using self-play.
    
    Args:
        episodes: Number of episodes to train for
        save_freq: How often to save the model
        model_dir: Directory to save models
        log_file: File to log training progress
        visualize: Whether to visualize the games
    """
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Initialize environment and agent
    env = ChessEnv(visualize=visualize)
    state_shape = env.observation_space_shape
    action_size = env.action_space_size
    
    agent = SelfPlayAgent(state_shape, action_size)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    win_rates = {"white": [], "black": [], "draw": []}
    
    # Load existing model if available
    latest_model = None
    for file in os.listdir(model_dir):
        if file.endswith(".weights.h5"):
            latest_model = file
    
    if latest_model:
        model_path = os.path.join(model_dir, latest_model)
        print(f"Loading existing model: {model_path}")
        agent.load(model_path)
        
        # Extract episode number from filename
        try:
            episode_start = int(latest_model.split("_")[-1].split(".")[0])
            print(f"Continuing training from episode {episode_start}")
        except:
            episode_start = 0
    else:
        episode_start = 0
    
    # Load training log if it exists
    training_log = {"episode_rewards": [], "episode_lengths": [], "win_rates": {"white": [], "black": [], "draw": []}}
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                training_log = json.load(f)
            print(f"Loaded training log from {log_file}")
        except:
            print(f"Could not load training log from {log_file}, starting fresh")
    
    # Main training loop
    for episode in range(episode_start, episode_start + episodes):
        print(f"Episode {episode+1}/{episode_start + episodes}")
        
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        move_count = 0
        
        # Play one episode
        while not done and move_count < 200:  # Limit to 200 moves to prevent infinite games
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Choose action
            action = agent.act(state, valid_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            move_count += 1
            
            # Train the agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
        
        # Update target network periodically
        if episode % agent.update_target_every == 0:
            agent.update_target_model()
        
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(move_count)
        
        # Determine winner
        if env.engine.checkmate:
            winner = "black" if env.engine.white_to_move else "white"
        elif env.engine.stalemate:
            winner = "draw"
        else:
            winner = "draw"  # Default to draw if game ended for other reasons
        
        win_rates[winner].append(1)
        for w in ["white", "black", "draw"]:
            if w != winner:
                win_rates[w].append(0)
        
        # Update training log
        training_log["episode_rewards"].append(float(episode_reward))
        training_log["episode_lengths"].append(move_count)
        for w in ["white", "black", "draw"]:
            training_log["win_rates"][w].append(win_rates[w][-1])
        
        # Save training log
        with open(log_file, 'w') as f:
            json.dump(training_log, f)
        
        # Print progress
        print(f"Episode {episode+1}: {move_count} moves, reward={episode_reward:.2f}, epsilon={agent.epsilon:.4f}, winner={winner}")
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            model_path = os.path.join(model_dir, f"chess_model_episode_{episode+1}.weights.h5")
            agent.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Log training progress
            log_training_progress(log_file)
    
    # Save final model
    model_path = os.path.join(model_dir, f"chess_model_episode_{episode_start + episodes}.weights.h5")
    agent.save(model_path)
    print(f"Final model saved to {model_path}")
    
    # Log final training progress
    log_training_progress(log_file)
    
    return agent


def play_against_model(model_path, human_player="white", visualize=True):
    """
    Play a game against a trained model.
    
    Args:
        model_path: Path to the model file
        human_player: Which side the human plays ("white" or "black")
        visualize: Whether to use the graphical visualizer
    """
    # Initialize environment and agent
    env = ChessEnv(visualize=visualize)
    state_shape = env.observation_space_shape
    action_size = env.action_space_size
    
    agent = SelfPlayAgent(state_shape, action_size)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during play
    
    # Reset environment
    state = env.reset()
    done = False
    move_count = 0
    
    # Render initial state
    env.render()
    
    # Main game loop
    while not done and move_count < 200:  # Limit to 200 moves
        # Determine whose turn it is
        is_human_turn = (env.engine.white_to_move and human_player == "white") or \
                        (not env.engine.white_to_move and human_player == "black")
        
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            break
        
        if is_human_turn:
            # Human player's turn
            print("Your turn!")
            
            # Display valid moves
            print("Valid moves:")
            for i, action in enumerate(valid_actions):
                start_idx = action // 64
                end_idx = action % 64
                
                start_row, start_col = start_idx // 8, start_idx % 8
                end_row, end_col = end_idx // 8, end_idx % 8
                
                start_alg = chr(start_col + ord('a')) + str(8 - start_row)
                end_alg = chr(end_col + ord('a')) + str(8 - end_row)
                
                print(f"{i+1}: {start_alg}{end_alg}")
            
            # Get human move
            valid_input = False
            while not valid_input:
                try:
                    move_input = input("Enter move (e.g., 'e2e4' or move number): ")
                    
                    if move_input.isdigit():
                        # Input is a move number
                        move_idx = int(move_input) - 1
                        if 0 <= move_idx < len(valid_actions):
                            action = valid_actions[move_idx]
                            valid_input = True
                        else:
                            print(f"Invalid move number. Please enter a number between 1 and {len(valid_actions)}")
                    else:
                        # Input is algebraic notation
                        if len(move_input) != 4:
                            print("Invalid format. Please use format 'e2e4'")
                            continue
                        
                        start_col = ord(move_input[0].lower()) - ord('a')
                        start_row = 8 - int(move_input[1])
                        end_col = ord(move_input[2].lower()) - ord('a')
                        end_row = 8 - int(move_input[3])
                        
                        if not (0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8):
                            print("Invalid coordinates. Row must be 1-8, column must be a-h")
                            continue
                        
                        start_idx = start_row * 8 + start_col
                        end_idx = end_row * 8 + end_col
                        action = start_idx * 64 + end_idx
                        
                        if action not in valid_actions:
                            print("Invalid move. Please choose from the list of valid moves")
                            continue
                        
                        valid_input = True
                except Exception as e:
                    print(f"Error: {e}")
                    print("Invalid input. Please try again")
        else:
            # AI player's turn
            print("AI thinking...")
            action = agent.act(state, valid_actions, training=False)
            
            # Decode the action for display
            start_idx = action // 64
            end_idx = action % 64
            
            start_row, start_col = start_idx // 8, start_idx % 8
            end_row, end_col = end_idx // 8, end_idx % 8
            
            start_alg = chr(start_col + ord('a')) + str(8 - start_row)
            end_alg = chr(end_col + ord('a')) + str(8 - end_row)
            
            print(f"AI moves: {start_alg}{end_alg}")
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        
        # Update state and counters
        state = next_state
        move_count += 1
        
        # Render the board
        env.render()
    
    # Game over
    if env.engine.checkmate:
        winner = "Black" if env.engine.white_to_move else "White"
        print(f"Checkmate! {winner} wins!")
    elif env.engine.stalemate:
        print("Stalemate! It's a draw.")
    else:
        print("Game ended. It's a draw (max moves reached).")
    
    return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess self-play training")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--save_freq", type=int, default=100, help="How often to save the model")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_file", type=str, default="training_log.json", help="File to log training progress")
    parser.add_argument("--play", action="store_true", help="Play against a trained model")
    parser.add_argument("--model", type=str, help="Model to play against")
    parser.add_argument("--human_player", type=str, default="white", choices=["white", "black"], help="Which side the human plays")
    parser.add_argument("--visualize", action="store_true", help="Use the graphical visualizer")
    
    args = parser.parse_args()
    
    if args.play:
        if not args.model:
            print("Please specify a model to play against with --model")
            exit(1)
        play_against_model(args.model, human_player=args.human_player, visualize=args.visualize)
    else:
        self_play_training(episodes=args.episodes, save_freq=args.save_freq, model_dir=args.model_dir, log_file=args.log_file, visualize=args.visualize) 