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


def self_play_training(episodes=1000, save_freq=100, model_dir="models", log_file="training_log.json"):
    """Train the agent using self-play"""
    # Create directory for saving models if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create the chess environment
    env = ChessEnv()
    
    # Get the state shape and action size
    state_shape = env.observation_space_shape
    action_size = env.action_space_size
    
    # Create the agent
    agent = SelfPlayAgent(state_shape, action_size)
    
    # Training statistics
    episode_rewards = []
    win_counts = {"white": 0, "black": 0, "draw": 0}
    
    # Logging data
    log_episodes = []
    log_rewards = []
    log_win_rates = []
    log_epsilons = []
    
    # Window size for calculating win rates
    window_size = 50
    
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        done = False
        total_reward = 0
        move_count = 0
        
        # Store experiences for both players
        white_experiences = []
        black_experiences = []
        
        # Play a game
        while not done and move_count < 200:  # Add move limit to prevent infinite games
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Choose an action
            action = agent.act(state, valid_actions)
            
            # Decode the action to get the move
            start_square_idx = action // 64
            end_square_idx = action % 64
            
            start_row, start_col = start_square_idx // 8, start_square_idx % 8
            end_row, end_col = end_square_idx // 8, end_square_idx % 8
            
            start = (start_row, start_col)
            end = (end_row, end_col)
            
            # Get the captured piece before making the move
            captured_piece = env.engine.board[end_row][end_col]
            
            # Take the action
            next_state, reward, done, info = env.step(action)
            
            # Calculate comprehensive reward
            reward = calculate_reward(env, (start, end), captured_piece, move_count)
            
            # Store the experience based on whose turn it is
            experience = (state, action, reward, next_state, done)
            if env.engine.white_to_move:
                white_experiences.append(experience)
            else:
                black_experiences.append(experience)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            move_count += 1
        
        # Determine the winner
        if env.engine.checkmate:
            winner = "black" if env.engine.white_to_move else "white"
            win_counts[winner] += 1
        elif env.engine.stalemate or move_count >= 200:
            win_counts["draw"] += 1
        
        # Process experiences and add to memory
        # For white's experiences
        for state, action, reward, next_state, done in white_experiences:
            # If white won, give positive reward to all its moves
            if not env.engine.white_to_move and env.engine.checkmate:
                reward = 1.0
            # If black won, give negative reward to all white's moves
            elif env.engine.white_to_move and env.engine.checkmate:
                reward = -1.0
            agent.remember(state, action, reward, next_state, done)
        
        # For black's experiences
        for state, action, reward, next_state, done in black_experiences:
            # If black won, give positive reward to all its moves
            if env.engine.white_to_move and env.engine.checkmate:
                reward = 1.0
            # If white won, give negative reward to all black's moves
            elif not env.engine.white_to_move and env.engine.checkmate:
                reward = -1.0
            agent.remember(state, action, reward, next_state, done)
        
        # Train the agent
        agent.replay()
        
        # Update target model periodically
        if episode % agent.update_target_every == 0:
            agent.update_target_model()
        
        # Save the model periodically
        if episode > 0 and episode % save_freq == 0:
            agent.save(f"{model_dir}/chess_model_episode_{episode}.weights.h5")
        
        # Track statistics
        episode_rewards.append(total_reward)
        
        # Log data for visualization every 10 episodes
        if episode % 10 == 0:
            # Calculate win rates over the last window_size episodes
            total_games = sum(win_counts.values())
            if total_games > 0:
                win_rate = {
                    "white": win_counts["white"] / total_games,
                    "black": win_counts["black"] / total_games,
                    "draw": win_counts["draw"] / total_games
                }
            else:
                win_rate = {"white": 0, "black": 0, "draw": 0}
            
            log_episodes.append(episode)
            log_rewards.append(np.mean(episode_rewards[-10:]) if episode_rewards else 0)
            log_win_rates.append(win_rate)
            log_epsilons.append(agent.epsilon)
            
            # Save the log
            log_training_progress(log_file, log_episodes, log_rewards, log_win_rates, log_epsilons)
            
            # Reset win counts for the next window
            if episode % window_size == 0 and episode > 0:
                win_counts = {"white": 0, "black": 0, "draw": 0}
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Episode: {episode}/{episodes}, Moves: {move_count}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            print(f"Wins - White: {win_counts['white']}, Black: {win_counts['black']}, Draws: {win_counts['draw']}")
    
    # Save the final model
    agent.save(f"{model_dir}/chess_model_final.weights.h5")
    
    print("Training complete!")
    print(f"Final statistics - Wins: White: {win_counts['white']}, Black: {win_counts['black']}, Draws: {win_counts['draw']}")
    
    return agent


def play_against_model(model_path, human_player="white"):
    """Play a game against the trained model"""
    # Create the chess environment
    env = ChessEnv()
    
    # Get the state shape and action size
    state_shape = env.observation_space_shape
    action_size = env.action_space_size
    
    # Create the agent
    agent = SelfPlayAgent(state_shape, action_size)
    
    # Load the trained model
    agent.load(model_path)
    agent.epsilon = 0.05  # Small epsilon for some exploration
    
    # Play a game
    state = env.reset()
    env.render()
    
    done = False
    move_count = 0
    
    # Determine if human plays as white or black
    human_is_white = human_player.lower() == "white"
    
    while not done and move_count < 100:
        current_player = "white" if env.engine.white_to_move else "black"
        
        # Human's turn
        if (human_is_white and current_player == "white") or (not human_is_white and current_player == "black"):
            print(f"\nYour turn ({current_player}):")
            
            # Get valid moves in algebraic notation for display
            valid_moves = env.engine.get_valid_moves()
            valid_moves_algebraic = []
            
            for move in valid_moves:
                start_row, start_col = move[0]
                end_row, end_col = move[1]
                start_alg = chr(start_col + ord('a')) + str(8 - start_row)
                end_alg = chr(end_col + ord('a')) + str(8 - end_row)
                valid_moves_algebraic.append(f"{start_alg}{end_alg}")
            
            # Display some valid moves as hints
            print(f"Some valid moves: {', '.join(valid_moves_algebraic[:5])}" + (" ..." if len(valid_moves_algebraic) > 5 else ""))
            
            # Get human move
            while True:
                move_input = input("Enter your move (e.g., e2e4) or 'quit' to exit: ").strip().lower()
                
                if move_input == 'quit':
                    return
                
                # Parse the move
                if len(move_input) != 4:
                    print("Invalid move format! Use format 'e2e4'.")
                    continue
                
                start_col = ord(move_input[0]) - ord('a')
                start_row = 8 - int(move_input[1])
                end_col = ord(move_input[2]) - ord('a')
                end_row = 8 - int(move_input[3])
                
                if not (0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8):
                    print("Move out of bounds! Try again.")
                    continue
                
                # Convert to action
                start_idx = start_row * 8 + start_col
                end_idx = end_row * 8 + end_col
                action = start_idx * 64 + end_idx
                
                # Check if the move is valid
                if move_input in valid_moves_algebraic:
                    break
                else:
                    print("Invalid move! Try again.")
            
            # Take the action
            next_state, reward, done, info = env.step(action)
            
        # AI's turn
        else:
            print(f"\nAI's turn ({current_player}):")
            
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Choose an action
            action = agent.act(state, valid_actions, training=False)
            
            # Decode the action for display
            start_square_idx = action // 64
            end_square_idx = action % 64
            
            start_row, start_col = start_square_idx // 8, start_square_idx % 8
            end_row, end_col = end_square_idx // 8, end_square_idx % 8
            
            start_alg = chr(start_col + ord('a')) + str(8 - start_row)
            end_alg = chr(end_col + ord('a')) + str(8 - end_row)
            
            print(f"AI moves: {start_alg}{end_alg}")
            
            # Take the action
            next_state, reward, done, info = env.step(action)
        
        # Update state
        state = next_state
        move_count += 1
        
        # Render the environment
        env.render()
        
        # Check for game end
        if env.engine.checkmate:
            winner = "Black" if env.engine.white_to_move else "White"
            print(f"\nCheckmate! {winner} wins!")
        elif env.engine.stalemate:
            print("\nStalemate! The game is a draw.")
    
    if move_count >= 100:
        print("\nGame ended due to move limit. It's a draw.")


if __name__ == "__main__":
    # Train the agent using self-play
    print("Starting self-play training...")
    agent = self_play_training(episodes=500, save_freq=100)
    
    # Uncomment to play against the trained model
    # play_against_model("models/chess_model_final.weights.h5", human_player="white") 