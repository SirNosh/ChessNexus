# ChessNexus: Strategic Self-Play Chess AI

A sophisticated chess environment implemented in Python with advanced reinforcement learning capabilities. This environment allows you to:

- Play chess with a graphical interface or command-line interface
- Validate moves according to chess rules
- Track game state and detect checkmate/stalemate
- Train reinforcement learning agents
- Train agents through self-play
- Visualize training progress
- Run tournaments between different model versions

## Setup

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Run the game:
```
# For graphical interface
python chess_game.py

# For command-line interface
python chess_cli.py

# For reinforcement learning example (random agent)
python chess_env.py

# For DQN agent information
python dqn_agent.py

# For self-play training
python self_play.py

# For visualizing training progress
python visualize_training.py --log_file training_log.json

# For running a tournament between models
python tournament.py --model_dir models --num_games 5
```

## Graphical Interface Controls

- Click on a piece to select it
- Click on a valid square to move the selected piece
- Press 'r' to reset the game
- Press 'u' to undo a move
- Press 'q' to quit

## Command-Line Interface

The command-line interface uses algebraic notation for moves:

- Enter moves in the format 'e2e4' (from e2 to e4)
- Type 'quit' to exit the game
- Type 'reset' to reset the board
- Type 'undo' to undo the last move
- Type 'help' to show instructions

## Reinforcement Learning Environment

The `ChessEnv` class provides a reinforcement learning environment with an API similar to OpenAI Gym:

```python
from chess_env import ChessEnv

# Create the environment
env = ChessEnv()

# Reset the environment
observation = env.reset()

# Take a step in the environment
action = 0  # This is an encoded action (see chess_env.py for details)
observation, reward, done, info = env.step(action)

# Get valid actions
valid_actions = env.get_valid_actions()

# Render the environment
env.render()
```

The observation is a 8x8x12 numpy array representing the board state, where each channel corresponds to a piece type and color.

The action space is discrete with 4096 possible actions (64 starting squares × 64 ending squares).

## DQN Agent

The project includes a Deep Q-Network (DQN) agent that can be trained to play chess:

```python
from dqn_agent import train_dqn, play_with_trained_agent

# Train a new agent
train_dqn()

# Play with a trained agent
play_with_trained_agent("dqn_chess_model_final.weights.h5")
```

The DQN agent uses a convolutional neural network to learn the Q-values of different actions in different states. The agent is trained using experience replay and a target network to stabilize learning.

To train the agent, uncomment the `train_dqn()` line in `dqn_agent.py`. To play with a trained agent, uncomment the `play_with_trained_agent()` line.

## Self-Play Training

The project includes a self-play training mechanism that allows the agent to improve by playing against itself:

```python
from self_play import self_play_training, play_against_model

# Train the agent using self-play
agent = self_play_training(episodes=500, save_freq=100)

# Play against the trained model
play_against_model("models/chess_model_final.weights.h5", human_player="white")
```

Self-play training offers several advantages:
- No need for a pre-existing opponent or labeled data
- The agent continuously faces an opponent of appropriate skill level
- The agent can discover novel strategies through exploration

The self-play implementation includes:
- Separate experience tracking for white and black pieces
- Comprehensive chess-specific reward structure
- Periodic saving of model checkpoints
- A human-vs-AI interface to play against the trained model

### Advanced Reward Structure

The self-play training uses a sophisticated reward system that incorporates chess principles:

1. **Strategic Elements**:
   - Center control: Rewards for occupying or controlling central squares
   - Piece development: Rewards for developing knights and bishops in the opening
   - Pawn structure: Penalties for isolated pawns, rewards for connected pawns

2. **Tactical Awareness**:
   - Mobility: Rewards for moves that increase piece mobility
   - King safety: Rewards for castling, penalties for exposing the king in early/middle game
   - Check: Rewards for putting the opponent in check

3. **Phase-Specific Rewards**:
   - Opening: Focus on development, center control, and castling
   - Middlegame: Focus on tactical opportunities and piece coordination
   - Endgame: Focus on pawn promotion and king activity

4. **Long-term Planning**:
   - Different reward weights based on game phase
   - Rewards for moves that contribute to long-term positional advantages
   - Balance between material gains and positional considerations

This reward structure helps the agent learn chess principles beyond simple material counting, leading to more sophisticated play.

## Training Visualization

The project includes a visualization tool to track the progress of self-play training:

```python
# Create sample visualization data
python visualize_training.py --create_sample

# Visualize training progress
python visualize_training.py --log_file training_log.json
```

The visualization tool generates plots for:
- Average reward per episode
- Win rates for white, black, and draws
- Epsilon decay over time
- Cumulative wins

The training progress is automatically logged during self-play training and saved to a JSON file, which can be visualized at any time.

## Model Tournament

The project includes a tournament system to evaluate and compare different versions of trained models:

```python
# Run a tournament between all models in the models directory
python tournament.py --model_dir models --num_games 5

# Run a tournament between specific models
python tournament.py --models chess_model_episode_100.weights.h5 chess_model_episode_200.weights.h5 chess_model_final.weights.h5 --num_games 10

# Run a tournament with game rendering
python tournament.py --render
```

The tournament system:
- Pits each model against all other models
- Plays multiple games with each model as both white and black
- Calculates win/loss/draw statistics
- Ranks models based on tournament performance (1 point for a win, 0.5 for a draw)

This is useful for:
- Evaluating the improvement of models over training
- Identifying the strongest model
- Understanding if newer models are actually better than older ones

## Files

- `chess_engine.py`: Core chess logic and rules
- `chess_game.py`: Pygame interface for playing chess
- `chess_cli.py`: Command-line interface for playing chess
- `chess_env.py`: Reinforcement learning environment
- `dqn_agent.py`: Deep Q-Network agent for learning to play chess
- `self_play.py`: Self-play training implementation
- `visualize_training.py`: Tool for visualizing training progress
- `tournament.py`: Tournament system for comparing models

## Features

- Complete chess rules implementation including:
  - Castling (kingside and queenside)
  - En passant captures
  - Pawn promotion (automatically to queen)
  - Check, checkmate, and stalemate detection
- Move validation
- Game state tracking
- Undo move functionality
- Reinforcement learning environment with rewards for:
  - Capturing pieces (based on piece value)
  - Checkmate (+100)
  - Avoiding stalemate (-50)
  - Making valid moves
- DQN agent for learning to play chess
- Self-play training with experience memory
- Training progress visualization
- Model comparison through tournaments 