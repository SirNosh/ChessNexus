import numpy as np
import os
import argparse
from chess_env import ChessEnv
from self_play import SelfPlayAgent

class Tournament:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.env = ChessEnv()
        self.state_shape = self.env.observation_space_shape
        self.action_size = self.env.action_space_size
    
    def load_agent(self, model_path):
        """Load an agent from a model file"""
        agent = SelfPlayAgent(self.state_shape, self.action_size)
        agent.load(model_path)
        agent.epsilon = 0.05  # Small epsilon for some exploration
        return agent
    
    def play_match(self, white_agent, black_agent, render=False, max_moves=200):
        """Play a match between two agents"""
        state = self.env.reset()
        done = False
        move_count = 0
        
        if render:
            self.env.render()
        
        while not done and move_count < max_moves:
            # Get valid actions
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Choose agent based on whose turn it is
            if self.env.engine.white_to_move:
                agent = white_agent
            else:
                agent = black_agent
            
            # Choose an action
            action = agent.act(state, valid_actions, training=False)
            
            # Take the action
            next_state, reward, done, info = self.env.step(action)
            
            # Update state
            state = next_state
            move_count += 1
            
            if render:
                self.env.render()
                
                # Decode the action for display
                start_square_idx = action // 64
                end_square_idx = action % 64
                
                start_row, start_col = start_square_idx // 8, start_square_idx % 8
                end_row, end_col = end_square_idx // 8, end_square_idx % 8
                
                start_alg = chr(start_col + ord('a')) + str(8 - start_row)
                end_alg = chr(end_col + ord('a')) + str(8 - end_row)
                
                player = "White" if self.env.engine.white_to_move else "Black"
                print(f"{player} moves: {start_alg}{end_alg}")
        
        # Determine the winner
        if self.env.engine.checkmate:
            winner = "black" if self.env.engine.white_to_move else "white"
        elif self.env.engine.stalemate or move_count >= max_moves:
            winner = "draw"
        else:
            winner = "draw"  # Default to draw if game ended for other reasons
        
        return winner, move_count
    
    def run_tournament(self, model_paths, num_games=10, render=False):
        """Run a tournament between multiple models"""
        # Load agents
        agents = {}
        for model_path in model_paths:
            model_name = os.path.basename(model_path).replace(".weights.h5", "")
            agents[model_name] = self.load_agent(model_path)
        
        # Initialize results
        results = {name: {"wins": 0, "losses": 0, "draws": 0} for name in agents.keys()}
        
        # Play matches
        for i, white_name in enumerate(agents.keys()):
            for j, black_name in enumerate(agents.keys()):
                if i == j:  # Skip self-play
                    continue
                
                print(f"\nPlaying {num_games} games: {white_name} (White) vs {black_name} (Black)")
                
                white_wins = 0
                black_wins = 0
                draws = 0
                
                for game in range(num_games):
                    print(f"Game {game+1}/{num_games}")
                    winner, moves = self.play_match(agents[white_name], agents[black_name], render=render)
                    
                    if winner == "white":
                        white_wins += 1
                        results[white_name]["wins"] += 1
                        results[black_name]["losses"] += 1
                        print(f"White ({white_name}) wins in {moves} moves")
                    elif winner == "black":
                        black_wins += 1
                        results[white_name]["losses"] += 1
                        results[black_name]["wins"] += 1
                        print(f"Black ({black_name}) wins in {moves} moves")
                    else:
                        draws += 1
                        results[white_name]["draws"] += 1
                        results[black_name]["draws"] += 1
                        print(f"Draw after {moves} moves")
                
                print(f"Results: White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
        
        # Print tournament results
        print("\nTournament Results:")
        print("-------------------")
        
        # Calculate points (win=1, draw=0.5, loss=0)
        points = {}
        for name, result in results.items():
            points[name] = result["wins"] + 0.5 * result["draws"]
        
        # Sort by points
        sorted_results = sorted(points.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(sorted_results):
            print(f"{i+1}. {name}: {score} points ({results[name]['wins']} wins, {results[name]['draws']} draws, {results[name]['losses']} losses)")
        
        return results

def find_model_files(model_dir="models", pattern="chess_model_episode_"):
    """Find model files in the model directory"""
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found.")
        return []
    
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith(".weights.h5") and pattern in file:
            model_files.append(os.path.join(model_dir, file))
    
    return sorted(model_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a tournament between chess models')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing model files')
    parser.add_argument('--num_games', type=int, default=5,
                        help='Number of games to play between each pair of models')
    parser.add_argument('--render', action='store_true',
                        help='Render the games')
    parser.add_argument('--models', nargs='+',
                        help='Specific model files to use (if not specified, all models in the directory will be used)')
    
    args = parser.parse_args()
    
    # Find model files
    if args.models:
        model_paths = [os.path.join(args.model_dir, model) for model in args.models]
    else:
        model_paths = find_model_files(args.model_dir)
    
    if not model_paths:
        print("No model files found.")
        exit(1)
    
    print(f"Found {len(model_paths)} model files:")
    for path in model_paths:
        print(f"  {path}")
    
    # Run tournament
    tournament = Tournament(args.model_dir)
    tournament.run_tournament(model_paths, num_games=args.num_games, render=args.render) 