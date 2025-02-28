import numpy as np
import os
import argparse
from chess_env import ChessEnv
from self_play import SelfPlayAgent

class Tournament:
    def __init__(self, model_dir="models", visualize=False):
        self.model_dir = model_dir
        self.env = ChessEnv(visualize=visualize)
        self.state_shape = self.env.observation_space_shape
        self.action_size = self.env.action_space_size
        self.visualize = visualize
    
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
        
        # Determine the winner
        if self.env.engine.checkmate:
            winner = "Black" if self.env.engine.white_to_move else "White"
        elif self.env.engine.stalemate:
            winner = "Draw"
        else:
            winner = "Draw (max moves reached)"
        
        return winner, move_count
    
    def run_tournament(self, model_paths, num_games=10, render=False):
        """Run a tournament between multiple models"""
        if len(model_paths) < 2:
            print("Need at least 2 models for a tournament")
            return
        
        # Load all agents
        agents = []
        for path in model_paths:
            agents.append(self.load_agent(path))
        
        # Initialize results matrix
        num_agents = len(agents)
        results = np.zeros((num_agents, num_agents, 3))  # [wins, losses, draws]
        
        # Play matches between all pairs of agents
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                print(f"Match between {os.path.basename(model_paths[i])} and {os.path.basename(model_paths[j])}")
                
                # Play multiple games
                for game in range(num_games):
                    print(f"  Game {game+1}/{num_games}")
                    
                    # Alternate who plays white
                    if game % 2 == 0:
                        white_agent, black_agent = agents[i], agents[j]
                        white_idx, black_idx = i, j
                    else:
                        white_agent, black_agent = agents[j], agents[i]
                        white_idx, black_idx = j, i
                    
                    # Play the match
                    winner, moves = self.play_match(white_agent, black_agent, render=render)
                    print(f"    Winner: {winner} in {moves} moves")
                    
                    # Update results
                    if winner == "White":
                        results[white_idx, black_idx, 0] += 1
                        results[black_idx, white_idx, 1] += 1
                    elif winner == "Black":
                        results[black_idx, white_idx, 0] += 1
                        results[white_idx, black_idx, 1] += 1
                    else:  # Draw
                        results[white_idx, black_idx, 2] += 1
                        results[black_idx, white_idx, 2] += 1
        
        # Print results
        print("\nTournament Results:")
        print("Model".ljust(30) + "Wins".rjust(10) + "Losses".rjust(10) + "Draws".rjust(10) + "Win %".rjust(10))
        print("-" * 70)
        
        for i in range(num_agents):
            wins = np.sum(results[i, :, 0])
            losses = np.sum(results[i, :, 1])
            draws = np.sum(results[i, :, 2])
            total = wins + losses + draws
            win_pct = (wins + 0.5 * draws) / total * 100 if total > 0 else 0
            
            model_name = os.path.basename(model_paths[i])
            print(f"{model_name[:30].ljust(30)}{int(wins):10d}{int(losses):10d}{int(draws):10d}{win_pct:10.1f}")
        
        return results

def find_model_files(model_dir="models", pattern="chess_model_episode_"):
    """Find all model files in the given directory"""
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist")
        return []
    
    model_files = []
    for filename in os.listdir(model_dir):
        if filename.startswith(pattern) and filename.endswith(".weights.h5"):
            model_files.append(os.path.join(model_dir, filename))
    
    return sorted(model_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a chess tournament between trained models")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory containing model files")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games to play between each pair of models")
    parser.add_argument("--render", action="store_true", help="Render the games")
    parser.add_argument("--visualize", action="store_true", help="Use the graphical visualizer")
    args = parser.parse_args()
    
    # Find model files
    model_files = find_model_files(args.model_dir)
    
    if not model_files:
        print(f"No model files found in {args.model_dir}")
    else:
        # Run tournament
        tournament = Tournament(model_dir=args.model_dir, visualize=args.visualize)
        tournament.run_tournament(model_files, num_games=args.num_games, render=args.render) 