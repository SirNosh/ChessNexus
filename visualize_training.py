import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse

def plot_training_progress(log_file):
    """Plot the training progress from a log file"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    episodes = data['episodes']
    rewards = data['rewards']
    win_rates = data['win_rates']
    epsilons = data['epsilons']
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axs[0, 0].plot(episodes, rewards)
    axs[0, 0].set_title('Average Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].grid(True)
    
    # Plot win rates
    axs[0, 1].plot(episodes, [wr['white'] for wr in win_rates], label='White')
    axs[0, 1].plot(episodes, [wr['black'] for wr in win_rates], label='Black')
    axs[0, 1].plot(episodes, [wr['draw'] for wr in win_rates], label='Draw')
    axs[0, 1].set_title('Win Rates')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Win Rate')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot epsilon decay
    axs[1, 0].plot(episodes, epsilons)
    axs[1, 0].set_title('Epsilon Decay')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Epsilon')
    axs[1, 0].grid(True)
    
    # Plot cumulative wins
    white_wins = np.cumsum([wr['white'] for wr in win_rates])
    black_wins = np.cumsum([wr['black'] for wr in win_rates])
    draws = np.cumsum([wr['draw'] for wr in win_rates])
    
    axs[1, 1].plot(episodes, white_wins, label='White')
    axs[1, 1].plot(episodes, black_wins, label='Black')
    axs[1, 1].plot(episodes, draws, label='Draw')
    axs[1, 1].set_title('Cumulative Wins')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Number of Wins')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

def log_training_progress(log_file, episodes, rewards, win_rates, epsilons):
    """Log the training progress to a file"""
    data = {
        'episodes': episodes,
        'rewards': rewards,
        'win_rates': win_rates,
        'epsilons': epsilons
    }
    
    with open(log_file, 'w') as f:
        json.dump(data, f)

def create_sample_data():
    """Create sample data for testing"""
    episodes = list(range(0, 500, 10))
    rewards = [np.random.normal(-5, 10) + i/50 for i in range(len(episodes))]
    
    # Smooth the rewards
    rewards = np.convolve(rewards, np.ones(5)/5, mode='valid').tolist()
    episodes = episodes[:len(rewards)]
    
    win_rates = []
    for i in range(len(episodes)):
        white_rate = min(0.8, 0.3 + i/len(episodes) * 0.5)
        black_rate = min(0.8, 0.3 + i/len(episodes) * 0.3)
        draw_rate = 1 - white_rate - black_rate
        win_rates.append({
            'white': white_rate,
            'black': black_rate,
            'draw': draw_rate
        })
    
    epsilons = [1.0 * (0.9995 ** i) for i in episodes]
    
    return episodes, rewards, win_rates, epsilons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training progress')
    parser.add_argument('--log_file', type=str, default='training_log.json',
                        help='Path to the training log file')
    parser.add_argument('--create_sample', action='store_true',
                        help='Create sample data for testing')
    
    args = parser.parse_args()
    
    if args.create_sample:
        print("Creating sample data...")
        episodes, rewards, win_rates, epsilons = create_sample_data()
        log_training_progress(args.log_file, episodes, rewards, win_rates, epsilons)
        print(f"Sample data saved to {args.log_file}")
    
    plot_training_progress(args.log_file) 