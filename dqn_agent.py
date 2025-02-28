import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from chess_env import ChessEnv

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.state_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Reshape state for the model
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        
        # Filter for valid actions only
        valid_act_values = {action: act_values[0][action] for action in valid_actions}
        return max(valid_act_values, key=valid_act_values.get)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def train_dqn():
    # Create the chess environment
    env = ChessEnv()
    
    # Get the state shape and action size
    state_shape = env.observation_space_shape
    action_size = env.action_space_size
    
    # Create the agent
    agent = DQNAgent(state_shape, action_size)
    
    # Training parameters
    batch_size = 32
    episodes = 100
    
    for e in range(episodes):
        # Reset the environment
        state = env.reset()
        total_reward = 0
        
        done = False
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Choose an action
            action = agent.act(state, valid_actions)
            
            # Take the action
            next_state, reward, done, _ = env.step(action)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # Train the agent
            agent.replay(batch_size)
        
        # Update the target model
        if e % 10 == 0:
            agent.update_target_model()
        
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        
        # Save the model
        if e % 50 == 0:
            agent.save(f"dqn_chess_model_{e}.h5")
    
    # Save the final model
    agent.save("dqn_chess_model_final.h5")


def play_with_trained_agent(model_path):
    # Create the chess environment
    env = ChessEnv()
    
    # Get the state shape and action size
    state_shape = env.observation_space_shape
    action_size = env.action_space_size
    
    # Create the agent
    agent = DQNAgent(state_shape, action_size)
    
    # Load the trained model
    agent.load(model_path)
    agent.epsilon = 0.01  # Set a small epsilon for some exploration
    
    # Play a game
    state = env.reset()
    env.render()
    
    done = False
    total_reward = 0
    
    while not done:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            break
        
        # Choose an action
        action = agent.act(state, valid_actions)
        
        # Take the action
        next_state, reward, done, _ = env.step(action)
        
        # Update state and total reward
        state = next_state
        total_reward += reward
        
        # Render the environment
        env.render()
        print(f"Reward: {reward}, Total: {total_reward}")
        
        if done:
            print("Game over!")


if __name__ == "__main__":
    # Uncomment to train the agent
    train_dqn()
    
    # Uncomment to play with a trained agent
    # play_with_trained_agent("dqn_chess_model_final.h5")
    
    # For now, just print a message
    print("This is a DQN agent for the chess environment.")
    print("Uncomment the appropriate function in the main block to train or play with the agent.") 