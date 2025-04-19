"""
dqn.py - Deep Q-Network implementation for Connect Four AI

This module provides a neural network model and DQN agent for
learning to play Connect Four through reinforcement learning.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Any, Union

from connect4.debug import debug, DebugLevel
from connect4.utils import ROWS, COLS, Player
from connect4.ai.utils import board_to_state, state_to_tensor, get_valid_action_mask

class DQNModel(nn.Module):
    """
    Neural network model for DQN implementation.
    
    This model takes a board state as input and outputs Q-values for each action.
    The architecture is configurable, allowing for different hidden layer sizes.
    """
    
    def __init__(self, input_channels: int = 3, hidden_size: int = 128):
        """
        Initialize the neural network model.
        
        Args:
            input_channels: Number of input channels (default 3: empty, player 1, player 2)
            hidden_size: Size of the hidden layer
        """
        super(DQNModel, self).__init__()
        
        debug.debug(f"Initializing DQNModel with hidden_size={hidden_size}", "ai")
        
        # Input layer: takes 3xROWSxCOLS state representation
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = 32 * ROWS * COLS
        
        # Hidden layer
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        
        # Output layer: one value per column
        self.fc2 = nn.Linear(hidden_size, COLS)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, rows, cols)
            
        Returns:
            Output tensor of shape (batch_size, cols)
        """
        # Apply convolutions with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply hidden layer with ReLU
        x = F.relu(self.fc1(x))
        
        # Output layer (no activation - will be used with softmax later)
        x = self.fc2(x)
        
        return x
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        debug.info(f"Saving model to {path}", "ai")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'DQNModel':
        """
        Load a model from a file.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded DQNModel instance
        """
        debug.info(f"Loading model from {path}", "ai")
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()  # Set to evaluation mode
        return model


class DQNAgent:
    """
    Agent using DQN for Connect Four decision making.
    
    This agent uses the DQN algorithm to learn an optimal policy for playing
    Connect Four. It employs an epsilon-greedy strategy for exploration.
    """
    
    def __init__(self, model: Optional[DQNModel] = None, 
                target_model: Optional[DQNModel] = None,
                hidden_size: int = 128,
                learning_rate: float = 0.001,
                epsilon: float = 1.0,
                epsilon_decay: float = 0.995,
                epsilon_min: float = 0.01,
                gamma: float = 0.99,
                target_update_freq: int = 10):
        """
        Initialize a DQN agent.
        
        Args:
            model: Main DQN model (will create new if None)
            target_model: Target network for stable updates (will create new if None)
            hidden_size: Size of hidden layer if creating new models
            learning_rate: Learning rate for optimizer
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum exploration rate
            gamma: Discount factor for future rewards
            target_update_freq: How often to update target network (in training steps)
        """
        debug.debug("Initializing DQNAgent", "ai")
        
        # Create models if not provided
        if model is None:
            debug.debug(f"Creating new model with hidden_size={hidden_size}", "ai")
            self.model = DQNModel(hidden_size=hidden_size)
        else:
            self.model = model
        
        if target_model is None:
            debug.debug("Creating new target model", "ai")
            self.target_model = DQNModel(hidden_size=hidden_size)
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.target_model = target_model
        
        # Set target model to evaluation mode
        self.target_model.eval()
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Agent parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        # Training tracking
        self.train_step_counter = 0
    
    def get_action(self, board, valid_moves: List[int], training: bool = False) -> int:
        """
        Choose an action using the current policy.
        
        Args:
            board: Current game board
            valid_moves: List of valid column indices
            training: Whether agent is in training mode (affects exploration)
            
        Returns:
            Selected column index
        """
        # Default to random if no valid moves
        if not valid_moves:
            debug.warning("No valid moves available!", "ai")
            return -1
        
        # Explore with probability epsilon during training
        if training and random.random() < self.epsilon:
            action = random.choice(valid_moves)
            debug.trace(f"Random exploration action: {action}", "ai")
            return action
        
        # Otherwise, use the model to pick the best action
        state = board_to_state(board.grid)
        state_tensor = state_to_tensor(state)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)
        
        # Create mask for valid moves
        mask = torch.ones(COLS) * float('-inf')
        mask[valid_moves] = 0.0
        
        # Apply mask and get best action
        masked_q_values = q_values + mask
        action = torch.argmax(masked_q_values).item()
        
        debug.trace(f"Model-based action: {action}, Q-values: {q_values.tolist()}", "ai")
        
        return action
    
    def train(self, batch, update_target: bool = False) -> float:
        """
        Train the model on a batch of experiences.
        
        Args:
            batch: List of (state, action, reward, next_state, done) tuples
            update_target: Whether to update the target network
            
        Returns:
            Loss value from training
        """
        if not batch:
            return 0.0
        
        debug.debug(f"Training on batch of size {len(batch)}", "ai")
        
        # Unpack the batch - convert each item to tensor first, then stack
        states_list = [torch.tensor(state, dtype=torch.float32) for state, _, _, _, _ in batch]
        next_states_list = [torch.tensor(next_state, dtype=torch.float32) for _, _, _, next_state, _ in batch]
        
        states = torch.stack(states_list)
        actions = torch.tensor([action for _, action, _, _, _ in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([reward for _, _, reward, _, _ in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack(next_states_list)
        dones = torch.tensor([done for _, _, _, _, done in batch], dtype=torch.float32).unsqueeze(1)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            # Get max Q values from target model
            max_next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            
            # Compute target using Bellman equation
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update training step counter
        self.train_step_counter += 1
        
        # Update target network if needed
        if update_target or (self.train_step_counter % self.target_update_freq == 0):
            debug.debug("Updating target network", "ai")
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()
    
    def save(self, path: str) -> None:
        """
        Save the agent's models and parameters.
        
        Args:
            path: Base path for saving
        """
        debug.info(f"Saving agent to {path}", "ai")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save models
        self.model.save(f"{path}_model.pt")
        self.target_model.save(f"{path}_target.pt")
        
        # Save parameters
        params = {
            'epsilon': self.epsilon,
            'train_step_counter': self.train_step_counter
        }
        torch.save(params, f"{path}_params.pt")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'DQNAgent':
        """
        Load an agent from saved files.
        
        Args:
            path: Base path for loading
            **kwargs: Additional arguments for agent initialization
            
        Returns:
            Loaded DQNAgent instance
        """
        debug.info(f"Loading agent from {path}", "ai")
        
        # Load models
        model = DQNModel.load(f"{path}_model.pt")
        target_model = DQNModel.load(f"{path}_target.pt")
        
        # Create agent with loaded models
        agent = cls(model=model, target_model=target_model, **kwargs)
        
        # Load parameters if they exist
        params_path = f"{path}_params.pt"
        if os.path.exists(params_path):
            params = torch.load(params_path)
            agent.epsilon = params['epsilon']
            agent.train_step_counter = params['train_step_counter']
        
        return agent