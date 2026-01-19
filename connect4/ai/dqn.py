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
    
    def __init__(self, input_channels: int = 4, hidden_sizes: Optional[List[int]] = None, hidden_size: int = 128, layers: int = 1):
        """
        Initialize the neural network model.
        
        Args:
            input_channels: Number of input channels (default 4: empty, player 1, player 2, side-to-move)
            hidden_sizes: List of sizes for each hidden layer (overrides layers parameter)
            hidden_size: Size of each hidden layer if hidden_sizes is not provided
            layers: Number of hidden layers (ignored if hidden_sizes is provided)
        """
        super(DQNModel, self).__init__()
        
        # Determine hidden layers configuration
        if hidden_sizes is None:
            hidden_sizes = [hidden_size] * layers
        
        debug.debug(f"Initializing DQNModel with {len(hidden_sizes)} layers: {hidden_sizes}", "ai")
        
        # Input layer: takes CxROWSxCOLS state representation
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = 32 * ROWS * COLS
        
        # Create dynamic fully connected layers
        self.fc_layers = nn.ModuleList()
        
        # Input to first hidden layer
        self.fc_layers.append(nn.Linear(conv_output_size, hidden_sizes[0]))
        
        # Additional hidden layers
        for i in range(1, len(hidden_sizes)):
            self.fc_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Output layer: one value per column
        self.output = nn.Linear(hidden_sizes[-1], COLS)
        
        # Store model configuration for later reference
        self.hidden_sizes = hidden_sizes
        self.input_channels = input_channels
    
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
        
        # Pass through each hidden layer
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        
        # Output layer
        x = self.output(x)
        
        return x
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        debug.info(f"Saving model to {path}", "ai")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict
        torch.save({
            'state_dict': self.state_dict(),
            'hidden_sizes': self.hidden_sizes,
            'input_channels': self.input_channels,
        }, path)
    
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
        
        # Load saved data
        saved_data = torch.load(path)
        
        # Create model with the same architecture
        # If caller didn't override, prefer saved input_channels for correctness.
        if 'input_channels' not in kwargs:
            kwargs['input_channels'] = saved_data.get('input_channels', 4)

        model = cls(hidden_sizes=saved_data.get('hidden_sizes', None), **kwargs)
        
        # Load the state dict
        model.load_state_dict(saved_data['state_dict'])
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
                hidden_size: int = 128,  # Changed from 256
                hidden_sizes: Optional[List[int]] = None,
                layers: int = 2,  # Changed from 1
                learning_rate: float = 0.00005,  # Changed from 0.0001
                epsilon: float = 1.0,
                epsilon_decay: float = 0.99999,  # Changed from 0.999919
                epsilon_min: float = 0.1,  # Changed from 0.2
                gamma: float = 0.99,
                target_update_freq: int = 100):  # Changed from 50
        """
        Initialize a DQN agent.
        
        Args:
            model: Main DQN model (will create new if None)
            target_model: Target network for stable updates (will create new if None)
            hidden_size: Size of hidden layer if creating new model with single layer
            hidden_sizes: List of sizes for multiple hidden layers (overrides hidden_size and layers)
            layers: Number of hidden layers if creating new model (all with size hidden_size)
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
            debug.debug(f"Creating new model with architecture: {hidden_sizes if hidden_sizes else [hidden_size] * layers}", "ai")
            self.model = DQNModel(hidden_sizes=hidden_sizes, hidden_size=hidden_size, layers=layers)
        else:
            self.model = model
        
        if target_model is None:
            debug.debug("Creating new target model", "ai")
            # Copy the architecture from the main model
            if hasattr(self.model, 'hidden_sizes'):
                self.target_model = DQNModel(hidden_sizes=self.model.hidden_sizes, input_channels=self.model.input_channels)
            else:
                self.target_model = DQNModel(hidden_size=hidden_size, layers=layers, input_channels=getattr(self.model, 'input_channels', 4))
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.target_model = target_model
        
        # Rest of the initialization remains the same...
        
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
    
    def get_action(self, board, valid_moves: List[int], current_player: Player, training: bool = False) -> int:
        """
        Choose an action using the current policy.
        
        Args:
            board: Current game board
            valid_moves: List of valid column indices
            current_player: Which player is to move for this decision
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
        state = board_to_state(board.grid, current_player)
        state_tensor = state_to_tensor(state)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)
        
        # Create mask for valid moves
        mask = torch.ones(COLS) * float('-inf')
        mask[valid_moves] = 0.0
        
        # Apply mask and get best action
        masked_q_values = q_values + mask
        action = int(torch.argmax(masked_q_values).item())
        
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
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)  # Clip by value instead of norm
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