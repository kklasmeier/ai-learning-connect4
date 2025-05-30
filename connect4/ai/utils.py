"""
utils.py - Utility functions for AI reinforcement learning

This module provides helper functions for state representation, conversion,
and other utility functions for the Connect Four AI.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Any

from connect4.debug import debug, DebugLevel
from connect4.utils import ROWS, COLS, Player, GameResult, is_valid_position
from connect4.game.board import Board

def board_to_state(board: np.ndarray) -> np.ndarray:
    """
    Convert the raw board representation to a state tensor for the neural network.
    Uses one-hot encoding for the three possible states (empty, player 1, player 2).
    
    Args:
        board: Raw board grid (ROWS x COLS with values 0, 1, 2)
        
    Returns:
        3-channel state representation (ROWS x COLS x 3)
    """
    debug.trace("Converting board to state tensor", "ai")
    
    # Create a 3-channel representation (empty, player 1, player 2)
    # Channel 0: Empty positions (1 where empty, 0 elsewhere)
    # Channel 1: Player 1 positions (1 where player 1 pieces are, 0 elsewhere)
    # Channel 2: Player 2 positions (1 where player 2 pieces are, 0 elsewhere)
    state = np.zeros((3, ROWS, COLS), dtype=np.float32)
    
    # Fill the channels
    state[0] = (board == Player.EMPTY.value).astype(np.float32)
    state[1] = (board == Player.ONE.value).astype(np.float32)
    state[2] = (board == Player.TWO.value).astype(np.float32)
    
    return state

def state_to_tensor(state: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy state representation to a PyTorch tensor.
    
    Args:
        state: 3-channel state representation (3 x ROWS x COLS)
        
    Returns:
        PyTorch tensor ready for neural network input
    """
    debug.trace("Converting state to tensor", "ai")
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def get_valid_action_mask(board: Board) -> torch.Tensor:
    """
    Create a mask for valid actions.
    
    Args:
        board: The Connect Four board
        
    Returns:
        Binary mask tensor (1 for valid actions, 0 for invalid)
    """
    debug.trace("Creating valid action mask", "ai")
    valid_moves = board.get_valid_moves()
    mask = torch.zeros(COLS, dtype=torch.float32)
    mask[valid_moves] = 1.0
    return mask

def preprocess_batch(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Preprocess a batch of experiences for training.
    
    Args:
        batch: List of (state, action, reward, next_state, done) tuples
        
    Returns:
        Tuple of tensors for states, actions, rewards, next_states, and dones
    """
    debug.trace(f"Preprocessing batch of size {len(batch)}", "ai")
    
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert to tensors
    states_tensor = torch.cat([torch.tensor(s, dtype=torch.float32) for s in states], dim=0)
    actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states_tensor = torch.cat([torch.tensor(s, dtype=torch.float32) for s in next_states], dim=0)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    
    return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

def check_connections(grid, row, col, player_value):
    """
    Check for 2-in-a-row, 3-in-a-row connections at the given position.
    
    Args:
        grid: The game board grid
        row, col: Position of the last move
        player_value: Value of the player (1 or 2)
        
    Returns:
        Dictionary with counts of connections
    """
    directions = [
        (0, 1),  # horizontal
        (1, 0),  # vertical
        (1, 1),  # diagonal down
        (-1, 1)  # diagonal up
    ]
    
    connections = {"two": 0, "three": 0, "blocks_two": 0, "blocks_three": 0}
    opponent_value = 3 - player_value  # Convert between 1 and 2
    
    for dr, dc in directions:
        # Count consecutive pieces in this direction for both player and opponent
        for player_type, check_value in [("player", player_value), ("opponent", opponent_value)]:
            consecutive = 1  # Start with the piece just placed
            
            # Check in the positive direction
            r, c = row + dr, col + dc
            while is_valid_position(r, c) and grid[r, c] == check_value:
                consecutive += 1
                r += dr
                c += dc
            
            # Check in the negative direction
            r, c = row - dr, col - dc
            while is_valid_position(r, c) and grid[r, c] == check_value:
                consecutive += 1
                r -= dr
                c -= dc
            
            # Record the findings
            if player_type == "player":
                if consecutive == 2:
                    connections["two"] += 1
                elif consecutive == 3:
                    connections["three"] += 1
            else:  # opponent
                # If we blocked opponent's potential connection
                # Check if adding one more piece would make a 3 or 4 in a row
                if consecutive == 2:
                    connections["blocks_two"] += 1
                elif consecutive == 3:
                    connections["blocks_three"] += 1
    
    return connections