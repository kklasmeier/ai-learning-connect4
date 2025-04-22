"""
replay_buffer.py - Experience replay buffer for Connect Four DQN

This module provides a replay buffer for storing experiences during 
reinforcement learning training. Experiences can be randomly sampled
for batch updates to the neural network.
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any

from connect4.debug import debug, DebugLevel

class ReplayBuffer:
    """
    Store and sample experiences for DQN training.
    
    The replay buffer stores transitions of the form (state, action, reward, next_state, done).
    These transitions are randomly sampled during training to break the correlation
    between consecutive training samples.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize a replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        debug.debug(f"Initializing ReplayBuffer with capacity {capacity}", "ai")
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        # Hard clip rewards to prevent extreme values
        reward = max(min(reward, 10.0), -10.0)
        
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Randomly sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        debug.trace(f"Sampling batch of size {batch_size}", "ai")
        
        # Limit batch size to current buffer size
        batch_size = min(batch_size, len(self.buffer))
        
        # Randomly sample from buffer
        batch = random.sample(self.buffer, batch_size)
        
        return batch
    
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        debug.debug("Clearing replay buffer", "ai")
        self.buffer.clear()