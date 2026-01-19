"""
connect4/ai/__init__.py - AI module for Connect Four reinforcement learning

This module provides reinforcement learning components that enable
the Connect Four game to learn through various training approaches.
"""

__all__ = [
    'DQNAgent', 
    'DQNModel', 
    'ReplayBuffer', 
    'MinimaxPlayer',
    'Trainer',
    'OpponentType',
    'train_against_minimax',
    'train_self_play'
]

from connect4.ai.dqn import DQNAgent, DQNModel
from connect4.ai.replay_buffer import ReplayBuffer
from connect4.ai.minimax import MinimaxPlayer
from connect4.ai.trainer import Trainer, OpponentType, train_against_minimax, train_self_play