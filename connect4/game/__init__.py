"""
connect4.game - Core game mechanics for Connect Four

This package contains the core game logic, board representation,
and game state management for the Connect Four implementation.
"""

from connect4.game.board import Board
from connect4.game.rules import ConnectFourGame, ConnectFourEnv

__all__ = ['Board', 'ConnectFourGame', 'ConnectFourEnv']