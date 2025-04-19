"""
utils.py - Utility functions and constants for Connect Four implementation

This module provides common constants, enumerations, and helper functions
used throughout the Connect Four game implementation.
"""

from connect4.debug import debug, DebugLevel
from enum import Enum, auto
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Game constants
ROWS = 6
COLS = 7
CONNECT_N = 4  # Number of pieces in a row to win

class Player(Enum):
    """Enumeration representing players and cell states."""
    EMPTY = 0
    ONE = 1    # First player
    TWO = 2    # Second player
    
    def other(self):
        """Get the other player."""
        if self == Player.ONE:
            return Player.TWO
        elif self == Player.TWO:
            return Player.ONE
        return Player.EMPTY
    
    def __str__(self):
        if self == Player.EMPTY:
            return " "
        elif self == Player.ONE:
            return "X"
        else:
            return "O"


class GameResult(Enum):
    """Enumeration representing the game outcome."""
    IN_PROGRESS = auto()
    PLAYER_ONE_WIN = auto()
    PLAYER_TWO_WIN = auto()
    DRAW = auto()
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self != GameResult.IN_PROGRESS


class Direction(Enum):
    """Enumeration representing directions for win checking."""
    HORIZONTAL = auto()
    VERTICAL = auto()
    DIAGONAL_UP = auto()  # Diagonal from bottom-left to top-right
    DIAGONAL_DOWN = auto()  # Diagonal from top-left to bottom-right


# Direction vectors (row, col) for each direction
DIRECTION_VECTORS = {
    Direction.HORIZONTAL: (0, 1),
    Direction.VERTICAL: (1, 0),
    Direction.DIAGONAL_UP: (-1, 1),
    Direction.DIAGONAL_DOWN: (1, 1)
}


def get_column_height(board: np.ndarray, column: int) -> int:
    """
    Get the current height of a column (number of pieces).
    
    Args:
        board: The game board
        column: The column to check
    
    Returns:
        The number of pieces in the column
    """
    for row in range(ROWS):
        if board[row, column] == Player.EMPTY.value:
            return ROWS - row - 1
    return ROWS


def is_valid_position(row: int, col: int) -> bool:
    """
    Check if a position is within the board boundaries.
    
    Args:
        row: Row index
        col: Column index
    
    Returns:
        True if position is valid, False otherwise
    """
    return 0 <= row < ROWS and 0 <= col < COLS


def get_winning_positions(board: np.ndarray, player: Player) -> List[Tuple[int, int]]:
    """
    Find all positions that would result in a win for the given player.
    
    Args:
        board: The game board
        player: The player to check for
    
    Returns:
        List of (row, col) positions that would result in a win
    """
    winning_positions = []
    player_value = player.value
    
    # Check each column for a winning move
    for col in range(COLS):
        # Find the row where a piece would land
        row = -1
        for r in range(ROWS-1, -1, -1):
            if board[r, col] == Player.EMPTY.value:
                row = r
                break
        
        if row == -1:  # Column is full
            continue
        
        # Temporarily place the piece
        board[row, col] = player_value
        
        # Check if this move would win
        if check_win_at_position(board, row, col):
            winning_positions.append((row, col))
        
        # Remove the temporary piece
        board[row, col] = Player.EMPTY.value
    
    return winning_positions


def check_win_at_position(board: np.ndarray, row: int, col: int) -> bool:
    """
    Check if placing a piece at the given position would result in a win.
    
    Args:
        board: The game board
        row: Row index where piece was placed
        col: Column index where piece was placed
    
    Returns:
        True if the move results in a win, False otherwise
    """
    player_value = board[row, col]
    if player_value == Player.EMPTY.value:
        return False
    
    # Check all four directions
    for direction, (dr, dc) in DIRECTION_VECTORS.items():
        count = 1  # Start with 1 for the piece just placed
        
        # Check in the positive direction
        r, c = row + dr, col + dc
        while is_valid_position(r, c) and board[r, c] == player_value:
            count += 1
            r += dr
            c += dc
        
        # Check in the negative direction
        r, c = row - dr, col - dc
        while is_valid_position(r, c) and board[r, c] == player_value:
            count += 1
            r -= dr
            c -= dc
        
        if count >= CONNECT_N:
            return True
    
    return False


def render_board_ascii(board: np.ndarray) -> str:
    """
    Render the board as ASCII art.
    
    Args:
        board: The game board
    
    Returns:
        ASCII representation of the board
    """
    result = []
    result.append("|" + "-" * (COLS * 2 - 1) + "|")
    
    for row in range(ROWS):
        line = "|"
        for col in range(COLS):
            cell = board[row, col]
            if cell == Player.EMPTY.value:
                line += " "
            elif cell == Player.ONE.value:
                line += "X"
            else:
                line += "O"
            
            if col < COLS - 1:
                line += " "
        line += "|"
        result.append(line)
    
    result.append("|" + "-" * (COLS * 2 - 1) + "|")
    
    # Fix column numbering
    col_numbers = "|"
    for i in range(COLS):
        col_numbers += str(i)
        if i < COLS - 1:
            col_numbers += " "
    col_numbers += "|"
    
    result.append(col_numbers)
    
    return "\n".join(result)


if __name__ == "__main__":
    # Test the utility functions
    test_board = np.zeros((ROWS, COLS), dtype=int)
    print(render_board_ascii(test_board))
    
    # Place some pieces
    test_board[5, 3] = Player.ONE.value
    test_board[5, 4] = Player.TWO.value
    test_board[4, 3] = Player.ONE.value
    test_board[5, 5] = Player.ONE.value
    test_board[5, 6] = Player.ONE.value
    
    print("\nBoard with pieces:")
    print(render_board_ascii(test_board))
    
    # Test win detection
    print("\nChecking win at (5, 3):", check_win_at_position(test_board, 5, 3))
    
    # Test winning positions
    winning_moves = get_winning_positions(test_board, Player.ONE)
    print(f"\nWinning positions for Player.ONE: {winning_moves}")