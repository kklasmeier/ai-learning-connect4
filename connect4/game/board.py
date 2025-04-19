"""
board.py - Board representation and core game mechanics for Connect Four

This module implements the Board class which represents the Connect Four game board
and provides methods for making moves, checking win conditions, and managing game state.
"""

import numpy as np
from typing import List, Tuple, Optional, Set

from connect4.debug import debug, DebugLevel
from connect4.utils import (ROWS, COLS, CONNECT_N, Player, GameResult, Direction,
                          check_win_at_position, is_valid_position,
                          get_column_height, render_board_ascii)


class Board:
    """
    Represents a Connect Four game board.
    
    This class manages the board state, validates and executes moves,
    and checks for win conditions.
    """
    
    def __init__(self):
        """Initialize an empty Connect Four board."""
        debug.debug("Initializing new Board", "board")
        self.reset()
    
    def reset(self):
        """Reset the board to an empty state."""
        debug.debug("Resetting board", "board")
        self.grid = np.zeros((ROWS, COLS), dtype=int)
        self.moves_made = []
        self.current_player = Player.ONE
        self.game_result = GameResult.IN_PROGRESS
        self.last_move = None
    
    def copy(self) -> 'Board':
        """
        Create a deep copy of the current board.
        
        Returns:
            A new Board instance with the same state
        """
        debug.trace("Creating board copy", "board")
        new_board = Board()
        new_board.grid = self.grid.copy()
        new_board.moves_made = self.moves_made.copy()
        new_board.current_player = self.current_player
        new_board.game_result = self.game_result
        new_board.last_move = self.last_move
        return new_board
    
    def is_valid_move(self, column: int) -> bool:
        """
        Check if a move is valid.
        
        Args:
            column: The column to place a piece (0-indexed)
            
        Returns:
            True if the move is valid, False otherwise
        """
        # Check if game is already over
        if self.game_result.is_game_over():
            debug.debug(f"Invalid move: game is over (result: {self.game_result})", "board")
            return False
        
        # Check if column is within bounds
        if not (0 <= column < COLS):
            debug.debug(f"Invalid move: column {column} out of bounds", "board")
            return False
        
        # Check if column is full
        if self.grid[0, column] != Player.EMPTY.value:
            debug.debug(f"Invalid move: column {column} is full", "board")
            return False
        
        return True
    
    def get_valid_moves(self) -> List[int]:
        """
        Get a list of valid columns where a piece can be placed.
        
        Returns:
            List of valid column indices
        """
        if self.game_result.is_game_over():
            return []
        
        return [col for col in range(COLS) if self.is_valid_move(col)]
    
    def make_move(self, column: int) -> bool:
        """
        Place a piece in the specified column.
        
        Args:
            column: The column to place a piece (0-indexed)
            
        Returns:
            True if the move was successful, False otherwise
        """
        debug.debug(f"Attempting move in column {column} for player {self.current_player}", "board")
        
        if not self.is_valid_move(column):
            return False
        
        # Find the lowest empty row in the column
        for row in range(ROWS-1, -1, -1):
            if self.grid[row, column] == Player.EMPTY.value:
                debug.trace(f"Placing piece at position ({row}, {column})", "board")
                self.grid[row, column] = self.current_player.value
                self.last_move = (row, column)
                self.moves_made.append(column)
                break
        
        # Check for win condition
        debug.start_timer("win_check")
        if self._check_win():
            if self.current_player == Player.ONE:
                self.game_result = GameResult.PLAYER_ONE_WIN
                debug.info(f"Player ONE wins after move at {self.last_move}", "board")
            else:
                self.game_result = GameResult.PLAYER_TWO_WIN
                debug.info(f"Player TWO wins after move at {self.last_move}", "board")
        elif len(self.moves_made) == ROWS * COLS:
            # Board is full with no winner
            self.game_result = GameResult.DRAW
            debug.info("Game ends in a draw", "board")
        debug.end_timer("win_check", "board")
        
        # Switch player if game is still in progress
        if not self.game_result.is_game_over():
            self.current_player = self.current_player.other()
            debug.debug(f"Switching to player {self.current_player}", "board")
        
        return True
    
    def undo_move(self) -> bool:
        """
        Undo the last move.
        
        Returns:
            True if a move was undone, False if no moves to undo
        """
        if not self.moves_made:
            debug.debug("No moves to undo", "board")
            return False
        
        # Get the last move
        last_column = self.moves_made.pop()
        
        # Find the top piece in that column and remove it
        for row in range(ROWS):
            if self.grid[row, last_column] != Player.EMPTY.value:
                debug.debug(f"Undoing move at ({row}, {last_column})", "board")
                self.grid[row, last_column] = Player.EMPTY.value
                break
        
        # Reset game result to in progress
        self.game_result = GameResult.IN_PROGRESS
        
        # Switch back to previous player
        self.current_player = self.current_player.other()
        
        # Update last move if there are any moves left
        if self.moves_made:
            last_column = self.moves_made[-1]
            for row in range(ROWS):
                if self.grid[row, last_column] != Player.EMPTY.value:
                    self.last_move = (row, last_column)
                    break
        else:
            self.last_move = None
        
        return True
    
    def _check_win(self) -> bool:
        """
        Check if the last move resulted in a win.
        
        Returns:
            True if there is a win, False otherwise
        """
        if self.last_move is None:
            return False
        
        row, col = self.last_move
        return check_win_at_position(self.grid, row, col)
    
    def get_winning_line(self) -> List[Tuple[int, int]]:
        """
        Get the positions of the winning line if the game is won.
        
        Returns:
            List of (row, col) positions forming the winning line, or empty list if no win
        """
        if not self.game_result.is_game_over() or self.game_result == GameResult.DRAW:
            return []
        
        if self.last_move is None:
            return []
        
        row, col = self.last_move
        player_value = self.grid[row, col]
        
        # Check all four directions
        for direction, (dr, dc) in {
            Direction.HORIZONTAL: (0, 1),
            Direction.VERTICAL: (1, 0),
            Direction.DIAGONAL_UP: (-1, 1),
            Direction.DIAGONAL_DOWN: (1, 1)
        }.items():
            # Count continuous pieces in this direction
            positions = [(row, col)]  # Start with the last move
            
            # Check in the positive direction
            r, c = row + dr, col + dc
            while is_valid_position(r, c) and self.grid[r, c] == player_value:
                positions.append((r, c))
                r += dr
                c += dc
            
            # Check in the negative direction
            r, c = row - dr, col - dc
            while is_valid_position(r, c) and self.grid[r, c] == player_value:
                positions.append((r, c))
                r -= dr
                c -= dc
            
            if len(positions) >= CONNECT_N:
                return positions
        
        return []
    
    def get_state(self) -> np.ndarray:
        """
        Get the current board state as a numpy array.
        
        Returns:
            2D numpy array representing the board
        """
        return self.grid.copy()
    
    def render(self) -> str:
        """
        Render the board as a string.
        
        Returns:
            String representation of the board
        """
        return render_board_ascii(self.grid)
    
    def __str__(self) -> str:
        """String representation of the board."""
        return self.render()


if __name__ == "__main__":
    # Test the Board class
    debug.configure(level=DebugLevel.DEBUG)
    
    board = Board()
    print("Initial board:")
    print(board)
    
    # Make some moves
    for col in [3, 2, 4, 2, 5, 2, 6]:
        print(f"\nMaking move in column {col}")
        if board.make_move(col):
            print(board)
            print(f"Game result: {board.game_result}")
        else:
            print(f"Invalid move: {col}")
    
    # Test the winning line detection
    if board.game_result.is_game_over() and board.game_result != GameResult.DRAW:
        winning_line = board.get_winning_line()
        print(f"\nWinning line: {winning_line}")
    
    # Test undo
    print("\nUndoing last move:")
    board.undo_move()
    print(board)
    
    # Test making an invalid move after game is over
    board.make_move(3)
    board.make_move(3)
    board.make_move(3)
    board.make_move(3)  # This should result in a win
    print("\nAfter making more moves:")
    print(board)
    print(f"Game result: {board.game_result}")
    
    # Attempt invalid move
    print("\nAttempting move after game is over:")
    result = board.make_move(0)
    print(f"Move successful: {result}")