"""
minimax.py - Minimax algorithm with alpha-beta pruning for Connect Four

This module provides a MinimaxPlayer class that can play Connect Four
optimally up to a configurable search depth.
"""

import math
from typing import Tuple, Optional

from connect4.utils import ROWS, COLS, CONNECT_N, Player, GameResult
from connect4.game.board import Board


class MinimaxPlayer:
    """
    A Connect Four player that uses the minimax algorithm with alpha-beta pruning.
    
    This player evaluates positions by searching the game tree up to a specified
    depth, assuming both players play optimally.
    """
    
    def __init__(self, depth: int = 6):
        """
        Initialize the minimax player.
        
        Args:
            depth: Maximum search depth (higher = stronger but slower)
        """
        self.depth = depth
        self.nodes_evaluated = 0  # For performance tracking
    
    def get_move(self, board: Board) -> int:
        """
        Get the best move for the current player.
        
        Args:
            board: The current game board
            
        Returns:
            The column index of the best move
        """
        self.nodes_evaluated = 0
        
        # Get the player we're finding a move for
        maximizing_player = board.current_player
        
        best_score = -math.inf
        best_column = None
        alpha = -math.inf
        beta = math.inf
        
        # Try each valid move
        for column in board.get_valid_moves():
            # Make the move - track if game ended (affects undo behavior)
            game_ended_before = board.game_result.is_game_over()
            board.make_move(column)
            game_ended_after = board.game_result.is_game_over()
            
            # Evaluate this move (next level is minimizing)
            score = self._minimax(board, self.depth - 1, alpha, beta, False, maximizing_player)
            
            # Undo the move - fix player tracking if game ended
            board.undo_move()
            # If the move ended the game, undo_move switched player but make_move didn't
            # So we need to switch back to correct this
            if game_ended_after and not game_ended_before:
                board.current_player = board.current_player.other()
            
            # Update best move
            if score > best_score:
                best_score = score
                best_column = column
            
            # Update alpha
            alpha = max(alpha, score)
        
        return best_column
    
    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, 
                 is_maximizing: bool, maximizing_player: Player) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning (best score maximizer can guarantee)
            beta: Beta value for pruning (best score minimizer can guarantee)
            is_maximizing: True if this is a maximizing node
            maximizing_player: The player we're trying to maximize score for
            
        Returns:
            The evaluation score for this position
        """
        self.nodes_evaluated += 1
        
        # Terminal conditions
        if board.game_result == GameResult.PLAYER_ONE_WIN:
            if maximizing_player == Player.ONE:
                return 1000 + depth  # Prefer faster wins
            else:
                return -1000 - depth
        
        if board.game_result == GameResult.PLAYER_TWO_WIN:
            if maximizing_player == Player.TWO:
                return 1000 + depth
            else:
                return -1000 - depth
        
        if board.game_result == GameResult.DRAW:
            return 0
        
        # Depth limit reached - use heuristic evaluation
        if depth == 0:
            return self._evaluate_position(board, maximizing_player)
        
        valid_moves = board.get_valid_moves()
        
        if is_maximizing:
            max_score = -math.inf
            
            for column in valid_moves:
                game_ended_before = board.game_result.is_game_over()
                board.make_move(column)
                game_ended_after = board.game_result.is_game_over()
                
                score = self._minimax(board, depth - 1, alpha, beta, False, maximizing_player)
                
                board.undo_move()
                if game_ended_after and not game_ended_before:
                    board.current_player = board.current_player.other()
                
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                # Beta cutoff
                if beta <= alpha:
                    break
            
            return max_score
        
        else:  # Minimizing
            min_score = math.inf
            
            for column in valid_moves:
                game_ended_before = board.game_result.is_game_over()
                board.make_move(column)
                game_ended_after = board.game_result.is_game_over()
                
                score = self._minimax(board, depth - 1, alpha, beta, True, maximizing_player)
                
                board.undo_move()
                if game_ended_after and not game_ended_before:
                    board.current_player = board.current_player.other()
                
                min_score = min(min_score, score)
                beta = min(beta, score)
                
                # Alpha cutoff
                if beta <= alpha:
                    break
            
            return min_score
    
    def _evaluate_position(self, board: Board, maximizing_player: Player) -> float:
        """
        Heuristic evaluation of a board position.
        
        This is used when we reach the depth limit without finding a terminal state.
        The evaluation considers:
        - Center column control (more valuable positions)
        - Potential winning lines (sequences of 2 and 3 in a row with room to grow)
        
        Args:
            board: The board to evaluate
            maximizing_player: The player we're evaluating for
            
        Returns:
            A score representing how good the position is for the maximizing player
        """
        score = 0.0
        grid = board.grid
        
        opponent = maximizing_player.other()
        my_value = maximizing_player.value
        opp_value = opponent.value
        
        # Center column preference (pieces in center are more valuable)
        center_col = COLS // 2
        center_count = sum(1 for row in range(ROWS) if grid[row, center_col] == my_value)
        opp_center_count = sum(1 for row in range(ROWS) if grid[row, center_col] == opp_value)
        score += (center_count - opp_center_count) * 3
        
        # Evaluate all windows of 4
        score += self._evaluate_windows(grid, my_value, opp_value)
        
        return score
    
    def _evaluate_windows(self, grid, my_value: int, opp_value: int) -> float:
        """
        Evaluate all possible 4-in-a-row windows on the board.
        
        Args:
            grid: The board grid
            my_value: The value of our pieces
            opp_value: The value of opponent's pieces
            
        Returns:
            Score based on window evaluation
        """
        score = 0.0
        empty_value = Player.EMPTY.value
        
        # Horizontal windows
        for row in range(ROWS):
            for col in range(COLS - 3):
                window = [grid[row, col + i] for i in range(4)]
                score += self._score_window(window, my_value, opp_value, empty_value)
        
        # Vertical windows
        for row in range(ROWS - 3):
            for col in range(COLS):
                window = [grid[row + i, col] for i in range(4)]
                score += self._score_window(window, my_value, opp_value, empty_value)
        
        # Diagonal (down-right) windows
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                window = [grid[row + i, col + i] for i in range(4)]
                score += self._score_window(window, my_value, opp_value, empty_value)
        
        # Diagonal (up-right) windows
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                window = [grid[row - i, col + i] for i in range(4)]
                score += self._score_window(window, my_value, opp_value, empty_value)
        
        return score
    
    def _score_window(self, window: list, my_value: int, opp_value: int, empty_value: int) -> float:
        """
        Score a single window of 4 cells.
        
        Args:
            window: List of 4 cell values
            my_value: Our piece value
            opp_value: Opponent's piece value
            empty_value: Empty cell value
            
        Returns:
            Score for this window
        """
        my_count = window.count(my_value)
        opp_count = window.count(opp_value)
        empty_count = window.count(empty_value)
        
        # If window has both players' pieces, it's useless
        if my_count > 0 and opp_count > 0:
            return 0
        
        # Score based on our pieces
        if my_count == 3 and empty_count == 1:
            return 5  # One away from winning
        elif my_count == 2 and empty_count == 2:
            return 2  # Building potential
        
        # Penalize opponent's progress
        if opp_count == 3 and empty_count == 1:
            return -4  # Block this!
        elif opp_count == 2 and empty_count == 2:
            return -1
        
        return 0


if __name__ == "__main__":
    # Test the minimax player
    from connect4.debug import debug, DebugLevel
    import time
    
    debug.configure(level=DebugLevel.INFO)
    
    print("Testing MinimaxPlayer")
    print("=" * 40)
    
    # Create a board and player
    board = Board()
    player = MinimaxPlayer(depth=6)
    
    # Test getting a move from empty board
    print("\nEmpty board:")
    print(board)
    
    start = time.time()
    move = player.get_move(board)
    elapsed = time.time() - start
    
    print(f"Best move: column {move}")
    print(f"Nodes evaluated: {player.nodes_evaluated}")
    print(f"Time: {elapsed:.3f} seconds")
    
    # Play a few moves and test again
    print("\n" + "=" * 40)
    print("After some moves:")
    
    board.make_move(3)  # Player 1 center
    board.make_move(2)  # Player 2
    board.make_move(3)  # Player 1 stacks
    board.make_move(4)  # Player 2
    board.make_move(3)  # Player 1 stacks again
    
    print(board)
    print(f"Current player: {board.current_player}")
    
    start = time.time()
    move = player.get_move(board)
    elapsed = time.time() - start
    
    print(f"Best move: column {move}")
    print(f"Nodes evaluated: {player.nodes_evaluated}")
    print(f"Time: {elapsed:.3f} seconds")
    
    # Test a position where blocking is critical
    print("\n" + "=" * 40)
    print("Critical blocking test:")
    
    board = Board()
    # Set up: Player 1 has 3 in a row horizontally
    board.make_move(0)  # P1
    board.make_move(0)  # P2
    board.make_move(1)  # P1
    board.make_move(1)  # P2
    board.make_move(2)  # P1
    # Now P2 must block at column 3
    
    print(board)
    print(f"Current player: {board.current_player}")
    
    start = time.time()
    move = player.get_move(board)
    elapsed = time.time() - start
    
    print(f"Best move: column {move} (should be 3 to block)")
    print(f"Nodes evaluated: {player.nodes_evaluated}")
    print(f"Time: {elapsed:.3f} seconds")