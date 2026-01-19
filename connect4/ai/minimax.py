"""
minimax.py - Minimax algorithm with alpha-beta pruning for Connect Four

This module provides a MinimaxPlayer class that can play Connect Four
optimally up to a configurable search depth.

The heuristic evaluation is designed to:
1. Balance offense and defense (slightly defensive bias)
2. Only count threats that are actually playable
3. Prefer horizontal/diagonal threats over vertical (more dangerous)
4. Detect and reward fork positions (two threats at once)
"""

import math
from typing import Tuple, Optional, List

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
        best_column = board.get_valid_moves()[0]
        alpha = -math.inf
        beta = math.inf
        
        # Try each valid move - prefer center columns for tie-breaking
        valid_moves = board.get_valid_moves()
        # Sort moves by distance from center (center first) for better pruning
        center = COLS // 2
        valid_moves = sorted(valid_moves, key=lambda c: abs(c - center))
        
        for column in valid_moves:
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
        # Sort moves by distance from center for better pruning
        center = COLS // 2
        valid_moves = sorted(valid_moves, key=lambda c: abs(c - center))
        
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
    
    def _is_playable(self, grid, row: int, col: int) -> bool:
        """
        Check if a cell is playable (empty and either on bottom row or has support).
        
        Args:
            grid: The board grid
            row: Row index
            col: Column index
            
        Returns:
            True if the cell can be played immediately
        """
        if col < 0 or col >= COLS or row < 0 or row >= ROWS:
            return False
        if grid[row, col] != Player.EMPTY.value:
            return False
        # Bottom row is always playable if empty
        if row == ROWS - 1:
            return True
        # Otherwise needs support below
        return grid[row + 1, col] != Player.EMPTY.value
    
    def _evaluate_position(self, board: Board, maximizing_player: Player) -> float:
        """
        Heuristic evaluation of a board position.
        
        This is used when we reach the depth limit without finding a terminal state.
        The evaluation considers:
        - Center column control (more valuable positions)
        - Playable threats (3 in a row where the 4th can be played NOW)
        - Potential threats (3 in a row where 4th exists but isn't playable yet)
        - Building potential (2 in a row with room to grow)
        - Fork detection (multiple threats)
        
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
        empty_value = Player.EMPTY.value
        
        # Center column preference (pieces in center are more valuable)
        # Center and near-center columns give more winning opportunities
        for col in range(COLS):
            col_weight = (COLS // 2) - abs(col - COLS // 2)  # 3,2,1,0,1,2,3 for 7 cols -> 0,1,2,3,2,1,0
            for row in range(ROWS):
                if grid[row, col] == my_value:
                    score += col_weight * 0.5
                elif grid[row, col] == opp_value:
                    score -= col_weight * 0.5
        
        # Count threats for both players
        my_playable_threats = 0
        my_potential_threats = 0
        opp_playable_threats = 0
        opp_potential_threats = 0
        
        # Evaluate all windows and track threats
        my_threat_positions = []  # List of (row, col) where we have playable winning moves
        opp_threat_positions = []
        
        # Check all possible 4-in-a-row windows
        window_results = self._evaluate_all_windows(grid, my_value, opp_value, empty_value)
        
        score += window_results['score']
        my_playable_threats = window_results['my_playable_threats']
        my_potential_threats = window_results['my_potential_threats']
        opp_playable_threats = window_results['opp_playable_threats']
        opp_potential_threats = window_results['opp_potential_threats']
        my_threat_positions = window_results['my_threat_positions']
        opp_threat_positions = window_results['opp_threat_positions']
        
        # Playable threats are very valuable (can win/lose immediately)
        # Use SYMMETRIC scoring - defense is just as important as offense
        score += my_playable_threats * 50
        score -= opp_playable_threats * 50  # Same weight for blocking!
        
        # Potential threats (not immediately playable)
        score += my_potential_threats * 8
        score -= opp_potential_threats * 8
        
        # Fork bonus: having multiple playable threats is extremely strong
        # (opponent can only block one)
        if my_playable_threats >= 2:
            score += 200  # Almost guaranteed win
        if opp_playable_threats >= 2:
            score -= 200  # Almost guaranteed loss
        
        # Multiple potential threats also valuable (creates fork opportunities)
        unique_my_threats = len(set(my_threat_positions))
        unique_opp_threats = len(set(opp_threat_positions))
        if unique_my_threats >= 2:
            score += unique_my_threats * 5
        if unique_opp_threats >= 2:
            score -= unique_opp_threats * 5
        
        return score
    
    def _evaluate_all_windows(self, grid, my_value: int, opp_value: int, empty_value: int) -> dict:
        """
        Evaluate all possible 4-in-a-row windows on the board.
        
        Returns a dictionary with:
        - score: Base score from window evaluation
        - my_playable_threats: Count of our immediately playable winning moves
        - my_potential_threats: Count of our non-immediate threats
        - opp_playable_threats: Count of opponent's immediately playable winning moves
        - opp_potential_threats: Count of opponent's non-immediate threats
        - my_threat_positions: List of positions where we can win
        - opp_threat_positions: List of positions where opponent can win
        """
        result = {
            'score': 0.0,
            'my_playable_threats': 0,
            'my_potential_threats': 0,
            'opp_playable_threats': 0,
            'opp_potential_threats': 0,
            'my_threat_positions': [],
            'opp_threat_positions': [],
        }
        
        # Direction vectors: (row_delta, col_delta, threat_multiplier)
        # Horizontal and diagonal threats are more dangerous than vertical
        directions = [
            (0, 1, 1.2),   # Horizontal - slightly more valuable
            (1, 0, 0.8),   # Vertical - less valuable (easier to see/block)
            (1, 1, 1.2),   # Diagonal down-right - more valuable
            (-1, 1, 1.2),  # Diagonal up-right - more valuable
        ]
        
        for row in range(ROWS):
            for col in range(COLS):
                for dr, dc, multiplier in directions:
                    # Check if window fits on board
                    end_row = row + 3 * dr
                    end_col = col + 3 * dc
                    if not (0 <= end_row < ROWS and 0 <= end_col < COLS):
                        continue
                    if dr == -1 and row < 3:  # For up-right diagonal
                        continue
                    
                    # Extract window
                    window = []
                    positions = []
                    for i in range(4):
                        r, c = row + i * dr, col + i * dc
                        window.append(grid[r, c])
                        positions.append((r, c))
                    
                    # Analyze window
                    window_result = self._analyze_window(
                        grid, window, positions, my_value, opp_value, empty_value, multiplier
                    )
                    
                    result['score'] += window_result['score']
                    result['my_playable_threats'] += window_result['my_playable_threat']
                    result['my_potential_threats'] += window_result['my_potential_threat']
                    result['opp_playable_threats'] += window_result['opp_playable_threat']
                    result['opp_potential_threats'] += window_result['opp_potential_threat']
                    if window_result['my_threat_pos']:
                        result['my_threat_positions'].append(window_result['my_threat_pos'])
                    if window_result['opp_threat_pos']:
                        result['opp_threat_positions'].append(window_result['opp_threat_pos'])
        
        return result
    
    def _analyze_window(self, grid, window: list, positions: list, 
                        my_value: int, opp_value: int, empty_value: int,
                        multiplier: float) -> dict:
        """
        Analyze a single window of 4 cells.
        
        Args:
            grid: The board grid (for playability checking)
            window: List of 4 cell values
            positions: List of 4 (row, col) positions
            my_value: Our piece value
            opp_value: Opponent's piece value
            empty_value: Empty cell value
            multiplier: Direction-based score multiplier
            
        Returns:
            Dictionary with score and threat information
        """
        result = {
            'score': 0.0,
            'my_playable_threat': 0,
            'my_potential_threat': 0,
            'opp_playable_threat': 0,
            'opp_potential_threat': 0,
            'my_threat_pos': None,
            'opp_threat_pos': None,
        }
        
        my_count = window.count(my_value)
        opp_count = window.count(opp_value)
        empty_count = window.count(empty_value)
        
        # If window has both players' pieces, it's blocked - no value
        if my_count > 0 and opp_count > 0:
            return result
        
        # Find empty positions in this window
        empty_positions = [positions[i] for i in range(4) if window[i] == empty_value]
        
        # Evaluate for our pieces
        if my_count == 3 and empty_count == 1:
            # One away from winning!
            empty_pos = empty_positions[0]
            if self._is_playable(grid, empty_pos[0], empty_pos[1]):
                result['my_playable_threat'] = 1
                result['my_threat_pos'] = empty_pos
            else:
                result['my_potential_threat'] = 1
                result['my_threat_pos'] = empty_pos
            result['score'] += 5 * multiplier
            
        elif my_count == 2 and empty_count == 2:
            # Building potential - check if at least one empty is playable
            playable_empties = sum(1 for pos in empty_positions 
                                   if self._is_playable(grid, pos[0], pos[1]))
            if playable_empties > 0:
                result['score'] += 3 * multiplier
            else:
                result['score'] += 1 * multiplier
                
        elif my_count == 1 and empty_count == 3:
            # Early positioning
            result['score'] += 0.5 * multiplier
        
        # Evaluate for opponent's pieces (SYMMETRIC scoring for defense)
        if opp_count == 3 and empty_count == 1:
            # Opponent one away from winning - must block!
            empty_pos = empty_positions[0]
            if self._is_playable(grid, empty_pos[0], empty_pos[1]):
                result['opp_playable_threat'] = 1
                result['opp_threat_pos'] = empty_pos
            else:
                result['opp_potential_threat'] = 1
                result['opp_threat_pos'] = empty_pos
            result['score'] -= 5 * multiplier  # Same magnitude as offense!
            
        elif opp_count == 2 and empty_count == 2:
            playable_empties = sum(1 for pos in empty_positions 
                                   if self._is_playable(grid, pos[0], pos[1]))
            if playable_empties > 0:
                result['score'] -= 3 * multiplier
            else:
                result['score'] -= 1 * multiplier
                
        elif opp_count == 1 and empty_count == 3:
            result['score'] -= 0.5 * multiplier
        
        return result


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
    
    # Test fork detection
    print("\n" + "=" * 40)
    print("Fork detection test:")
    
    board = Board()
    # Set up a position where player can create a fork
    board.make_move(3)  # P1 center
    board.make_move(0)  # P2 corner
    board.make_move(2)  # P1
    board.make_move(0)  # P2
    board.make_move(4)  # P1 - now has 2-3-4, threatening both 1 and 5
    
    print(board)
    print(f"Current player: {board.current_player}")
    
    start = time.time()
    move = player.get_move(board)
    elapsed = time.time() - start
    
    print(f"Best move: column {move}")
    print(f"Nodes evaluated: {player.nodes_evaluated}")
    print(f"Time: {elapsed:.3f} seconds")
    
    # Test vertical stacking isn't overvalued
    print("\n" + "=" * 40)
    print("Vertical vs Horizontal test:")
    print("(Should prefer horizontal threats over vertical)")
    
    board = Board()
    board.make_move(3)  # P1 center
    board.make_move(6)  # P2 far right
    board.make_move(3)  # P1 stacks
    board.make_move(6)  # P2 stacks
    # P1 should prefer building horizontal (2 or 4) over more vertical (3)
    
    print(board)
    print(f"Current player: {board.current_player}")
    
    start = time.time()
    move = player.get_move(board)
    elapsed = time.time() - start
    
    print(f"Best move: column {move} (should prefer 2 or 4 over 3)")
    print(f"Nodes evaluated: {player.nodes_evaluated}")
    print(f"Time: {elapsed:.3f} seconds")