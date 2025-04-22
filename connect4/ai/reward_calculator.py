import numpy as np
from typing import List, Tuple, Optional

from connect4.utils import ROWS, COLS, Player, is_valid_position
from connect4.game.rules import ConnectFourGame
from connect4.debug import debug, DebugLevel

class ConnectFourRewardCalculator:
    """Calculates rewards for Connect Four game states."""
    
    def __init__(self):
        # Core game outcome rewards
        self.reward_win = 20.0  # Increased from 1.0
        self.reward_draw = 0.1
        self.reward_loss = -1.0
        
        # Offensive rewards
        self.reward_two_in_row_open_ends = 0.15
        self.reward_three_in_row_open_end = 0.65
        self.reward_fork = 0.35
        self.penalty_missed_win = -0.5
        
        # Defensive rewards
        self.reward_block_three = 0.2
        self.reward_block_two_open = 0.1
        
        # Anti-stacking penalties
        self.penalty_stack_one = -2.0      # First stack in same column (was -0.4)
        self.penalty_stack_two = -5.0      # Second consecutive stack (was -0.6)
        self.penalty_stack_three_plus = -10.0  # Third+ consecutive stack (was -0.8)
        
        # Positional rewards
        self.reward_center_column = 0.05
        self.reward_adjacent_center = 0.02
        self.reward_supported_piece = 0.03
        
        # Early game rewards
        self.reward_different_quadrants = 0.04
        self.reward_bottom_rows = 0.03
        
        # Direction multipliers (more extreme difference)
        self.multiplier_horizontal = 3.0    # Up from 1.5
        self.multiplier_diagonal = 2.5      # Up from 1.2
        self.multiplier_vertical = 0.3      # Down from 0.7
        
        # Base move reward (small negative)
        self.reward_base_move = -0.01
    
    def calculate_reward(self, game, last_row, last_col, last_column, action, 
                         current_player, stacking_count, valid_moves):
        """
        Calculate the reward for the current game state after a move.
        
        Args:
            game: The Connect Four game instance
            last_row, last_col: Position of the last move
            last_column: Previous column played
            action: Current column played
            current_player: Current player (Player.ONE or Player.TWO)
            stacking_count: Counter for consecutive plays in same column
            valid_moves: List of valid move columns
            
        Returns:
            float: The calculated reward
        """
        reward = self.reward_base_move
        
        # Convert player enum to grid value
        player_value = 1 if current_player == Player.ONE else 2
        opponent_value = 3 - player_value
        
        # Check if game is over
        if game.is_game_over():
            winner = game.get_winner()
            if winner == Player.ONE:
                return self.reward_win  # Player 1 wins
            elif winner == Player.TWO:
                return self.reward_loss  # Player 2 wins
            else:
                return self.reward_draw  # Draw
        
#        # Calculate anti-stacking penalty
#        if last_column == action:
#            if stacking_count == 1:
#                reward += self.penalty_stack_one
#            elif stacking_count == 2:
#                reward += self.penalty_stack_two
#            else:
#                reward += self.penalty_stack_three_plus

# Add after your existing stacking penalty calculation, before you check for connections

        # Check for problematic stacking
        alternating_stack_penalty = self._check_alternating_stack(
            game.board.grid, action, last_row, player_value
        )
        reward += alternating_stack_penalty

        # If we found a problematic stack, log it for debugging
        if alternating_stack_penalty < 0:
            debug.warning(f"Problematic stacking detected in column {action}! Applied penalty: {alternating_stack_penalty}", "training")
            

        # Positional rewards
        if last_col == 3:  # Center column
            reward += self.reward_center_column
        elif last_col in [2, 4]:  # Adjacent to center
            reward += self.reward_adjacent_center
        
        # Check for piece supported from below
        if last_row < ROWS - 1 and game.board.grid[last_row + 1, last_col] == player_value:
            reward += self.reward_supported_piece
        
        # Early game rewards
        if len(game.board.moves_made) <= 4:
            # Check if playing in different quadrants (simplified)
            if game.board.moves_made.count(action) == 0:
                reward += self.reward_different_quadrants
            
            # Reward for playing in bottom rows
            if last_row >= ROWS - 2:
                reward += self.reward_bottom_rows
        
        # Direction-based checks for connections and blocks
        directions = [
            ((0, 1), self.multiplier_horizontal),   # horizontal
            ((1, 0), self.multiplier_vertical),     # vertical
            ((1, 1), self.multiplier_diagonal),     # diagonal down
            ((-1, 1), self.multiplier_diagonal)     # diagonal up
        ]
        
        # Track if we found any potential winning moves that were missed
        missed_win_detected = False
        
        for (dr, dc), direction_multiplier in directions:
            # Analyze connections in this direction
            consecutive, open_ends = self._count_consecutive(
                game.board.grid, last_row, last_col, dr, dc, player_value
            )
            
            # Apply direction-specific rewards
            if consecutive == 2 and open_ends == 2:
                reward += self.reward_two_in_row_open_ends * direction_multiplier
            elif consecutive == 3 and open_ends >= 1:
                reward += self.reward_three_in_row_open_end * direction_multiplier
            
            # Check for missed win
            if consecutive == 3 and open_ends >= 1:
                # Check if there was a winning move available but not taken
                if not self._check_if_winning_move_taken(
                    game.board.grid, last_row, last_col, dr, dc, 
                    player_value, valid_moves
                ):
                    missed_win_detected = True
            
            # Check if we blocked opponent
            opponent_consecutive, opponent_open_ends = self._check_blocking(
                game.board.grid, last_row, last_col, dr, dc, opponent_value
            )
            
            if opponent_consecutive == 3:
                reward += self.reward_block_three * direction_multiplier
            elif opponent_consecutive == 2 and opponent_open_ends == 2:
                reward += self.reward_block_two_open * direction_multiplier
        
        # Apply fork detection (simplified)
        threats = self._count_threats(game.board.grid, player_value)
        if threats >= 2:
            reward += self.reward_fork
        
        # Apply missed win penalty if detected
        if missed_win_detected:
            reward += self.penalty_missed_win
        
        return reward
    
    def _count_consecutive(self, grid, row, col, dr, dc, player_value):
        """Count consecutive pieces and open ends in a direction."""
        consecutive = 1  # Start with the piece just placed
        open_ends = 0
        
        # Check in the positive direction
        r, c = row + dr, col + dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            consecutive += 1
            r += dr
            c += dc
        
        # Check if this end is open
        if is_valid_position(r, c) and grid[r, c] == 0:
            open_ends += 1
        
        # Check in the negative direction
        r, c = row - dr, col - dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            consecutive += 1
            r -= dr
            c -= dc
        
        # Check if this end is open
        if is_valid_position(r, c) and grid[r, c] == 0:
            open_ends += 1
        
        return consecutive, open_ends
    
    def _check_blocking(self, grid, row, col, dr, dc, opponent_value):
        """Check if this move blocked opponent's connection."""
        # Temporarily remove our piece to see what was there
        temp = grid[row, col]
        grid[row, col] = 0
        
        # Count opponent's pieces in this direction
        consecutive = 0
        open_ends = 0
        
        # Check in the positive direction
        r, c = row + dr, col + dc
        while is_valid_position(r, c) and grid[r, c] == opponent_value:
            consecutive += 1
            r += dr
            c += dc
        
        # Check if this end is open
        if is_valid_position(r, c) and grid[r, c] == 0:
            open_ends += 1
        
        # Check in the negative direction
        r, c = row - dr, col - dc
        while is_valid_position(r, c) and grid[r, c] == opponent_value:
            consecutive += 1
            r -= dr
            c -= dc
        
        # Check if this end is open
        if is_valid_position(r, c) and grid[r, c] == 0:
            open_ends += 1
        
        # Restore our piece
        grid[row, col] = temp
        
        return consecutive, open_ends
    
    def _check_if_winning_move_taken(self, grid, row, col, dr, dc, player_value, valid_moves):
        """Check if there was a winning move available but not taken."""
        # Find the positions of the 3-in-a-row
        positions = [(row, col)]
        
        # Add positions in positive direction
        r, c = row + dr, col + dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            positions.append((r, c))
            r += dr
            c += dc
        
        # Add positions in negative direction
        r, c = row - dr, col - dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            positions.append((r, c))
            r -= dr
            c -= dc
        
        # Check for potential winning positions at both ends
        for direction_mult in [1, -1]:
            for base_r, base_c in positions:
                next_r = base_r + (dr * direction_mult)
                next_c = base_c + (dc * direction_mult)
                
                # If this position would complete 4-in-a-row and is a valid move
                if (is_valid_position(next_r, next_c) and grid[next_r, next_c] == 0 and
                    next_c in valid_moves):
                    
                    # Check if position is accessible (not floating)
                    if next_r == ROWS - 1 or (next_r + 1 < ROWS and grid[next_r + 1, next_c] != 0):
                        return False  # There was a winning move not taken
        
        return True  # No missed winning moves detected
    
    def _count_threats(self, grid, player_value):
        """Count number of threatening positions (simplified)."""
        threats = 0
        
        # Horizontal threats
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = grid[r, c:c+4]
                if np.count_nonzero(window == player_value) == 2 and np.count_nonzero(window == 0) == 2:
                    threats += 1
        
        # Vertical threats (simplified)
        for c in range(COLS):
            for r in range(ROWS - 3):
                window = grid[r:r+4, c]
                if np.count_nonzero(window == player_value) == 2 and np.count_nonzero(window == 0) == 2:
                    threats += 1
        
        # Diagonal threats (simplified)
        # Would implement similar logic for both diagonal directions
        
        return threats
    
    def _check_alternating_stack(self, grid, col, last_row, player_value):
        """
        Check for problematic stacking patterns while allowing strategic blocking.
        
        Args:
            grid: The game board grid
            col: The column where the move was made
            last_row: The row where the piece was placed
            player_value: The value of the current player (1 or 2)
        
        Returns:
            float: Penalty value (negative) if problematic stacking is detected
        """
        # Get the pieces in this column
        column_pieces = []
        for r in range(ROWS-1, -1, -1):  # Start from bottom
            if grid[r, col] != 0:
                column_pieces.append(grid[r, col])
            else:
                break  # Stop at first empty cell
        
        # If the column isn't very tall yet, don't worry about it
        if len(column_pieces) < 3:
            return 0.0
        
        # Check if this was a blocking move (preventing opponent's connect-4)
        was_blocking_move = False
        opponent_value = 3 - player_value  # Convert between 1 and 2
        
        # Check horizontal potential blocks
        for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:  # All four directions
            count = 0
            # Count opponent pieces in a line
            for i in range(1, 4):  # Look up to 3 steps away
                r, c = last_row + (dr * i), col + (dc * i)
                if is_valid_position(r, c) and grid[r, c] == opponent_value:
                    count += 1
                else:
                    break
            
            # Check opposite direction
            for i in range(1, 4):
                r, c = last_row - (dr * i), col - (dc * i)
                if is_valid_position(r, c) and grid[r, c] == opponent_value:
                    count += 1
                else:
                    break
            
            # If this prevented 3 in a row, it's a blocking move
            if count >= 2:
                was_blocking_move = True
                break
        
        # If this was a blocking move, don't penalize it
        if was_blocking_move:
            return 0.0
        
        # Check for isolated stack (pieces that aren't connected horizontally)
        is_isolated = True
        for r in range(max(0, last_row-1), min(ROWS, last_row+2)):
            for c in range(max(0, col-1), min(COLS, col+2)):
                if r == last_row and c == col:
                    continue  # Skip the current position
                if is_valid_position(r, c) and grid[r, c] == player_value:
                    is_isolated = False
                    break
        
        # Check for alternating pattern (X,O,X,O)
        alternating_pattern = False
        if len(column_pieces) >= 4:
            for i in range(len(column_pieces) - 3):
                if (column_pieces[i] != column_pieces[i+2]) or (column_pieces[i+1] != column_pieces[i+3]):
                    continue
                if column_pieces[i] != column_pieces[i+1]:
                    alternating_pattern = True
                    break
        
        # Apply penalties based on what we found
        if alternating_pattern and is_isolated:
            return -15.0  # Severe penalty for isolated alternating stack
        elif alternating_pattern:
            return -10.0  # Strong penalty for any alternating stack
        elif is_isolated and len(column_pieces) >= 4:
            return -8.0   # Penalty for tall isolated stack
        
        return 0.0  # No penalty if none of the bad patterns were found