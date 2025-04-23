import numpy as np
from typing import List, Tuple, Optional, Dict

from connect4.utils import ROWS, COLS, Player, is_valid_position
from connect4.game.rules import ConnectFourGame
from connect4.debug import debug, DebugLevel

class ConnectFourRewardCalculator:
    """Calculates rewards for Connect Four game states."""
    
    def __init__(self):
        # Core game outcome rewards
        self.reward_win = 5.0  # Changed from 20.0
        self.reward_draw = 0.1
        self.reward_loss = -1.0
        
        # Offensive rewards
        self.reward_two_in_row_open_ends = 0.15
        self.reward_three_in_row_open_end = 1.0  # Changed from 0.65
        self.reward_fork = 0.35
        self.penalty_missed_win = -0.5
        
        # Defensive rewards
        self.reward_block_three = 0.2
        self.reward_block_two_open = 0.1
        
        # Anti-stacking penalties
        self.penalty_stack_one = -0.2  # Changed from -2.0
        self.penalty_stack_two = -0.4  # Changed from -5.0
        self.penalty_stack_three_plus = -0.6  # Changed from -10.0
        
        # Positional rewards
        self.reward_center_column = 0.05
        self.reward_adjacent_center = 0.02
        self.reward_supported_piece = 0.03
        
        # Early game rewards
        self.reward_different_quadrants = 0.04
        self.reward_bottom_rows = 0.03
        
        # Direction multipliers
        self.multiplier_horizontal = 3.0
        self.multiplier_diagonal = 2.5
        self.multiplier_vertical = 1.0  # Changed from 0.3
        
        # Base move reward
        self.reward_base_move = -0.01
    
    def calculate_reward(self, game, last_row, last_col, last_column, action, 
                         current_player, stacking_count, valid_moves) -> Tuple[float, Dict[str, float]]:
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
            Tuple of (total reward, dictionary of reward components)
        """
        reward_components = {}
        reward = self.reward_base_move
        reward_components['base_move'] = self.reward_base_move
        
        # Convert player enum to grid value
        player_value = 1 if current_player == Player.ONE else 2
        opponent_value = 3 - player_value
        
        # Check if game is over
        if game.is_game_over():
            winner = game.get_winner()
            if winner == Player.ONE:
                reward_components['win'] = self.reward_win
                return self.reward_win, reward_components
            elif winner == Player.TWO:
                reward_components['loss'] = self.reward_loss
                return self.reward_loss, reward_components
            else:
                reward_components['draw'] = self.reward_draw
                return self.reward_draw, reward_components
        
        # Check for problematic stacking
        alternating_stack_penalty = self._check_alternating_stack(
            game.board.grid, action, last_row, player_value
        )
        if alternating_stack_penalty < 0:
            reward_components['stacking'] = alternating_stack_penalty
            reward += alternating_stack_penalty
            debug.warning(f"Problematic stacking detected in column {action}! Applied penalty: {alternating_stack_penalty}", "training")
        
        # Positional rewards
        if last_col == 3:  # Center column
            reward_components['center_column'] = self.reward_center_column
            reward += self.reward_center_column
        elif last_col in [2, 4]:  # Adjacent to center
            reward_components['adjacent_center'] = self.reward_adjacent_center
            reward += self.reward_adjacent_center
        
        # Check for piece supported from below
        if last_row < ROWS - 1 and game.board.grid[last_row + 1, last_col] == player_value:
            reward_components['supported_piece'] = self.reward_supported_piece
            reward += self.reward_supported_piece
        
        # Early game rewards
        if len(game.board.moves_made) <= 4:
            # Check if playing in different quadrants
            if game.board.moves_made.count(action) == 0:
                reward_components['different_quadrants'] = self.reward_different_quadrants
                reward += self.reward_different_quadrants
            
            # Reward for playing in bottom rows
            if last_row >= ROWS - 2:
                reward_components['bottom_rows'] = self.reward_bottom_rows
                reward += self.reward_bottom_rows
        
        # Direction-based checks for connections and blocks
        directions = [
            ((0, 1), self.multiplier_horizontal, 'horizontal'),
            ((1, 0), self.multiplier_vertical, 'vertical'),
            ((1, 1), self.multiplier_diagonal, 'diagonal_down'),
            ((-1, 1), self.multiplier_diagonal, 'diagonal_up')
        ]
        
        missed_win_detected = False
        
        for (dr, dc), direction_multiplier, direction_name in directions:
            # Analyze connections in this direction
            consecutive, open_ends = self._count_consecutive(
                game.board.grid, last_row, last_col, dr, dc, player_value
            )
            
            # Apply direction-specific rewards
            if consecutive == 2 and open_ends == 2:
                reward_value = self.reward_two_in_row_open_ends * direction_multiplier
                reward_components[f'two_in_row_{direction_name}'] = reward_value
                reward += reward_value
            elif consecutive == 3 and open_ends >= 1:
                reward_value = self.reward_three_in_row_open_end * direction_multiplier
                reward_components[f'three_in_row_{direction_name}'] = reward_value
                reward += reward_value
            
            # Check for missed win
            if consecutive == 3 and open_ends >= 1:
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
                reward_value = self.reward_block_three * direction_multiplier
                reward_components[f'block_three_{direction_name}'] = reward_value
                reward += reward_value
            elif opponent_consecutive == 2 and opponent_open_ends == 2:
                reward_value = self.reward_block_two_open * direction_multiplier
                reward_components[f'block_two_open_{direction_name}'] = reward_value
                reward += reward_value
        
        # Apply fork detection
        threats = self._count_threats(game.board.grid, player_value)
        if threats >= 2:
            reward_components['fork'] = self.reward_fork
            reward += self.reward_fork
        
        # Apply missed win penalty
        if missed_win_detected:
            reward_components['missed_win'] = self.penalty_missed_win
            reward += self.penalty_missed_win
        
        return reward, reward_components
    
    def _count_consecutive(self, grid, row, col, dr, dc, player_value):
        """Count consecutive pieces and open ends in a direction."""
        consecutive = 1
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
        temp = grid[row, col]
        grid[row, col] = 0
        
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
        
        grid[row, col] = temp
        
        return consecutive, open_ends
    
    def _check_if_winning_move_taken(self, grid, row, col, dr, dc, player_value, valid_moves):
        """Check if there was a winning move available but not taken."""
        positions = [(row, col)]
        
        r, c = row + dr, col + dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            positions.append((r, c))
            r += dr
            c += dc
        
        r, c = row - dr, col - dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            positions.append((r, c))
            r -= dr
            c -= dc
        
        for direction_mult in [1, -1]:
            for base_r, base_c in positions:
                next_r = base_r + (dr * direction_mult)
                next_c = base_c + (dc * direction_mult)
                
                if (is_valid_position(next_r, next_c) and grid[next_r, next_c] == 0 and
                    next_c in valid_moves):
                    
                    if next_r == ROWS - 1 or (next_r + 1 < ROWS and grid[next_r + 1, next_c] != 0):
                        return False
        
        return True
    
    def _count_threats(self, grid, player_value):
        """Count number of threatening positions."""
        threats = 0
        
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = grid[r, c:c+4]
                if np.count_nonzero(window == player_value) == 2 and np.count_nonzero(window == 0) == 2:
                    threats += 1
        
        for c in range(COLS):
            for r in range(ROWS - 3):
                window = grid[r:r+4, c]
                if np.count_nonzero(window == player_value) == 2 and np.count_nonzero(window == 0) == 2:
                    threats += 1
        
        return threats
    
    def _check_alternating_stack(self, grid, col, last_row, player_value):
        """
        Check for problematic stacking patterns while allowing strategic blocking.
        """
        column_pieces = []
        for r in range(ROWS-1, -1, -1):
            if grid[r, col] != 0:
                column_pieces.append(grid[r, col])
            else:
                break
        
        if len(column_pieces) < 3:
            return 0.0
        
        was_blocking_move = False
        opponent_value = 3 - player_value
        
        for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
            count = 0
            for i in range(1, 4):
                r, c = last_row + (dr * i), col + (dc * i)
                if is_valid_position(r, c) and grid[r, c] == opponent_value:
                    count += 1
                else:
                    break
            
            for i in range(1, 4):
                r, c = last_row - (dr * i), col - (dc * i)
                if is_valid_position(r, c) and grid[r, c] == opponent_value:
                    count += 1
                else:
                    break
            
            if count >= 2:
                was_blocking_move = True
                break
        
        if was_blocking_move:
            return 0.0
        
        is_isolated = True
        for r in range(max(0, last_row-1), min(ROWS, last_row+2)):
            for c in range(max(0, col-1), min(COLS, col+2)):
                if r == last_row and c == col:
                    continue
                if is_valid_position(r, c) and grid[r, c] == player_value:
                    is_isolated = False
                    break
        
        alternating_pattern = False
        if len(column_pieces) >= 4:
            for i in range(len(column_pieces) - 3):
                if (column_pieces[i] != column_pieces[i+2]) or (column_pieces[i+1] != column_pieces[i+3]):
                    continue
                if column_pieces[i] != column_pieces[i+1]:
                    alternating_pattern = True
                    break
        
        if alternating_pattern and is_isolated:
            return -0.6  # Changed from -15.0
        elif alternating_pattern:
            return -0.4  # Changed from -10.0
        elif is_isolated and len(column_pieces) >= 4:
            return -0.3  # Changed from -8.0
        elif len(column_pieces) >= 3:
            return -0.2  # Added mild penalty for non-alternating stacks
        
        return 0.0