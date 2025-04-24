import numpy as np
from typing import List, Tuple, Optional, Dict

from connect4.utils import ROWS, COLS, Player, is_valid_position
from connect4.game.rules import ConnectFourGame
from connect4.debug import debug, DebugLevel

class ConnectFourRewardCalculator:
    """Calculates rewards for Connect Four game states."""
    
    def __init__(self):
        # Core game outcome rewards
        self.reward_win = 2.0  # Was 5.0
        self.reward_draw = 0.1
        self.reward_loss = -1.0
        
        # Offensive rewards
        self.reward_two_in_row_open_ends = 0.15
        self.reward_three_in_row_open_end = 0.5  # Was 1.0
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
        self.multiplier_horizontal = 2.0  # Was 3.0
        self.multiplier_diagonal = 1.5  # Was 2.5
        self.multiplier_vertical = 1.0  # Changed from 0.3
        
        # Base move reward
        self.reward_base_move = -0.01
    
    def calculate_reward(self, game, last_row, last_col, last_column, action, 
                        current_player, stacking_count, valid_moves):
        reward_components = {}
        reward = self.reward_base_move
        reward_components['base_move'] = self.reward_base_move
        
        player_value = 1 if current_player == Player.ONE else 2
        
        # Check connections first to determine if move is strategic
        two_in_row_detected = False
        three_in_row_detected = False
        directions = [
            ((0, 1), self.multiplier_horizontal, 'horizontal'),
            ((1, 0), self.multiplier_vertical, 'vertical'),
            ((1, 1), self.multiplier_diagonal, 'diagonal_down'),
            ((-1, 1), self.multiplier_diagonal, 'diagonal_up')
        ]
        for (dr, dc), direction_multiplier, direction_name in directions:
            consecutive, open_ends = self._count_consecutive(
                game.board.grid, last_row, last_col, dr, dc, player_value
            )
            if consecutive == 2 and open_ends >= 1:
                two_in_row_detected = True
                reward_value = self.reward_two_in_row_open_ends * direction_multiplier
                reward_components[f'two_in_row_{direction_name}'] = reward_value
                reward += reward_value
            elif consecutive == 3 and open_ends >= 1:
                three_in_row_detected = True
                reward_value = self.reward_three_in_row_open_end * direction_multiplier
                reward_components[f'three_in_row_{direction_name}'] = reward_value
                reward += reward_value
        
        # Apply stacking count penalties only for non-strategic consecutive plays
        debug.info(f"Stacking check: action={action}, last_column={last_column}, "
                f"stacking_count={stacking_count}, two_in_row={two_in_row_detected}, "
                f"three_in_row={three_in_row_detected}", "training")
        if (action == last_column and stacking_count > 0 and 
            not two_in_row_detected and not three_in_row_detected and 
            last_column is not None):  # Explicitly check last_column
            if stacking_count == 1:
                reward += self.penalty_stack_one
                reward_components['stacking_count'] = self.penalty_stack_one
                debug.warning(f"Applied stacking_count penalty: {self.penalty_stack_one} "
                            f"for stacking_count=1 in column {action}", "training")
            elif stacking_count == 2:
                reward += self.penalty_stack_two
                reward_components['stacking_count'] = self.penalty_stack_two
                debug.warning(f"Applied stacking_count penalty: {self.penalty_stack_two} "
                            f"for stacking_count=2 in column {action}", "training")
            elif stacking_count >= 3:
                reward += self.penalty_stack_three_plus
                reward_components['stacking_count'] = self.penalty_stack_three_plus
                debug.warning(f"Applied stacking_count penalty: {self.penalty_stack_three_plus} "
                            f"for stacking_count>=3 in column {action}", "training")
        
        # Check for problematic stacking
        alternating_stack_penalty = self._check_alternating_stack(
            game.board.grid, action, last_row, player_value
        )
        if alternating_stack_penalty < 0:
            reward_components['stacking'] = alternating_stack_penalty
            reward += alternating_stack_penalty
            debug.warning(f"Problematic stacking detected in column {action}! Applied penalty: {alternating_stack_penalty}", "training")
        
        # Positional rewards
        if last_col == 3:
            reward_components['center_column'] = self.reward_center_column
            reward += self.reward_center_column
        elif last_col in [2, 4]:
            reward_components['adjacent_center'] = self.reward_adjacent_center
            reward += self.reward_adjacent_center
        
        # Check for piece supported from below
        if last_row < ROWS - 1 and game.board.grid[last_row + 1, last_col] == player_value:
            reward_components['supported_piece'] = self.reward_supported_piece
            reward += self.reward_supported_piece
        
        # Early game rewards
        if len(game.board.moves_made) <= 4:
            if game.board.moves_made.count(action) == 0:
                reward_components['different_quadrants'] = self.reward_different_quadrants
                reward += self.reward_different_quadrants
            if last_row >= ROWS - 2:
                reward_components['bottom_rows'] = self.reward_bottom_rows
                reward += self.reward_bottom_rows
        
        # Blocking and missed win checks
        missed_win_detected = False
        for (dr, dc), direction_multiplier, direction_name in directions:
            opponent_consecutive, opponent_open_ends = self._check_blocking(
                game.board.grid, last_row, last_col, dr, dc, 3 - player_value
            )
            if opponent_consecutive == 3:
                reward_value = self.reward_block_three * direction_multiplier
                reward_components[f'block_three_{direction_name}'] = reward_value
                reward += reward_value
            elif opponent_consecutive == 2 and opponent_open_ends == 2:
                reward_value = self.reward_block_two_open * direction_multiplier
                reward_components[f'block_two_open_{direction_name}'] = reward_value
                reward += reward_value
            
            # Check for missed win
            consecutive, open_ends = self._count_consecutive(
                game.board.grid, last_row, last_col, dr, dc, player_value
            )
            if consecutive == 3 and open_ends >= 1:
                if not self._check_if_winning_move_taken(
                    game.board.grid, last_row, last_col, dr, dc, player_value, valid_moves
                ):
                    missed_win_detected = True
        
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
        consecutive = 1  # Include the placed piece
        open_ends = 0
        
        # Check in the positive direction
        r, c = row + dr, col + dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            consecutive += 1
            r += dr
            c += dc
        # Check if this end is open (empty or above/below board for vertical)
        if is_valid_position(r, c) and grid[r, c] == 0:
            open_ends += 1
        elif dr == 1 and r < 0:  # Vertical, above board
            open_ends += 1
        elif dr == -1 and r >= ROWS:  # Vertical, below board
            open_ends += 1
        
        # Check in the negative direction
        r, c = row - dr, col - dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            consecutive += 1
            r -= dr
            c += dc
        # Check if this end is open
        if is_valid_position(r, c) and grid[r, c] == 0:
            open_ends += 1
        elif dr == 1 and r >= ROWS:  # Vertical, below board
            open_ends += 1
        elif dr == -1 and r < 0:  # Vertical, above board
            open_ends += 1
        
        # Debug logging
        debug.info(f"Count consecutive: row={row}, col={col}, direction=({dr},{dc}), "
                f"player={player_value}, consecutive={consecutive}, open_ends={open_ends}", "training")
        
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
        
        # Collect consecutive pieces
        r, c = row + dr, col + dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            positions.append((r, c))
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while is_valid_position(r, c) and grid[r, c] == player_value:
            positions.append((r, c))
            r -= dr
            c += dc
        
        # Check for winning move in open ends
        for direction_mult in [1, -1]:
            for base_r, base_c in positions:
                next_r = base_r + (dr * direction_mult)
                next_c = base_c + (dc * direction_mult)
                
                # Ensure the position is valid, empty, and playable
                if (is_valid_position(next_r, next_c) and 
                    grid[next_r, next_c] == 0 and 
                    next_c in valid_moves):
                    # Check if the piece can be placed (bottom or supported)
                    if next_r == ROWS - 1 or (next_r + 1 < ROWS and grid[next_r + 1, next_c] != 0):
                        # Simulate placing the piece
                        temp_grid = grid.copy()
                        temp_grid[next_r, next_c] = player_value
                        # Verify four-in-a-row
                        count = 1
                        for i in range(1, 4):
                            r, c = next_r + (dr * i), next_c + (dc * i)
                            if is_valid_position(r, c) and temp_grid[r, c] == player_value:
                                count += 1
                            else:
                                break
                        for i in range(1, 4):
                            r, c = next_r - (dr * i), next_c - (dc * i)
                            if is_valid_position(r, c) and temp_grid[r, c] == player_value:
                                count += 1
                            else:
                                break
                        if count >= 4:
                            debug.warning(f"Missed win detected: could win in col {next_c}, chose col {col}", "training")
                            return False
        return True
    
    def _count_threats(self, grid, player_value):
        """Count number of threatening positions (two-in-a-row with two empty)."""
        threats = 0
        
        # Horizontal threats
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = grid[r, c:c+4]
                if (np.count_nonzero(window == player_value) == 2 and 
                    np.count_nonzero(window == 0) == 2):
                    # Verify at least one empty slot is playable
                    playable = False
                    for i in range(4):
                        if window[i] == 0 and (r == ROWS - 1 or grid[r + 1, c + i] != 0):
                            playable = True
                            break
                    if playable:
                        threats += 1
                        debug.info(f"Horizontal threat: row={r}, cols={c}:{c+3}, window={window}", "training")
        
        # Vertical threats
        for c in range(COLS):
            for r in range(ROWS - 3):
                window = grid[r:r+4, c]
                if (np.count_nonzero(window == player_value) == 2 and 
                    np.count_nonzero(window == 0) == 2):
                    # Verify top empty slot is playable
                    for i in range(4):
                        if window[i] == 0 and (r + i == 0 or grid[r + i - 1, c] != 0):
                            threats += 1
                            debug.info(f"Vertical threat: col={c}, rows={r}:{r+3}, window={window}", "training")
                            break
        
        return threats
    
    def _check_alternating_stack(self, grid, col, last_row, player_value):
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
            return -2.0
        elif alternating_pattern:
            return -1.5
        elif is_isolated and len(column_pieces) >= 4:
            return -1.0
        elif len(column_pieces) >= 4:  # Was >= 3
            return -0.5
        return 0.0    
        
