"""
game.py - Game state management and Gymnasium environment for Connect Four

This module provides:
1. A gymnasium-compatible environment for reinforcement learning
2. Game state management for the Connect Four game
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any, Union

from connect4.debug import debug, DebugLevel
from connect4.utils import ROWS, COLS, Player, GameResult
from connect4.game.board import Board  # Update board import


class ConnectFourEnv(gym.Env):
    """
    Connect Four environment following the OpenAI Gymnasium interface.
    
    This environment allows reinforcement learning agents to interact
    with a Connect Four game.
    """
    
    metadata = {'render_modes': ['ascii', 'human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the Connect Four environment.
        
        Args:
            render_mode: Mode for rendering the environment
        """
        debug.debug("Initializing ConnectFourEnv", "env")
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(COLS)
        
        # Observation space: 6x7 board with 3 possible values (0, 1, 2)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(ROWS, COLS), dtype=np.int8
        )
        
        # Initialize board and game state
        self.board = Board()
        self.render_mode = render_mode
        
        # Track reward settings
        self.reward_win = 1.0
        self.reward_lose = -1.0
        self.reward_draw = 0.1
        self.reward_invalid_move = -0.5
        self.reward_step = -0.01  # Small negative reward to encourage faster solutions
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Initial observation and info dictionary
        """
        debug.debug("Resetting environment", "env")
        
        # Reset random number generator
        super().reset(seed=seed)
        
        # Reset board
        self.board.reset()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment by making a move.
        
        Args:
            action: Column to place a piece (0-indexed)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        debug.debug(f"Environment step with action {action}", "env")
        
        # Default values
        reward = self.reward_step
        terminated = False
        truncated = False
        
        # Check if action is valid
        if not self.board.is_valid_move(action):
            debug.warning(f"Invalid action: {action}", "env")
            observation = self._get_observation()
            info = self._get_info()
            info['invalid_move'] = True
            
            return observation, self.reward_invalid_move, False, True, info
        
        # Make the move
        self.board.make_move(action)
        
        # Check if game is over
        if self.board.game_result == GameResult.PLAYER_ONE_WIN:
            debug.info("Game over: Player ONE wins", "env")
            reward = self.reward_win
            terminated = True
        elif self.board.game_result == GameResult.PLAYER_TWO_WIN:
            debug.info("Game over: Player TWO wins", "env")
            reward = self.reward_lose
            terminated = True
        elif self.board.game_result == GameResult.DRAW:
            debug.info("Game over: Draw", "env")
            reward = self.reward_draw
            terminated = True
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Union[str, np.ndarray]]:
        """
        Render the current state of the environment.
        
        Returns:
            Rendered frame depending on render_mode
        """
        if self.render_mode is None:
            return None
        
        if self.render_mode == "ascii":
            return self.board.render()
        
        elif self.render_mode == "human":
            print(self.board.render())
            return None
        
        elif self.render_mode == "rgb_array":
            # Create a simple RGB representation
            # This could be enhanced with better graphics later
            rgb_array = np.zeros((ROWS * 50, COLS * 50, 3), dtype=np.uint8)
            
            # Background color (dark blue)
            rgb_array[:, :] = [0, 0, 128]
            
            # Draw the pieces
            for row in range(ROWS):
                for col in range(COLS):
                    # Calculate center position of the cell
                    center_y = row * 50 + 25
                    center_x = col * 50 + 25
                    radius = 20
                    
                    # Draw a circle for each piece
                    for y in range(center_y - radius, center_y + radius):
                        for x in range(center_x - radius, center_x + radius):
                            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                                if 0 <= y < ROWS * 50 and 0 <= x < COLS * 50:
                                    if self.board.grid[row, col] == Player.EMPTY.value:
                                        rgb_array[y, x] = [0, 0, 0]  # Black for empty
                                    elif self.board.grid[row, col] == Player.ONE.value:
                                        rgb_array[y, x] = [255, 0, 0]  # Red for player 1
                                    else:
                                        rgb_array[y, x] = [255, 255, 0]  # Yellow for player 2
            
            return rgb_array
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation of the environment.
        
        Returns:
            Current board state as a numpy array
        """
        return self.board.get_state()
    
    def _get_info(self) -> Dict:
        """
        Get additional information about the current state.
        
        Returns:
            Dictionary with info about the current state
        """
        valid_moves = self.board.get_valid_moves()
        winning_line = self.board.get_winning_line()
        
        return {
            'valid_moves': valid_moves,
            'num_valid_moves': len(valid_moves),
            'current_player': self.board.current_player.value,
            'game_result': self.board.game_result.name,
            'moves_made': len(self.board.moves_made),
            'winning_line': winning_line,
            'last_move': self.board.last_move
        }
    
    def close(self):
        """Clean up resources."""
        pass


class ConnectFourGame:
    """
    High-level Connect Four game manager.
    
    This class provides a simplified interface for playing Connect Four
    outside of the reinforcement learning context.
    """
    
    def __init__(self):
        """Initialize a new Connect Four game."""
        debug.debug("Initializing ConnectFourGame", "game")
        self.board = Board()
        self.history = []
    
    def reset(self) -> None:
        """Reset the game to initial state."""
        debug.debug("Resetting game", "game")
        self.board.reset()
        self.history = []
    
    def make_move(self, column: int) -> bool:
        """
        Make a move in the game.
        
        Args:
            column: Column to place a piece (0-indexed)
            
        Returns:
            True if the move was successful, False otherwise
        """
        debug.debug(f"Game: Making move in column {column}", "game")
        
        # Store the current state for history
        if self.board.make_move(column):
            self.history.append(self.board.copy())
            return True
        
        return False
    
    def undo_move(self) -> bool:
        """
        Undo the last move.
        
        Returns:
            True if a move was undone, False otherwise
        """
        if not self.history:
            debug.debug("No moves to undo", "game")
            return False
        
        debug.debug("Undoing last move", "game")
        self.history.pop()  # Remove the last state
        
        # Reset the board and replay moves up to the desired point
        self.board.reset()
        
        # If history is empty, we're done
        if not self.history:
            return True
        
        # Otherwise, get the last board state from history
        self.board = self.history[-1].copy()
        
        return True
    
    def get_state(self) -> Board:
        """
        Get the current game state.
        
        Returns:
            The current board object
        """
        return self.board
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            True if the game is over, False otherwise
        """
        return self.board.game_result.is_game_over()
    
    def get_winner(self) -> Optional[Player]:
        """
        Get the winner of the game.
        
        Returns:
            The winning player, or None if no winner yet or draw
        """
        if self.board.game_result == GameResult.PLAYER_ONE_WIN:
            return Player.ONE
        elif self.board.game_result == GameResult.PLAYER_TWO_WIN:
            return Player.TWO
        else:
            return None
    
    def get_current_player(self) -> Player:
        """
        Get the current player.
        
        Returns:
            The current player
        """
        return self.board.current_player
    
    def get_valid_moves(self) -> List[int]:
        """
        Get a list of valid moves.
        
        Returns:
            List of valid column indices
        """
        return self.board.get_valid_moves()
    
    def render(self) -> str:
        """
        Render the game as a string.
        
        Returns:
            String representation of the board
        """
        return self.board.render()


if __name__ == "__main__":
    # Test the ConnectFourEnv
    debug.configure(level=DebugLevel.INFO)
    
    print("Testing ConnectFourEnv:")
    env = ConnectFourEnv(render_mode="human")
    observation, info = env.reset()
    
    # Make some random moves
    done = False
    while not done:
        action = np.random.choice(info['valid_moves'])
        print(f"Taking action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}, Done: {done}")
    
    print("\nTesting ConnectFourGame:")
    game = ConnectFourGame()
    print(game.render())
    
    # Make some moves
    for col in [3, 2, 4, 2, 5, 2, 6]:
        print(f"\nMaking move in column {col}")
        if game.make_move(col):
            print(game.render())
            print(f"Game over: {game.is_game_over()}")
            if game.is_game_over():
                print(f"Winner: {game.get_winner()}")
        else:
            print(f"Invalid move: {col}")
    
    # Test undo
    print("\nUndoing last move:")
    game.undo_move()
    print(game.render())