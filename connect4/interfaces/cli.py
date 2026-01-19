"""
cli.py - Command-line interface for testing Connect Four implementation

This module provides a CLI for interactively testing the Connect Four game,
analyzing board positions, and running validation tests.
"""

import argparse
import sys
import time
import random
import numpy as np
from typing import List, Optional, Tuple, Dict

from connect4.debug import debug, DebugLevel
from connect4.utils import ROWS, COLS, Player, GameResult
from connect4.game.board import Board
from connect4.game.rules import ConnectFourGame, ConnectFourEnv
from connect4.ai.minimax import MinimaxPlayer



class SimpleCLI:
    """Simple command-line interface for Connect Four testing."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.game = ConnectFourGame()
        self.args = None
        self.minimax_player = None  # Initialized when needed
    
    def parse_args(self) -> None:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description='Connect Four CLI')
        
        # Main commands
        subparsers = parser.add_subparsers(dest='command', help='Command to run')
        
        # Play command
        play_parser = subparsers.add_parser('play', help='Play a game interactively')
        play_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        play_parser.add_argument('--ai', choices=['random', 'minimax', 'none'], default='random',
                                 help='AI opponent type')
        play_parser.add_argument('--depth', type=int, default=6,
                                 help='Search depth for minimax AI (default: 6)')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Test specific scenarios')
        test_parser.add_argument('--position', type=str, help='Board position to test')
        
        # Test all command
        test_all_parser = subparsers.add_parser('test_all', help='Run all validation tests')
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark performance')
        benchmark_parser.add_argument('--iterations', type=int, default=1000,
                                    help='Number of iterations for benchmarking')
        
        # Parse arguments
        self.args = parser.parse_args()
        
        # Set debug level
        if hasattr(self.args, 'debug') and self.args.debug:
            debug.configure(level=DebugLevel.DEBUG)
        else:
            debug.configure(level=DebugLevel.ERROR)
        
        # Initialize minimax player if needed
        if hasattr(self.args, 'ai') and self.args.ai == 'minimax':
            self.minimax_player = MinimaxPlayer(depth=self.args.depth)
    
    def run(self) -> None:
        """Run the CLI based on the parsed arguments."""
        if not self.args:
            self.parse_args()
        
        if self.args.command == 'play':
            self.play_game()
        elif self.args.command == 'test':
            self.test_position()
        elif self.args.command == 'test_all':
            self.run_all_tests()
        elif self.args.command == 'benchmark':
            self.benchmark()
        else:
            print("Please specify a command. Use --help for options.")
            sys.exit(1)
    
    def play_game(self) -> None:
        """Play a Connect Four game interactively."""
        print("Starting a new Connect Four game!")
        if self.args.ai == 'minimax':
            print(f"Playing against Minimax AI (depth={self.args.depth})")
        elif self.args.ai == 'random':
            print("Playing against Random AI")
        else:
            print("Two player mode")
        print("Enter column number (0-6) to make a move.")
        print("Other commands: 'q' to quit, 'u' to undo, 'r' to restart.")
        
        self.game.reset()
        print(self.game.render())
        
        while not self.game.is_game_over():
            current_player = self.game.get_current_player()
            
            # Human player's turn (or both players if ai='none')
            if current_player == Player.ONE or self.args.ai == 'none':
                move = self.get_human_move()
                
                # Process special commands
                if move is None:
                    continue
                elif move == -1:  # Quit
                    print("Quitting game.")
                    return
                elif move == -2:  # Undo
                    if self.game.undo_move():
                        print("Move undone.")
                        print(self.game.render())
                    else:
                        print("No moves to undo.")
                    continue
                elif move == -3:  # Restart
                    self.game.reset()
                    print("Game restarted.")
                    print(self.game.render())
                    continue
            
            # AI player's turn
            else:
                print("AI is thinking...")
                start_time = time.time()
                move = self.get_ai_move()
                elapsed = time.time() - start_time
                
                if self.args.ai == 'minimax':
                    print(f"AI plays column {move} ({elapsed:.2f}s, {self.minimax_player.nodes_evaluated} nodes)")
                else:
                    print(f"AI plays column {move}")
            
            # Make the move
            if self.game.make_move(move):
                print(self.game.render())
            else:
                print(f"Invalid move: {move}")
        
        # Game over
        print("Game over!")
        winner = self.game.get_winner()
        if winner == Player.ONE:
            print("You win! Congratulations!")
        elif winner == Player.TWO:
            print("AI wins! Better luck next time.")
        else:
            print("It's a draw!")
    
    def get_human_move(self) -> Optional[int]:
        """
        Get a move from human player input.
        
        Returns:
            Column index, or special command code, or None if invalid input
        """
        while True:
            try:
                user_input = input(f"Your move (columns 0-6, q/u/r): ").strip().lower()
                
                # Check for special commands
                if user_input == 'q':
                    return -1  # Quit
                elif user_input == 'u':
                    return -2  # Undo
                elif user_input == 'r':
                    return -3  # Restart
                
                # Parse column number
                move = int(user_input)
                if 0 <= move < COLS:
                    return move
                else:
                    print(f"Column must be between 0 and {COLS-1}.")
            except ValueError:
                print("Invalid input. Please enter a column number or special command.")
            
            return None  # Invalid input
    
    def get_ai_move(self) -> int:
        """
        Get a move from the AI.
        
        Returns:
            Column index for the AI's move
        """
        if self.args.ai == 'minimax':
            # Use minimax player - it works directly with the Board
            return self.minimax_player.get_move(self.game.board)
        elif self.args.ai == 'random':
            valid_moves = self.game.get_valid_moves()
            return random.choice(valid_moves)
        else:
            # Fallback to random
            valid_moves = self.game.get_valid_moves()
            return random.choice(valid_moves)
    
    def test_position(self) -> None:
        """Test a specific board position."""
        if not self.args.position:
            print("Please provide a position string with --position")
            return
        
        # Parse the position string
        try:
            position = [int(c) for c in self.args.position.split(',')]
            if len(position) != ROWS * COLS:
                raise ValueError(f"Position string must have {ROWS * COLS} values")
            
            # Create a board with this position
            board = Board()
            board.grid = np.array(position).reshape(ROWS, COLS)
            
            print("Loaded position:")
            print(board.render())
            
            # Test win conditions
            print("\nTesting win conditions:")
            has_win = False
            
            # Check for win for each player
            for player in [Player.ONE, Player.TWO]:
                for row in range(ROWS):
                    for col in range(COLS):
                        if board.grid[row, col] == player.value:
                            if self.check_win_from_position(board.grid, row, col):
                                print(f"Win for {player} detected at ({row}, {col})")
                                has_win = True
            
            if not has_win:
                print("No win detected for any player")
            

            # Check for full board
            if np.all(board.grid != Player.EMPTY.value):
                print("Board is full")
            else:
                empty_count = np.sum(board.grid == Player.EMPTY.value)
                print(f"Board has {empty_count} empty cells")
        
        except Exception as e:
            print(f"Error processing position: {e}")
    
    def run_all_tests(self) -> None:
        """Run all validation tests."""
        print("Running validation tests...")
        
        # Run win detection tests
        print("\n" + "="*50)
        print("WIN DETECTION TESTS")
        print("="*50)
        
        win_tests = self.create_win_tests()
        passed = 0
        failed = 0
        
        for i, (board, expected) in enumerate(win_tests):
            actual = self.test_win_condition(board)
            status = "PASSED" if actual == expected else "FAILED"
            
            if actual == expected:
                passed += 1
            else:
                failed += 1
                print(f"\nTest {i+1}: {status}")
                print(f"Expected: {expected}, Got: {actual}")
                print(board.render())
        
        print(f"\nWin detection: {passed}/{passed+failed} tests passed")
        
        # Run draw detection tests
        print("\n" + "="*50)
        print("DRAW DETECTION TESTS")
        print("="*50)
        
        draw_tests = self.create_draw_tests()
        passed = 0
        failed = 0
        
        for i, (board, expected) in enumerate(draw_tests):
            actual = self.test_draw_condition(board)
            status = "PASSED" if actual == expected else "FAILED"
            
            if actual == expected:
                passed += 1
            else:
                failed += 1
                print(f"\nTest {i+1}: {status}")
                print(f"Expected: {expected}, Got: {actual}")
                print(board.render())
        
        print(f"\nDraw detection: {passed}/{passed+failed} tests passed")
        
        # Run move validation tests
        print("\n" + "="*50)
        print("MOVE VALIDATION TESTS")
        print("="*50)
        
        move_tests = self.create_move_validation_tests()
        passed = 0
        failed = 0
        
        for i, (board, move, expected) in enumerate(move_tests):
            actual = board.is_valid_move(move)
            status = "PASSED" if actual == expected else "FAILED"
            
            if actual == expected:
                passed += 1
            else:
                failed += 1
                print(f"\nTest {i+1}: {status}")
                print(f"Move: {move}, Expected: {expected}, Got: {actual}")
                print(board.render())
        
        print(f"\nMove validation: {passed}/{passed+failed} tests passed")
    
    def create_win_tests(self) -> List[Tuple[Board, bool]]:
        """Create test cases for win detection."""
        tests = []
        
        # Test 1: Horizontal win for Player 1
        board = Board()
        board.grid[5, 0] = Player.ONE.value
        board.grid[5, 1] = Player.ONE.value
        board.grid[5, 2] = Player.ONE.value
        board.grid[5, 3] = Player.ONE.value
        tests.append((board, True))
        
        # Test 2: Vertical win for Player 2
        board = Board()
        board.grid[2, 3] = Player.TWO.value
        board.grid[3, 3] = Player.TWO.value
        board.grid[4, 3] = Player.TWO.value
        board.grid[5, 3] = Player.TWO.value
        tests.append((board, True))
        
        # Test 3: Diagonal win (down-right) for Player 1
        board = Board()
        board.grid[2, 0] = Player.ONE.value
        board.grid[3, 1] = Player.ONE.value
        board.grid[4, 2] = Player.ONE.value
        board.grid[5, 3] = Player.ONE.value
        tests.append((board, True))
        
        # Test 4: Diagonal win (up-right) for Player 2
        board = Board()
        board.grid[5, 0] = Player.TWO.value
        board.grid[4, 1] = Player.TWO.value
        board.grid[3, 2] = Player.TWO.value
        board.grid[2, 3] = Player.TWO.value
        tests.append((board, True))
        
        # Test 5: No win (only 3 in a row)
        board = Board()
        board.grid[5, 0] = Player.ONE.value
        board.grid[5, 1] = Player.ONE.value
        board.grid[5, 2] = Player.ONE.value
        tests.append((board, False))
        
        # Test 6: Empty board (no win)
        board = Board()
        tests.append((board, False))
        
        return tests
    
    def create_draw_tests(self) -> List[Tuple[Board, bool]]:
        """Create test cases for draw detection."""
        tests = []
        
        # Test 1: Empty board (not a draw)
        board = Board()
        tests.append((board, False))
        
        # Test 2: Nearly full board (one space left)
        board = Board()
        for row in range(ROWS):
            for col in range(COLS):
                if row != 0 or col != 3:  # Leave one space empty
                    board.grid[row, col] = Player.ONE.value if (row + col) % 2 == 0 else Player.TWO.value
        tests.append((board, False))
        
        return tests
    
    def create_move_validation_tests(self) -> List[Tuple[Board, int, bool]]:
        """Create test cases for move validation."""
        tests = []
        
        # Test 1: Valid move in empty column
        board = Board()
        tests.append((board, 3, True))
        
        # Test 2: Valid move in partially filled column
        board = Board()
        board.grid[ROWS-1, 2] = Player.ONE.value
        tests.append((board, 2, True))
        
        # Test 3: Invalid move in full column
        board = Board()
        for row in range(ROWS):
            board.grid[row, 4] = Player.TWO.value if row % 2 == 0 else Player.ONE.value
        tests.append((board, 4, False))
        
        # Test 4: Invalid column index (too low)
        board = Board()
        tests.append((board, -1, False))
        
        # Test 5: Invalid column index (too high)
        board = Board()
        tests.append((board, COLS, False))
        
        return tests
    
    def test_win_condition(self, board: Board) -> bool:
        """
        Test if the board has a win condition.
        
        Args:
            board: The board to test
        
        Returns:
            True if there is a win, False otherwise
        """
        # Check for win for each player
        for player in [Player.ONE, Player.TWO]:
            for row in range(ROWS):
                for col in range(COLS):
                    if board.grid[row, col] == player.value:
                        if self.check_win_from_position(board.grid, row, col):
                            return True
        
        return False
    
    def check_win_from_position(self, grid: np.ndarray, row: int, col: int) -> bool:
        """
        Check if there's a win starting from a specific position.
        
        Args:
            grid: The board grid
            row: Row index
            col: Column index
        
        Returns:
            True if there is a win, False otherwise
        """
        player_value = grid[row, col]
        if player_value == Player.EMPTY.value:
            return False
        
        # Check all four directions
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal down-right
            (-1, 1)   # Diagonal up-right
        ]
        
        for dr, dc in directions:
            count = 1  # Start with 1 for the current position
            
            # Check in the positive direction
            r, c = row + dr, col + dc
            while (0 <= r < ROWS and 0 <= c < COLS and 
                   grid[r, c] == player_value):
                count += 1
                r += dr
                c += dc
            
            # Check in the negative direction
            r, c = row - dr, col - dc
            while (0 <= r < ROWS and 0 <= c < COLS and 
                   grid[r, c] == player_value):
                count += 1
                r -= dr
                c -= dc
            
            if count >= 4:
                return True
        
        return False
    
    def test_draw_condition(self, board: Board) -> bool:
        """
        Test if the board is in a draw state.
        
        Args:
            board: The board to test
        
        Returns:
            True if it's a draw, False otherwise
        """
        # Check if board is full
        is_full = np.all(board.grid != Player.EMPTY.value)
        
        # Check if there's no winner
        has_win = self.test_win_condition(board)
        
        return is_full and not has_win
    
    def benchmark(self) -> None:
        """Benchmark the performance of the Connect Four implementation."""
        print(f"Running benchmark with {self.args.iterations} iterations...")
        
        # Benchmark board initialization
        debug.start_timer("board_init")
        for _ in range(self.args.iterations):
            board = Board()
        board_init_time = debug.end_timer("board_init")
        print(f"Board initialization: {board_init_time:.6f} seconds total, "
              f"{board_init_time/self.args.iterations*1000:.6f} ms per board")
        
        # Benchmark move making
        board = Board()
        debug.start_timer("moves")
        moves_made = 0
        for _ in range(self.args.iterations):
            col = random.randint(0, COLS-1)
            if board.is_valid_move(col):
                board.make_move(col)
                moves_made += 1
                
                # Reset board if game is over
                if board.game_result.is_game_over():
                    board.reset()
        moves_time = debug.end_timer("moves")
        print(f"Making {moves_made} moves: {moves_time:.6f} seconds total, "
              f"{moves_time/moves_made*1000:.6f} ms per move")
        
        # Benchmark win checking
        debug.start_timer("win_check")
        checks_done = 0
        for _ in range(self.args.iterations):
            # Create a random board
            board = Board()
            for _ in range(random.randint(7, 20)):
                col = random.randint(0, COLS-1)
                if board.is_valid_move(col):
                    board.make_move(col)
            
            # Check for win
            for row in range(ROWS):
                for col in range(COLS):
                    if board.grid[row, col] != Player.EMPTY.value:
                        self.check_win_from_position(board.grid, row, col)
                        checks_done += 1
        win_check_time = debug.end_timer("win_check")
        print(f"Performing {checks_done} win checks: {win_check_time:.6f} seconds total, "
              f"{win_check_time/checks_done*1000:.6f} ms per check")
        
        # Benchmark game simulation
        debug.start_timer("game_simulation")
        games_played = 0
        total_moves = 0
        for _ in range(self.args.iterations // 10):  # Fewer iterations for full games
            game = ConnectFourGame()
            move_count = 0
            
            while not game.is_game_over():
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break
                
                col = random.choice(valid_moves)
                game.make_move(col)
                move_count += 1
            
            games_played += 1
            total_moves += move_count
        
        simulation_time = debug.end_timer("game_simulation")
        print(f"Played {games_played} games with {total_moves} total moves: "
              f"{simulation_time:.6f} seconds total, "
              f"{simulation_time/games_played*1000:.6f} ms per game, "
              f"{simulation_time/total_moves*1000:.6f} ms per move")
        
        # Benchmark rendering
        board = Board()
        for _ in range(10):  # Add some pieces to the board
            col = random.randint(0, COLS-1)
            if board.is_valid_move(col):
                board.make_move(col)
        
        debug.start_timer("rendering")
        for _ in range(self.args.iterations):
            board.render()
        rendering_time = debug.end_timer("rendering")
        print(f"Rendering board {self.args.iterations} times: {rendering_time:.6f} seconds total, "
              f"{rendering_time/self.args.iterations*1000:.6f} ms per render")


def main():
    """Main entry point for the CLI."""
    cli = SimpleCLI()
    cli.run()


if __name__ == "__main__":
    main()