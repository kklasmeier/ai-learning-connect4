#!/usr/bin/env python3
"""
run.py - Main entry point for Connect Four AI Learning System
"""

import sys
import os
import argparse
from connect4.debug import debug, DebugLevel

# Add the project root to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now try importing the module
try:
    from connect4.interfaces.cli import SimpleCLI
except ImportError as e:
    print(f"Error importing: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Files in connect4 directory: {os.listdir('connect4')}")
    if os.path.exists('connect4/interfaces'):
        print(f"Files in interfaces directory: {os.listdir('connect4/interfaces')}")
    else:
        print("interfaces directory not found")
    sys.exit(1)

def main():
    """Main entry point for the Connect Four AI Learning System."""
    parser = argparse.ArgumentParser(description='Connect Four AI Learning System')
    
    # Add subparsers for different components
    subparsers = parser.add_subparsers(dest='component', help='Component to run')
    
    # Game component
    game_parser = subparsers.add_parser('game', help='Run the Connect Four game')
    game_parser.add_argument('command', choices=['play', 'test', 'test_all', 'benchmark'],
                           help='Command to run for the game component')
    game_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    game_parser.add_argument('--position', type=str, help='Board position to test (for test command)')
    game_parser.add_argument('--iterations', type=int, default=1000, 
                          help='Number of iterations for benchmarking')
    game_parser.add_argument('--ai', choices=['random', 'none'], default='random',
                          help='AI opponent type for play mode')
    # AI component
    ai_parser = subparsers.add_parser('ai', help='Run AI components')
    ai_parser.add_argument('command', choices=['init', 'play', 'train'],
                        help='Command to run for the AI component')
    ai_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    ai_parser.add_argument('--debug_level', choices=['none', 'error', 'warning', 'info', 'debug', 'trace'],
                        default='error', help='Set debug level')
    ai_parser.add_argument('--hidden_size', type=int, default=128, 
                        help='Hidden layer size for neural network')
    ai_parser.add_argument('--model', type=str, help='Path to model file')
    ai_parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training')    

    # Parse arguments
    args = parser.parse_args()
    
    if args.component == 'game':
        # Run the game CLI with the parsed arguments
        cli = SimpleCLI()
        
        # Convert the run.py arguments to CLI arguments
        sys.argv = ['cli.py', args.command]
        
        # Add additional arguments if provided
        if hasattr(args, 'debug') and args.debug:
            sys.argv.append('--debug')
        
        if args.command == 'test' and args.position:
            sys.argv.extend(['--position', args.position])
        
        if args.command == 'benchmark' and args.iterations:
            sys.argv.extend(['--iterations', str(args.iterations)])
        
        if args.command == 'play' and args.ai:
            sys.argv.extend(['--ai', args.ai])
        
        cli.run()
    elif args.component == 'ai':
        try:
            from connect4.ai.dqn import DQNAgent, DQNModel
            from connect4.game.rules import ConnectFourGame
            from connect4.utils import Player 

            # Set up debugging
            if args.debug:
                # For backward compatibility, --debug still sets DEBUG level
                debug.configure(level=DebugLevel.DEBUG)
            else:
                # Use the specified level or default to ERROR
                debug_level = getattr(DebugLevel, args.debug_level.upper())
                debug.configure(level=debug_level)
            
            if args.command == 'init':
                # Initialize a new model
                print(f"Initializing new DQN model with hidden size {args.hidden_size}")
                agent = DQNAgent(hidden_size=args.hidden_size)
                
                # Save the initialized model
                os.makedirs('models', exist_ok=True)
                model_path = os.path.join('models', f'initial_model')
                agent.save(model_path)
                print(f"Model saved to {model_path}")
                
            elif args.command == 'play':
                # Load a model and play against it
                if not args.model:
                    print("Error: --model parameter required for play command")
                    return
                    
                print(f"Loading model from {args.model}")
                agent = DQNAgent.load(args.model)
                
                # Set up a game
                game = ConnectFourGame()
                print("Starting game against AI...")
                print(game.render())
                
                while not game.is_game_over():
                    current_player = game.get_current_player()
                    
                    if current_player == Player.ONE:
                        # Human player
                        valid_moves = game.get_valid_moves()
                        print(f"Valid moves: {valid_moves}")
                        
                        while True:
                            try:
                                move = int(input("Enter column (0-6): "))
                                if move in valid_moves:
                                    break
                                print(f"Invalid move. Valid options: {valid_moves}")
                            except ValueError:
                                print("Please enter a number from 0-6")
                    else:
                        # AI player
                        print("AI is thinking...")
                        valid_moves = game.get_valid_moves()
                        move = agent.get_action(game.board, valid_moves)
                        print(f"AI plays column {move}")
                    
                    game.make_move(move)
                    print(game.render())
                
                # Game over
                winner = game.get_winner()
                if winner == Player.ONE:
                    print("You win!")
                elif winner == Player.TWO:
                    print("AI wins!")
                else:
                    print("It's a draw!")
                    
            elif args.command == 'train':
                print("Training functionality will be implemented in the next phase")
                # Import the training module
                from connect4.ai.training import SelfPlayTrainer
                
                print(f"Starting training with {args.episodes} episodes")
                
                # Create or load an agent
                if args.model:
                    print(f"Loading model from {args.model}")
                    agent = DQNAgent.load(args.model)
                else:
                    print(f"Creating new agent with hidden size {args.hidden_size}")
                    agent = DQNAgent(hidden_size=args.hidden_size)
                
                # Create trainer and start training
                trainer = SelfPlayTrainer(agent=agent)
                
                # Set training parameters
                log_interval = max(1, args.episodes // 100)  # Log ~100 times
                save_interval = max(1, args.episodes // 10)  # Save ~10 checkpoints
                
                try:
                    print("Starting self-play training...")
                    trainer.train(
                        episodes=args.episodes,
                        log_interval=log_interval,
                        save_interval=save_interval
                    )
                    print("Training completed successfully")
                except KeyboardInterrupt:
                    print("\nTraining interrupted. Saving current model...")
                    trainer._save_checkpoint(trainer.episode_count, final=True)
                    print("Model saved. Exiting.")





        except ImportError as e:
            print(f"Error importing AI modules: {e}")
            print("Make sure all AI components are implemented before using this command")




    else:
        parser.print_help()

if __name__ == "__main__":
    main()