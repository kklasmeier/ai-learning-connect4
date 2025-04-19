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
    

    # Main parser
    parser = argparse.ArgumentParser(
        description='Connect Four AI Learning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:

    GAME COMPONENT:
    ---------------
    # Play Connect Four against a random AI
    python run.py game play
    
    # Play Connect Four with two human players
    python run.py game play --ai none
    
    # Run tests on a specific board position
    python run.py game test --position 0,0,0,0,1,1,1,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    
    # Run all validation tests
    python run.py game test_all
    
    # Benchmark performance with 5000 iterations
    python run.py game benchmark --iterations 5000

    AI COMPONENT - MODEL MANAGEMENT:
    --------------------------------
    # Initialize a new model with default settings (128 hidden size)
    python run.py ai init
    
    # Initialize a model with a larger hidden layer
    python run.py ai init --hidden_size 256
    
    # Play against a trained model
    python run.py ai play --model models/final_model_TIMESTAMP
    
    # Play with debug information shown
    python run.py ai play --model models/final_model_TIMESTAMP --debug_level info

    AI COMPONENT - TRAINING:
    ------------------------
    # Train the AI for 1000 episodes
    python run.py ai train --episodes 1000
    
    # Train with a specific log interval
    python run.py ai train --episodes 500 --log_interval 10
    
    # Train with a larger hidden layer size
    python run.py ai train --episodes 1000 --hidden_size 256
    
    # Continue training from an existing model
    python run.py ai train --episodes 2000 --model models/final_model_TIMESTAMP
    
    # Train with detailed logging
    python run.py ai train --episodes 100 --debug_level debug

    AI COMPONENT - JOB MANAGEMENT:
    ------------------------------
    # View all training jobs
    python run.py ai jobs
    
    # View details of a specific job
    python run.py ai jobs --job_id 1
    
    # Purge all data (requires confirmation)
    python run.py ai purge --confirm

    AI COMPONENT - GAME REPLAY:
    ---------------------------
    # List all saved games
    python run.py ai games --list
    
    # List games for a specific job
    python run.py ai games --list --job_id 1
    
    # Replay a specific game
    python run.py ai games --replay 0
    
    # Replay a game with slower movement
    python run.py ai games --replay 5 --delay 1.0
    """
    )

    # Add subparsers for different components
    subparsers = parser.add_subparsers(dest='component', help='Component to run')
    

    # Game component
    game_parser = subparsers.add_parser('game', 
        help='Run the Connect Four game',
        description='Play Connect Four or run game tests')
    game_parser.add_argument('command', 
        choices=['play', 'test', 'test_all', 'benchmark'],
        help='Game command: play (interactive game), test (test specific position), '
            'test_all (run all validation tests), benchmark (performance testing)')
    game_parser.add_argument('--debug', 
        action='store_true', 
        help='Enable debug mode with detailed logging')
    game_parser.add_argument('--position', 
        type=str, 
        help='Board position to test (comma-separated values for test command)')
    game_parser.add_argument('--iterations', 
        type=int, 
        default=1000, 
        help='Number of iterations for benchmarking')
    game_parser.add_argument('--ai', 
        choices=['random', 'none'], 
        default='random',
        help='AI opponent type: random (makes random moves), none (two human players)')

    # AI component
    ai_parser = subparsers.add_parser('ai', 
        help='Run AI components',
        description='Train, play against, or evaluate Connect Four AI')
    ai_parser.add_argument('command', 
        choices=['init', 'play', 'train', 'jobs', 'purge', 'games'],
        help="""AI commands:
        init: Initialize a new model
        play: Play against a trained model
        train: Train a model through self-play
        jobs: View training job information
        purge: Clear all data (jobs, logs, games)
        games: List and replay saved games""")

    # Common AI arguments
    ai_parser.add_argument('--debug', 
        action='store_true', 
        help='Enable debug mode (equivalent to --debug_level debug)')
    ai_parser.add_argument('--debug_level', 
        choices=['none', 'error', 'warning', 'info', 'debug', 'trace'],
        default='error', 
        help='Set debug level: none (silent), error, warning, info, debug, trace (most verbose)')

    # Model arguments
    model_group = ai_parser.add_argument_group('Model options')
    model_group.add_argument('--hidden_size', 
        type=int, 
        default=128, 
        help='Hidden layer size for neural network (larger = more capacity but slower training)')
    model_group.add_argument('--model', 
        type=str, 
        help='Path to saved model file for loading (used with play and train commands)')

    # Training arguments
    training_group = ai_parser.add_argument_group('Training options')
    training_group.add_argument('--episodes', 
        type=int, 
        default=1000,
        help='Number of episodes for training (more = better learning but longer time)')
    training_group.add_argument('--log_interval', 
        type=int, 
        default=None,
        help='Interval between training log updates (default: episodes/100)')

    # Job and games arguments
    data_group = ai_parser.add_argument_group('Data options')
    data_group.add_argument('--job_id', 
        type=int, 
        help='Job ID for commands that need it (jobs, games)')
    data_group.add_argument('--list', 
        action='store_true', 
        help='List saved games (used with games command)')
    data_group.add_argument('--replay', 
        type=int, 
        help='Replay a specific game by ID (used with games command)')
    data_group.add_argument('--delay', 
        type=float, 
        default=0.5, 
        help='Delay between moves during replay in seconds (used with games --replay)')
    data_group.add_argument('--confirm', 
        action='store_true', 
        help='Confirm dangerous operations like purge (required for purge command)')


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
                
                # Show job ID information
                print(f"Created training job with ID: {trainer.job_id}")
                print(f"You can check status with: python run.py ai jobs --job_id {trainer.job_id}")
                
                # Set training parameters
                log_interval = args.log_interval if hasattr(args, 'log_interval') and args.log_interval is not None else max(1, args.episodes // 100)
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
            elif args.command == 'jobs':
                # Import the data manager for job listing
                from connect4.data.data_manager import get_job_data, get_episode_logs
                
                # List all jobs or view a specific job
                job_id = None
                if hasattr(args, 'job_id') and args.job_id is not None:
                    job_id = args.job_id
                    
                jobs = get_job_data(job_id)
                
                if job_id is not None:
                    # Show details for a specific job
                    if not jobs:
                        print(f"Job {job_id} not found")
                    else:
                        print(f"Job {job_id} Details:")
                        print(f"Started: {jobs['start_time']}")
                        print(f"Status: {jobs['status']}")
                        print(f"Episodes: {jobs['episodes_completed']}/{jobs['total_episodes']}")
                        
                        if jobs['end_time']:
                            print(f"Completed: {jobs['end_time']}")
                        
                        print("\nTraining Parameters:")
                        for key, value in jobs['parameters'].items():
                            print(f"  {key}: {value}")
                            
                        # Show recent episodes
                        print("\nRecent Episodes:")
                        logs = get_episode_logs(job_id)[-5:]  # Last 5 episodes
                        for log in logs:
                            print(f"Episode {log['episode']}: Reward={log['reward']:.2f}, "
                                f"Length={log['length']}, Winner={log['winner']}")
                else:
                    # List all jobs
                    print(f"Found {len(jobs)} training jobs:")
                    for job in jobs:
                        status = job['status']
                        progress = f"{job['episodes_completed']}/{job['total_episodes']}"
                        print(f"Job {job['job_id']}: {status}, Progress: {progress}")
                        
                    print("\nUse 'python run.py ai jobs --job_id <id>' to view job details")

            elif args.command == 'purge':
                import shutil
                import os
                
                if not args.confirm:
                    print("WARNING: This will delete all jobs, logs, games, and training data.")
                    print("To confirm, run: python run.py ai purge --confirm")
                    return
                
                print("Purging all data...")
                
                # Clear jobs and models JSON files
                if os.path.exists('data/jobs.json'):
                    with open('data/jobs.json', 'w') as f:
                        f.write('[]')
                    print("Cleared jobs.json")
                
                if os.path.exists('data/models.json'):
                    with open('data/models.json', 'w') as f:
                        f.write('[]')
                    print("Cleared models.json")
                
                # Remove log files
                if os.path.exists('data/logs'):
                    log_files = os.listdir('data/logs')
                    for file in log_files:
                        os.remove(os.path.join('data/logs', file))
                    print(f"Removed {len(log_files)} log files")
                
                # Remove game files
                if os.path.exists('data/games'):
                    game_files = os.listdir('data/games')
                    for file in game_files:
                        os.remove(os.path.join('data/games', file))
                    print(f"Removed {len(game_files)} game files")
                
                print("Purge completed")            

            elif args.command == 'games':
                from connect4.data.data_manager import get_saved_games
                from connect4.game.rules import ConnectFourGame
                import time
                
                if args.replay is not None:
                    # Replay a specific game
                    all_games = get_saved_games(args.job_id)
                    game_id = args.replay
                    
                    if 0 <= game_id < len(all_games):
                        game_data = all_games[game_id]
                        print(f"Replaying game {game_id}: Job {game_data['job_id']}, Episode {game_data['episode']}")
                        print(f"Winner: {game_data['winner'] or 'Draw'}, Length: {game_data['game_length']}")
                        
                        # Create a game and replay the moves
                        game = ConnectFourGame()
                        print("\nInitial board:")
                        print(game.render())
                        time.sleep(args.delay)
                        
                        for i, move in enumerate(game_data['moves']):
                            current_player = "X" if game.get_current_player() == Player.ONE else "O"
                            print(f"\nMove {i+1}: Player {current_player} plays column {move}")
                            game.make_move(move)
                            print(game.render())
                            time.sleep(args.delay)
                        
                        # Show final result
                        winner = game.get_winner()
                        if winner:
                            winner_str = "X" if winner == Player.ONE else "O"
                            print(f"\nGame over! {winner_str} wins!")
                        else:
                            print("\nGame over! It's a draw!")
                    else:
                        print(f"Error: Game ID {game_id} not found")
                        print(f"Available game IDs: 0-{len(all_games)-1}")
                elif args.list:
                    # List available games
                    job_id = args.job_id
                    all_games = get_saved_games(job_id)
                    
                    if not all_games:
                        print("No saved games found")
                        if job_id is not None:
                            print(f"No games found for job {job_id}")
                        return
                    
                    print(f"Found {len(all_games)} saved games:")
                    print("\nID | Job | Episode | Winner | Moves | Timestamp")
                    print("-" * 60)
                    
                    for i, game in enumerate(all_games):
                        winner = game['winner'] or "Draw"
                        timestamp = game['timestamp'].split('T')[0] if 'timestamp' in game else "Unknown"
                        print(f"{i:2d} | {game['job_id']:3d} | {game['episode']:7d} | {winner:6s} | {game['game_length']:5d} | {timestamp}")
                    
                    print("\nTo replay a game: python run.py ai games --replay GAME_ID")
                    if job_id is not None:
                        print(f"Currently showing games for job {job_id} only")
                        print("To see all games: python run.py ai games --list")
                    else:
                        print("To filter by job: python run.py ai games --list --job_id JOB_ID")
                else:
                    print("Please specify an action: --list or --replay")
                    print("Example: python run.py ai games --list")
                    print("Example: python run.py ai games --replay 0 --delay 1.0")

        except ImportError as e:
            print(f"Error importing AI modules: {e}")
            print("Make sure all AI components are implemented before using this command")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()