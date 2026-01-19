#!/usr/bin/env python3
"""
run.py - Main entry point for Connect Four AI Learning System
"""

import sys
import os
import argparse
import time
from connect4.debug import debug, DebugLevel
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored output
colorama.init()

# Add the project root to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def highlight_last_move(board_str, column, player_symbol):
    """
    Highlight the most recently played piece on the board.
    
    Args:
        board_str: The rendered board string from game.render()
        column: The column where the piece was played (0-6)
        player_symbol: 'X' or 'O' - the piece that was just played
    
    Returns:
        Board string with the last played piece highlighted
    """
    lines = board_str.split('\n')
    
    # Board layout: each cell is 2 chars wide, columns are at positions 1, 3, 5, 7, 9, 11, 13
    # Format: |X   O   X   O   X   O   X|
    # The piece position in each row is: 1 + (column * 2)
    piece_pos = 1 + (column * 2)
    
    # Find the topmost row containing the player's piece in this column
    # Skip header line (index 0) and footer lines (last 2)
    # Board rows are indices 1-6 (6 rows, top to bottom)
    for i in range(1, 7):  # rows 1-6 are the playable area
        line = lines[i]
        if len(line) > piece_pos and line[piece_pos] == player_symbol:
            # Found the piece - highlight it with bright/bold color
            highlight_color = Fore.LIGHTGREEN_EX if player_symbol == 'X' else Fore.LIGHTCYAN_EX
            highlighted_piece = f"{highlight_color}{Style.BRIGHT}{player_symbol}{Style.RESET_ALL}"
            # Replace just this occurrence
            lines[i] = line[:piece_pos] + highlighted_piece + line[piece_pos + 1:]
            break
    
    return '\n'.join(lines)

def configure_debug(args):
    """Configure debug level based on args.debug or args.debug_level."""
    if args.debug:
        debug.configure(level=DebugLevel.DEBUG)
    else:
        debug_level = getattr(DebugLevel, args.debug_level.upper())
        debug.configure(level=debug_level)

def ensure_directory(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def parse_layer_sizes(layer_sizes_str):
    """Parse comma-separated layer sizes or return None on error."""
    if not layer_sizes_str:
        return None
    try:
        return [int(size) for size in layer_sizes_str.split(',')]
    except ValueError:
        print(f"Error parsing layer sizes '{layer_sizes_str}'. Using default configuration.")
        return None

def create_or_load_agent(args, model_path=None):
    """Create a new agent or load an existing one based on args."""
    from connect4.ai.dqn import DQNAgent
    
    hidden_sizes = parse_layer_sizes(getattr(args, 'layer_sizes', None))
    
    if model_path:
        print(f"Loading model from {model_path}")
        agent = DQNAgent.load(model_path)
        if hasattr(agent.model, 'hidden_sizes'):
            print(f"Model architecture: {agent.model.hidden_sizes}")
        return agent
    
    initial_model_path = os.path.join('models', 'initial_model_model.pt')
    if not hidden_sizes and os.path.exists(initial_model_path):
        print(f"Loading initial model from {initial_model_path}")
        agent = DQNAgent.load('models/initial_model')
        if hasattr(agent.model, 'hidden_sizes'):
            print(f"Model architecture: {agent.model.hidden_sizes}")
        return agent
    
    if hidden_sizes:
        print(f"Creating new agent with custom architecture: {hidden_sizes}")
        return DQNAgent(hidden_sizes=hidden_sizes)
    
    layers = getattr(args, 'layers', 1)
    hidden_size = getattr(args, 'hidden_size', 128)
    print(f"Creating new agent with {layers} layer(s) of size {hidden_size}")
    return DQNAgent(hidden_size=hidden_size, layers=layers)

def replay_game(game_data, delay, show_metrics=False, job_id=None):
    """Replay a game with per-turn statistics."""
    from connect4.game.rules import ConnectFourGame
    from connect4.data.data_manager import get_episode_logs
    from connect4.utils import Player
    
    episode = game_data['episode']
    print(f"Replaying game: Job {game_data['job_id']}, Episode {episode}")
    print(f"Winner: {game_data['winner'] or 'Draw'}, Length: {game_data['game_length']}")
    
    # Load episode metrics if available
    episode_log = None
    if job_id is not None:
        try:
            episode_logs = get_episode_logs(job_id)
            if episode_logs is None:
                print(f"Error: get_episode_logs returned None for job {job_id}")
            elif not episode_logs:
                print(f"No episode logs found for job {job_id}")
            else:
                print(f"Found {len(episode_logs)} episode logs for job {job_id}")
                episode_log = next((log for log in episode_logs if log['episode'] == episode), None)
                if episode_log:
                    print("\nEpisode Metrics:")
                    print(f"Reward: {episode_log.get('reward', 'N/A')}")
                    print(f"Epsilon: {episode_log.get('epsilon', 'N/A')}")
                    print(f"Loss: {episode_log.get('loss', 'N/A')}")
                else:
                    print(f"No log found for episode {episode} in job {job_id}")
        except Exception as e:
            print(f"Error retrieving episode metrics for job {job_id}, episode {episode}: {str(e)}")
    
    game = ConnectFourGame()
    board_lines = game.render().split('\n')
    print("\nInitial board:")
    for line in board_lines:
        print(line)
    time.sleep(delay)
    
    for i, move in enumerate(game_data['moves']):
        current_player = "X" if game.get_current_player() == Player.ONE else "O"
        move_info = f"Episode: {episode}, Move {i+1}: Player {current_player} plays column {move['column']}"
        
        # Prepare stats - handle both old and new formats
        stats_lines = []
        
        # Check for new simplified format (from new trainer)
        if 'action_source' in move:
            action_source = move.get('action_source', 'unknown')
            epsilon = move.get('epsilon')
            curriculum_stage = move.get('curriculum_stage')
            curriculum_stage_name = move.get('curriculum_stage_name')
            
            # Color code based on who made the move
            if action_source.startswith('opponent_'):
                # Opponent move - parse the specific type
                source_color = Fore.CYAN
                opponent_detail = action_source.replace('opponent_', '')
                
                # Make the display more readable
                if opponent_detail == 'random':
                    source_display = "RANDOM"
                elif opponent_detail.startswith('minimax_d'):
                    depth = opponent_detail.replace('minimax_d', '')
                    source_display = f"MINIMAX (depth {depth})"
                elif opponent_detail == 'model':
                    source_display = "MODEL"
                elif opponent_detail == 'self':
                    source_display = "SELF-PLAY"
                else:
                    source_display = opponent_detail.upper()
                
                stats_lines.append(f"{source_color}Source: {source_display} (opponent, Player {current_player}){Style.RESET_ALL}")
            elif action_source == 'agent_model':
                # Agent using learned policy
                source_color = Fore.GREEN
                stats_lines.append(f"{source_color}Source: AGENT (model, Player {current_player}){Style.RESET_ALL}")
                if epsilon is not None:
                    stats_lines.append(f"Epsilon: {epsilon:.3f}")
            elif action_source == 'agent_random':
                # Agent exploring randomly
                source_color = Fore.YELLOW
                stats_lines.append(f"{source_color}Source: AGENT (random exploration, Player {current_player}){Style.RESET_ALL}")
                if epsilon is not None:
                    stats_lines.append(f"Epsilon: {epsilon:.3f}")
            elif action_source == 'model':
                # Legacy format - agent using learned policy
                source_color = Fore.GREEN
                stats_lines.append(f"{source_color}Source: AGENT (model, Player {current_player}){Style.RESET_ALL}")
                if epsilon is not None:
                    stats_lines.append(f"Epsilon: {epsilon:.3f}")
            elif action_source == 'random':
                # Legacy format - agent exploring randomly
                source_color = Fore.YELLOW
                stats_lines.append(f"{source_color}Source: AGENT (random exploration, Player {current_player}){Style.RESET_ALL}")
                if epsilon is not None:
                    stats_lines.append(f"Epsilon: {epsilon:.3f}")
            elif action_source == 'random_opening':
                source_color = Fore.MAGENTA
                stats_lines.append(f"{source_color}Source: Random opening move (Player {current_player}){Style.RESET_ALL}")
            else:
                stats_lines.append(f"Source: {action_source} (Player {current_player})")
            
            # Add curriculum stage info if available
            if curriculum_stage is not None:
                stats_lines.append(f"{Fore.WHITE}Curriculum: Stage {curriculum_stage} - {curriculum_stage_name}{Style.RESET_ALL}")
            
            # Add old-style detailed stats if available
            if 'q_values' in move and move['q_values']:
                q_values = move['q_values']
                chosen_action = move['column']
                q_values_str = '[' + ', '.join(
                    f"{Fore.GREEN}{q:.3f}{Style.RESET_ALL}" if idx == chosen_action else f"{q:.3f}"
                    for idx, q in enumerate(q_values)
                ) + ']'
                stats_lines.append(f"Q-Values: {q_values_str}")
            
            if 'reward' in move:
                reward = move['reward']
                reward_color = Fore.GREEN if reward > 0 else Fore.RED if reward < 0 else Fore.WHITE
                stats_lines.append(f"{reward_color}Reward: {reward:.3f}{Style.RESET_ALL}")
            
            if 'loss' in move and move['loss'] is not None:
                loss = move['loss']
                loss_color = Fore.RED if loss > 1.0 else Fore.WHITE
                stats_lines.append(f"{loss_color}Loss: {loss:.3f}{Style.RESET_ALL}")
                
        # Old format with all detailed fields
        elif isinstance(move, dict) and all(key in move for key in ['reward', 'epsilon', 'q_values', 'loss', 'action_source']):
            # Reward with color and width-based wrapping
            reward = move['reward']
            reward_color = Fore.GREEN if reward > 0 else Fore.RED if reward < 0 else Fore.WHITE
            components = move.get('reward_components', {})
            components_str = ', '.join(
                f"{k}: {v:.3f}" for k, v in components.items() if v != 0
            )
            reward_prefix = f"{reward_color}Reward: {reward:.3f} ("
            reward_line = f"{reward_prefix}{components_str}){Style.RESET_ALL}"
            
            # Check if reward line needs wrapping (threshold: 50 characters)
            max_reward_width = 95
            if len(reward_line) > max_reward_width:
                # Split components based on width
                comp_list = components_str.split(', ')
                current_len = len(reward_prefix)
                split_idx = 0
                for j, comp in enumerate(comp_list):
                    comp_len = len(comp) + (2 if j > 0 else 0)  # +2 for ", "
                    if current_len + comp_len > max_reward_width - 1:  # -1 for closing parenthesis
                        break
                    current_len += comp_len
                    split_idx = j + 1
                
                if split_idx == 0:  # Ensure at least one component on first line if possible
                    split_idx = 1
                
                # Format two lines
                first_half = ', '.join(comp_list[:split_idx])
                second_half = ', '.join(comp_list[split_idx:])
                stats_lines.append(
                    f"{reward_color}Reward: {reward:.3f} ({first_half}){Style.RESET_ALL}"
                )
                if second_half:  # Only add second line if there are remaining components
                    stats_lines.append(
                        f"{reward_color}{' ' * len('Reward: ')}({second_half}){Style.RESET_ALL}"
                    )
            else:
                stats_lines.append(reward_line)
            
            # Epsilon with color
            epsilon = move['epsilon']
            epsilon_color = Fore.YELLOW if epsilon > 0.5 else Fore.BLUE
            stats_lines.append(
                f"{epsilon_color}Epsilon: {epsilon:.3f}{Style.RESET_ALL}"
            )
            
            # Q-Values with chosen action highlighted
            q_values = move['q_values']
            chosen_action = move['column']
            q_values_str = '[' + ', '.join(
                f"{Fore.GREEN}{q:.3f}{Style.RESET_ALL}" if idx == chosen_action else f"{q:.3f}"
                for idx, q in enumerate(q_values)
            ) + ']'
            stats_lines.append(f"Q-Values: {q_values_str} (Chosen: {q_values[chosen_action]:.3f})")
            
            # Loss with color
            loss = move['loss']
            if loss is not None:
                loss_color = Fore.RED if loss > 1.0 else Fore.WHITE
                stats_lines.append(f"{loss_color}Loss: {loss:.3f}{Style.RESET_ALL}")
            else:
                stats_lines.append(f"{Fore.WHITE}Loss: N/A{Style.RESET_ALL}")
            
            # Action Source with color
            action_source = move['action_source']
            action_color = Fore.GREEN if action_source == 'model' else Fore.RED
            stats_lines.append(f"{action_color}Action: {action_source.capitalize()}{Style.RESET_ALL}")
        else:
            # Fallback for minimal data
            stats_lines = [f"{Fore.WHITE}(No detailed stats available){Style.RESET_ALL}"]
        
        # Check terminal width
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80
        
        max_board_width = max(len(line) for line in board_lines)
        max_stats_width = max(len(line) for line in stats_lines + [move_info])
        combined_width = max_board_width + max_stats_width + 2  # 2 for padding
        
        # Make move
        column_played = move['column'] if isinstance(move, dict) else move
        game.make_move(column_played)
        board_rendered = game.render()
        
        # Highlight the last played piece
        board_rendered = highlight_last_move(board_rendered, column_played, current_player)
        board_lines = board_rendered.split('\n')
        
        # Display board and stats
        if combined_width <= terminal_width:
            # Side-by-side layout
            print(f"\n{move_info}")
            for j, board_line in enumerate(board_lines):
                stats_line = stats_lines[j] if j < len(stats_lines) else ''
                print(f"{board_line:<{max_board_width}}  {stats_line}")
        else:
            # Below-board layout
            print(f"\n{move_info}")
            for line in board_lines:
                print(line)
            for line in stats_lines:
                print(line)
        
        time.sleep(delay)
    
    winner = game.get_winner()
    if winner:
        winner_str = "X" if winner == Player.ONE else "O"
        print(f"\nGame over! {winner_str} wins!")
    else:
        print("\nGame over! It's a draw!")
    
    # Optionally show episode metrics again
    if episode_log:
        print("\nEpisode Metrics:")
        print(f"Reward: {episode_log.get('reward', 'N/A')}")
        print(f"Epsilon: {episode_log.get('epsilon', 'N/A')}")
        print(f"Loss: {episode_log.get('loss', 'N/A')}")

def handle_game_command(args):
    """Handle the 'game' component commands."""
    cli = SimpleCLI()
    sys.argv = ['cli.py', args.command]
    
    if args.debug:
        sys.argv.append('--debug')
    if args.command == 'test' and args.position:
        sys.argv.extend(['--position', args.position])
    if args.command == 'benchmark' and args.iterations:
        sys.argv.extend(['--iterations', str(args.iterations)])
    if args.command == 'play' and args.ai:
        sys.argv.extend(['--ai', args.ai])
    if args.command == 'play' and hasattr(args, 'depth') and args.depth:
        sys.argv.extend(['--depth', str(args.depth)])
    
    cli.run()

def handle_ai_init(args):
    """Handle the 'ai init' command."""
    configure_debug(args)
    agent = create_or_load_agent(args)
    ensure_directory('models')
    model_path = os.path.join('models', 'initial_model')
    agent.save(model_path)
    print(f"Model saved to {model_path}")

def handle_ai_play(args):
    """Handle the 'ai play' command."""
    from connect4.game.rules import ConnectFourGame
    from connect4.utils import Player
    
    configure_debug(args)
    if not args.model:
        print("Error: --model parameter required for play command")
        return
    
    agent = create_or_load_agent(args, args.model)
    game = ConnectFourGame()
    print("Starting game against AI...")
    print(game.render())
    
    while not game.is_game_over():
        current_player = game.get_current_player()
        if current_player == Player.ONE:
            valid_moves = game.get_valid_moves()
            print(f"Valid moves: {valid_moves}")
            while True:
                try:
                    move = int(input("Enter column (0-6): "))
                    if move in valid_moves:
                        break
                    print(f"Invalid move. Choose from: {valid_moves}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            move = agent.get_action(game.board, valid_moves=game.get_valid_moves(), training=False)
            print(f"AI plays column {move}")
        
        game.make_move(move)
        print(game.render())
    
    winner = game.get_winner()
    if winner == Player.ONE:
        print("Congratulations! You won!")
    elif winner == Player.TWO:
        print("AI wins! Better luck next time.")
    else:
        print("It's a draw!")

def handle_ai_train(args):
    """Handle the 'ai train' command."""
    from connect4.ai.trainer import Trainer, OpponentType, CurriculumTrainer
    
    configure_debug(args)
    
    # Load existing agent or create new one
    agent = None
    if args.model:
        agent = create_or_load_agent(args, args.model)
    else:
        agent = create_or_load_agent(args)
    
    ensure_directory('models')
    ensure_directory('data')
    ensure_directory(os.path.join('data', 'games'))
    ensure_directory(os.path.join('data', 'logs'))
    
    # Handle curriculum training separately
    if args.opponent == 'curriculum':
        print("Starting CURRICULUM training...")
        print("This will progressively train against harder opponents.")
        if args.model:
            print(f"Continuing from model: {args.model}")
        else:
            print("Starting with new model")
        print()
        
        log_interval = args.log_interval if args.log_interval else 50
        
        curriculum_trainer = CurriculumTrainer(
            agent=agent,
        )
        
        stats = curriculum_trainer.train(
            log_interval=log_interval,
        )
        
        print("\nCurriculum Training Complete!")
        print("-" * 40)
        print(f"Total episodes: {stats['total_episodes']}")
        print(f"Total time: {stats['total_time']:.0f}s ({stats['total_time']/60:.1f} min)")
        print(f"Final model: {stats['final_model_path']}")
        return
    
    # Standard training (non-curriculum)
    opponent_type = OpponentType.MINIMAX  # Default
    if args.opponent == 'self':
        opponent_type = OpponentType.SELF
    elif args.opponent == 'minimax':
        opponent_type = OpponentType.MINIMAX
    elif args.opponent == 'random':
        opponent_type = OpponentType.RANDOM
    elif args.opponent == 'model':
        opponent_type = OpponentType.MODEL
        if not args.opponent_model:
            print("Error: --opponent_model required when using --opponent model")
            return
    
    log_interval = args.log_interval if args.log_interval else max(1, args.episodes // 100)
    
    # Get minimax depth
    minimax_depth = getattr(args, 'depth', 5)
    
    print(f"Starting training for {args.episodes} episodes...")
    print(f"Opponent: {args.opponent}")
    if args.opponent == 'minimax':
        print(f"Minimax depth: {minimax_depth}")
    if args.model:
        print(f"Continuing from model: {args.model}")
    else:
        print("Starting with new model")
    
    # Create trainer
    trainer = Trainer(
        agent=agent,
        opponent_type=opponent_type,
        opponent_model_path=args.opponent_model if args.opponent == 'model' else None,
        minimax_depth=minimax_depth,
    )
    
    # Run training
    stats = trainer.train(
        episodes=args.episodes,
        log_interval=log_interval,
        save_interval=max(100, args.episodes // 10),
        eval_interval=max(50, args.episodes // 20),
    )
    
    print("\nTraining Complete!")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def handle_ai_jobs(args):
    """Handle the 'ai jobs' command."""
    from connect4.data.data_manager import get_all_jobs, get_job_info, get_episode_logs
    
    configure_debug(args)
    
    if args.job_id is not None:
        job = get_job_info(args.job_id)
        if job:
            print(f"\nJob {args.job_id} Details:")
            print("-" * 40)
            for key, value in job.items():
                print(f"{key}: {value}")
            
            logs = get_episode_logs(args.job_id)
            if logs:
                print(f"\nTraining Progress ({len(logs)} logged episodes):")
                print("-" * 40)
                
                if len(logs) <= 10:
                    for log in logs:
                        print(f"Episode {log['episode']}: "
                              f"Reward={log.get('reward', 'N/A'):.2f}, "
                              f"Epsilon={log.get('epsilon', 'N/A'):.3f}")
                else:
                    print("First 5 episodes:")
                    for log in logs[:5]:
                        print(f"  Episode {log['episode']}: "
                              f"Reward={log.get('reward', 'N/A'):.2f}, "
                              f"Epsilon={log.get('epsilon', 'N/A'):.3f}")
                    print("...")
                    print("Last 5 episodes:")
                    for log in logs[-5:]:
                        print(f"  Episode {log['episode']}: "
                              f"Reward={log.get('reward', 'N/A'):.2f}, "
                              f"Epsilon={log.get('epsilon', 'N/A'):.3f}")
        else:
            print(f"Job {args.job_id} not found")
    else:
        jobs = get_all_jobs()
        if jobs:
            print("\nTraining Jobs:")
            print("-" * 60)
            print(f"{'ID':<5} {'Status':<12} {'Episodes':<10} {'Started':<20}")
            print("-" * 60)
            for job in jobs:
                print(f"{job['id']:<5} {job['status']:<12} {job['episodes']:<10} {job['start_time']:<20}")
        else:
            print("No training jobs found")

def handle_ai_purge(args):
    """Handle the 'ai purge' command."""
    from connect4.data.data_manager import purge_all_data
    
    if not args.confirm:
        print("Warning: This will delete all training data, logs, and saved games.")
        print("Run with --confirm to proceed.")
        return
    
    purge_all_data()
    print("All data purged successfully")

def handle_ai_games(args):
    """Handle the 'ai games' command."""
    from connect4.data.data_manager import get_saved_games
    
    configure_debug(args)
    
    if args.watch_latest:
        watch_latest_game(args.delay)
    elif args.list:
        games = get_saved_games(args.job_id)
        if games:
            print("\nSaved Games:")
            print("-" * 70)
            print(f"{'ID':<5} {'Job':<5} {'Episode':<10} {'Winner':<10} {'Length':<10}")
            print("-" * 70)
            for i, game in enumerate(games):
                winner = game.get('winner', 'Draw') or 'Draw'
                print(f"{i:<5} {game['job_id']:<5} {game['episode']:<10} {winner:<10} {game['game_length']:<10}")
        else:
            print("No saved games found")
    elif args.replay is not None:
        games = get_saved_games(args.job_id)
        if games and 0 <= args.replay < len(games):
            game_data = games[args.replay]
            replay_game(game_data, args.delay, job_id=game_data.get('job_id'))
        else:
            print(f"Game {args.replay} not found")
    else:
        print("Use --list to see saved games or --replay <id> to replay a game")

def watch_latest_game(delay):
    """Continuously watch and replay the latest game as it becomes available."""
    from connect4.data.data_manager import get_saved_games, get_job_info
    
    print("Watching for latest games... Press Ctrl+C to stop.")
    last_game_id = None
    last_job_id = None
    waiting_printed = False
    
    try:
        while True:
            games = get_saved_games()
            if games:
                latest_game = games[-1]
                game_id = (latest_game['job_id'], latest_game['episode'])
                
                if game_id != last_game_id:
                    waiting_printed = False  # Reset waiting flag
                    print("\n" + "=" * 50)
                    print("New game detected!")
                    
                    # Show job info if job changed (helps track curriculum progress)
                    current_job_id = latest_game.get('job_id')
                    if current_job_id != last_job_id and current_job_id is not None:
                        job_info = get_job_info(current_job_id)
                        if job_info and 'params' in job_info:
                            params = job_info['params']
                            opponent = params.get('opponent_type', 'unknown')
                            depth = params.get('minimax_depth')
                            depth_str = f" (depth {depth})" if depth else ""
                            print(f"Job {current_job_id}: Training vs {opponent}{depth_str}")
                        last_job_id = current_job_id
                    
                    replay_game(latest_game, delay, job_id=latest_game.get('job_id'))
                    last_game_id = game_id
                    print("\nWaiting for next game...", end="", flush=True)
                    waiting_printed = True
                else:
                    # Still waiting for a new game - show a dot to indicate activity
                    if waiting_printed:
                        print(".", end="", flush=True)
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped watching.")

def handle_ai_command(args):
    """Handle the 'ai' component commands."""
    if args.command == 'init':
        handle_ai_init(args)
    elif args.command == 'play':
        handle_ai_play(args)
    elif args.command == 'train':
        handle_ai_train(args)
    elif args.command == 'jobs':
        handle_ai_jobs(args)
    elif args.command == 'purge':
        handle_ai_purge(args)
    elif args.command == 'games':
        handle_ai_games(args)
    else:
        print(f"Unknown AI command: {args.command}")

def main():
    """Main entry point for the Connect Four AI Learning System."""
    parser = argparse.ArgumentParser(
        description='Connect Four AI Learning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:

    GAME COMPONENT:
    ---------------
    # Play Connect Four against a random AI
    python run.py game play
    
    # Play Connect Four against Minimax AI (depth 6)
    python run.py game play --ai minimax
    
    # Play against Minimax with custom depth (stronger but slower)
    python run.py game play --ai minimax --depth 8
    
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

    # Initialize a model with 3 layers with a decreasing size
    python run.py ai init --layer_sizes 256,128,64
    
    # Play against a trained model
    python run.py ai play --model models/final_model_TIMESTAMP
    
    # Play with debug information shown
    python run.py ai play --model models/final_model_TIMESTAMP --debug_level info

    AI COMPONENT - TRAINING:
    ------------------------
    # Train against minimax opponent (default)
    python run.py ai train --opponent minimax --episodes 5000
    
    # Train against minimax with specific depth (higher = harder opponent)
    python run.py ai train --opponent minimax --depth 6 --episodes 5000
    
    # Train using self-play (agent plays against itself)
    python run.py ai train --opponent self --episodes 5000
    
    # Train against random opponent (for baseline/testing)
    python run.py ai train --opponent random --episodes 1000
    
    # Train against a saved model
    python run.py ai train --opponent model --opponent_model models/final_model_TIMESTAMP --episodes 5000
    
    # Continue training an existing model against minimax
    python run.py ai train --opponent minimax --model models/existing_model --episodes 5000
    
    # Train with custom neural network architecture (3 layers)
    python run.py ai train --opponent minimax --layer_sizes 256,128,64 --episodes 5000
    
    # Train with 2-layer architecture
    python run.py ai train --layer_sizes 256,128 --episodes 5000
    
    # Train with a specific log interval
    python run.py ai train --opponent minimax --episodes 500 --log_interval 10
    
    # Train with detailed logging
    python run.py ai train --episodes 100 --debug_level debug

    AI COMPONENT - CURRICULUM TRAINING (RECOMMENDED FOR NEW MODELS):
    ----------------------------------------------------------------
    # Use curriculum training to progressively train against harder opponents
    # This is RECOMMENDED for training new models from scratch
    # The agent learns to win against easy opponents first, then graduates
    # to harder ones. This prevents the "learned helplessness" problem.
    
    # Start curriculum training with default settings (recommended)
    python run.py ai train --opponent curriculum
    
    # Curriculum training with custom architecture
    python run.py ai train --opponent curriculum --layer_sizes 256,128,64
    
    # Continue curriculum training from an existing model
    python run.py ai train --opponent curriculum --model models/existing_model
    
    # Curriculum stages (default - with mixed transition stages):
    #   1. Random opponent                    - 1000 episodes, promote at 75%
    #   2. Mixed (70% Random, 30% Minimax D1) - 1500 episodes, promote at 65%
    #   3. Mixed (50% Random, 50% Minimax D1) - 1500 episodes, promote at 55%
    #   4. Mixed (30% Random, 70% Minimax D1) - 1500 episodes, promote at 50%
    #   5. Minimax depth 1                    - 2000 episodes, promote at 45%
    #   6. Minimax depth 2                    - 2500 episodes, promote at 35%
    #   7. Minimax depth 3                    - 3000 episodes, promote at 25%
    #   8. Minimax depth 4                    - 4000 episodes, promote at 15%
    #   9. Minimax depth 5                    - 5000 episodes, promote at 10%
    # Total: ~22,000 episodes across all stages (smoother progression)

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

    # Replay the latest game as it becomes available
    python run.py ai games --watch-latest

    # Watch the latest game with a delay of 1 second
    python run.py ai games --watch-latest --delay 1.0
    
    # Watch the latest game with a delay of 0.5 seconds
    python run.py ai games --watch-latest --delay 0.5
    """
    )

    subparsers = parser.add_subparsers(dest='component', help='Component to run')
    
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
        choices=['random', 'minimax', 'none'], 
        default='random',
        help='AI opponent type: random (makes random moves), minimax (strategic AI), none (two human players)')
    game_parser.add_argument('--depth',
        type=int,
        default=6,
        help='Search depth for minimax AI (default: 6, higher = stronger but slower)')

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
    ai_parser.add_argument('--debug', 
        action='store_true', 
        help='Enable debug mode (equivalent to --debug_level debug)')
    ai_parser.add_argument('--debug_level', 
        choices=['none', 'error', 'warning', 'info', 'debug', 'trace'],
        default='error', 
        help='Set debug level: none (silent), error, warning, info, debug, trace (most verbose)')
    model_group = ai_parser.add_argument_group('Model options')
    model_group.add_argument('--hidden_size', 
        type=int, 
        default=128, 
        help='Size of each hidden layer for neural network (default: 128)')
    model_group.add_argument('--layers', 
        type=int, 
        default=1, 
        help='Number of hidden layers in the neural network (default: 1)')
    model_group.add_argument('--layer_sizes', 
        type=str, 
        help='Comma-separated list of layer sizes (e.g., "256,128,64"), overrides --hidden_size and --layers')
    model_group.add_argument('--model', 
        type=str, 
        help='Path to saved model file for loading (used with play and train commands)')
    training_group = ai_parser.add_argument_group('Training options')
    training_group.add_argument('--episodes', 
        type=int, 
        default=1000,
        help='Number of episodes for training (more = better learning but longer time)')
    training_group.add_argument('--log_interval', 
        type=int, 
        default=None,
        help='Interval between training log updates (default: episodes/100)')
    training_group.add_argument('--opponent',
        choices=['minimax', 'self', 'random', 'model', 'curriculum'],
        default='minimax',
        help='Opponent type for training: minimax (algorithmic), self (self-play), random, model (saved model), curriculum (progressive difficulty)')
    training_group.add_argument('--opponent_model',
        type=str,
        default=None,
        help='Path to opponent model when using --opponent model')
    training_group.add_argument('--depth',
        type=int,
        default=5,
        help='Minimax search depth when using --opponent minimax (default: 5)')
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
    data_group.add_argument('--watch-latest', 
        action='store_true', 
        help='Continuously watch and replay the latest game as it becomes available')

    args = parser.parse_args()
    if args.component == 'game':
        handle_game_command(args)
    elif args.component == 'ai':
        handle_ai_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()