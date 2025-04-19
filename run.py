#!/usr/bin/env python3
"""
run.py - Main entry point for Connect Four AI Learning System
"""

import sys
import os
import argparse

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
    
    # TODO: Add parsers for AI and Web components in the future
    
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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()