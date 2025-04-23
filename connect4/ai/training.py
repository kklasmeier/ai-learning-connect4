"""
training.py - Self-play training framework for Connect Four AI

This module implements the training process for the Connect Four AI agent
using reinforcement learning through self-play.
"""

import os
import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

from connect4.debug import debug, DebugLevel
from connect4.utils import ROWS, COLS, Player, GameResult, is_valid_position
from connect4.game.rules import ConnectFourGame, ConnectFourEnv
from connect4.ai.dqn import DQNAgent, DQNModel
from connect4.ai.replay_buffer import ReplayBuffer
from connect4.ai.utils import board_to_state, state_to_tensor, check_connections
from connect4.ai.reward_calculator import ConnectFourRewardCalculator

from connect4.data.data_manager import (
    create_job, update_job_progress, add_episode_log,
    register_model, complete_job
)

class TrainingStats:
    """Track and save statistics during training."""
    
    def __init__(self, save_dir: str = 'stats'):
        """
        Initialize training statistics tracker.
        
        Args:
            save_dir: Directory to save statistics
        """
        self.save_dir = save_dir
        self.reset()
        
        # Create stats directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def reset(self):
        """Reset all statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = {'player_one': [], 'player_two': [], 'draw': []}
        self.losses = []
        self.exploration_rates = []
        self.timestamps = []
    
    def add_episode(self, reward: float, length: int, winner: Optional[Player], 
                loss: float, epsilon: float):
        """
        Add statistics for a completed episode.
        
        Args:
            reward: Final reward from the episode
            length: Number of moves in the episode
            winner: Winning player (None for draw)
            loss: Training loss value
            epsilon: Current exploration rate
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Track winner
        if winner is None:
            self.win_rates['draw'].append(1)
            self.win_rates['player_one'].append(0)
            self.win_rates['player_two'].append(0)
            winner_str = None
        elif winner == Player.ONE:
            self.win_rates['player_one'].append(1)
            self.win_rates['player_two'].append(0)
            self.win_rates['draw'].append(0)
            winner_str = "X"
        else:  # Player.TWO
            self.win_rates['player_two'].append(1)
            self.win_rates['player_one'].append(0)
            self.win_rates['draw'].append(0)
            winner_str = "O"
        
        self.losses.append(loss)
        self.exploration_rates.append(epsilon)
        self.timestamps.append(time.time())
        
        # Prepare episode data for data manager
        episode_data = {
            'episode': len(self.episode_rewards),
            'reward': float(reward),
            'length': int(length),
            'winner': winner_str,
            'p1_win_rate': float(np.mean(self.win_rates['player_one'][-100:])),
            'p2_win_rate': float(np.mean(self.win_rates['player_two'][-100:])),
            'draw_rate': float(np.mean(self.win_rates['draw'][-100:])),
            'epsilon': float(epsilon),
            'loss': float(loss) if loss is not None else None
        }
        
        # Add to data manager if job_id is available
        if hasattr(self, 'job_id'):
            add_episode_log(self.job_id, episode_data)

    def get_summary(self, window: int = 100) -> Dict[str, Any]:
        """
        Get a summary of recent statistics.
        
        Args:
            window: Number of recent episodes to include
            
        Returns:
            Dictionary with summarized statistics
        """
        if not self.episode_rewards:
            return {
                'avg_reward': 0,
                'avg_length': 0,
                'win_rate_p1': 0,
                'win_rate_p2': 0,
                'draw_rate': 0,
                'avg_loss': 0,
                'current_epsilon': 0
            }
        
        # Limit window to available data
        window = min(window, len(self.episode_rewards))
        
        # Calculate statistics over the window
        return {
            'avg_reward': np.mean(self.episode_rewards[-window:]),
            'avg_length': np.mean(self.episode_lengths[-window:]),
            'win_rate_p1': np.mean(self.win_rates['player_one'][-window:]),
            'win_rate_p2': np.mean(self.win_rates['player_two'][-window:]),
            'draw_rate': np.mean(self.win_rates['draw'][-window:]),
            'avg_loss': np.mean([l for l in self.losses[-window:] if l is not None]),
            'current_epsilon': self.exploration_rates[-1] if self.exploration_rates else 0
        }
    
    def save(self, filename: str = None):
        """
        Save statistics to a file.
        
        Args:
            filename: Base filename (timestamp will be added)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_stats_{timestamp}.json"
        
        path = os.path.join(self.save_dir, filename)
        
        # Prepare data for serialization
        data = {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'win_rates': self.win_rates,
            'losses': [float(l) if l is not None else None for l in self.losses],
            'exploration': self.exploration_rates,
            'timestamps': self.timestamps
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        debug.info(f"Saved training statistics to {path}", "training")
        
        return path
class SelfPlayTrainer:
    """
    Train Connect Four AI through self-play.
    
    This class manages the training process where the AI agent plays against
    itself to improve its policy through reinforcement learning.
    """

    def __init__(self, agent: Optional[DQNAgent] = None, 
                model_dir: str = 'models', 
                batch_size: int = 64,
                replay_buffer_size: int = 10000,
                gamma: float = 0.99,
                target_update_freq: int = 10):
        """
        Initialize the self-play trainer.
        
        Args:
            agent: Pre-initialized DQN agent (creates new if None)
            model_dir: Directory to save models
            batch_size: Batch size for training
            replay_buffer_size: Size of replay buffer
            gamma: Discount factor for future rewards
            target_update_freq: How often to update target network
        """
        debug.debug("Initializing SelfPlayTrainer", "training")

        # Initialize total_episodes
        self.total_episodes = 0

        # Create directory for models
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Create or use provided agent
        self.agent = agent if agent is not None else DQNAgent(
            gamma=gamma, target_update_freq=target_update_freq
        )
        
        # Replay buffer for experience
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        # Training parameters
        self.batch_size = batch_size
        
        # Statistics tracking
        self.stats = TrainingStats()
        self.episode_count = 0
        
        # Timestamp for saving models
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create job entry for tracking
        training_params = {
            'episodes': 0,  # Will be updated in train() method
            'layers': len(agent.model.hidden_sizes) if hasattr(agent.model, 'hidden_sizes') else 1,
            'layer_sizes': agent.model.hidden_sizes if hasattr(agent.model, 'hidden_sizes') else [agent.model.fc1.out_features],
            'batch_size': batch_size,
            'gamma': gamma,
            'target_update_freq': target_update_freq,
            'model_dir': model_dir
        }
        self.job_id = create_job(training_params)
        debug.info(f"Created training job with ID: {self.job_id}", "training")    

    def _play_episode(self, training: bool = True) -> Tuple[float, int, Optional[Player], List[float]]:
        """Play a complete self-play episode."""
        game = ConnectFourGame()
        done = False
        total_reward = 0
        episode_length = 0
        losses = []
        
        # Keep track of states and actions for rewarding after game completion
        history = []
        
        # Track moves and stats for saving
        move_data = []
        
        # Track the last column played (to detect stacking)
        last_column = -1
        
        # Initialize stacking counter
        stacking_count = 0
        
        # Initialize reward calculator
        reward_calculator = ConnectFourRewardCalculator()
        
        while not done:
            current_player = game.get_current_player()
            
            # Get board state
            state = board_to_state(game.board.grid)
            
            # Get valid moves
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                # Game might be over (draw)
                break
            
            # Get action from agent and track whether it was random
            state_tensor = state_to_tensor(state)
            is_random = False
            with torch.no_grad():
                q_values = self.agent.model(state_tensor).squeeze(0).tolist()
            if training and random.random() < self.agent.epsilon:
                action = random.choice(valid_moves)
                is_random = True
                debug.trace(f"Random exploration action: {action}", "ai")
            else:
                action = self.agent.get_action(game.board, valid_moves, training=training)
                debug.trace(f"Model-based action: {action}, Q-values: {q_values}", "ai")
            
            # Update stacking counter
            if last_column == action:
                stacking_count += 1
            else:
                stacking_count = 1
            
            # Update last column
            last_column = action
            
            # Make the move
            game.make_move(action)
            episode_length += 1
            
            # Get the position of the last move
            last_row, last_col = game.board.last_move
            
            # Calculate reward and components
            reward, reward_components = reward_calculator.calculate_reward(
                game, last_row, last_col, last_column, action, 
                current_player, stacking_count, valid_moves
            )
            
            # Store total reward (from player 1's perspective)
            if current_player == Player.ONE:
                total_reward += reward
            else:
                total_reward -= reward  # Negate reward for player 2
            
            # Get next state
            next_state = board_to_state(game.board.grid)
            
            # Store the transition and stats
            move_stats = {
                'column': action,
                'player': 'X' if current_player == Player.ONE else 'O',
                'reward': reward,
                'reward_components': reward_components,
                'epsilon': self.agent.epsilon,
                'q_values': [round(q, 3) for q in q_values],
                'loss': None,
                'action_source': 'random' if is_random else 'model'
            }
            move_data.append(move_stats)
            
            if training:
                history.append((state, action, reward, next_state, done, current_player))
            
            # If game is not over and we're training, train on a batch from replay buffer
            if training and not done and len(self.replay_buffer) > self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.agent.train(batch)
                losses.append(loss)
                # Update loss for the last move (approximate, as loss is batch-based)
                move_data[-1]['loss'] = loss
        
        # Process the history to add experiences to replay buffer
        if training:
            # If the game is over, process experiences
            for i, (state, action, reward, next_state, is_done, player) in enumerate(history):
                # For final state, use the outcome reward
                if i == len(history) - 1:
                    self.replay_buffer.add(state, action, reward, next_state, True)
                else:
                    self.replay_buffer.add(state, action, reward, next_state, False)
        
        # Determine whether to save this game
        winner = game.get_winner()
        winner_str = "X" if winner == Player.ONE else "O" if winner == Player.TWO else None
        
        should_save_game = (
            self.episode_count == 0 or                      # First game
            self.episode_count == self.total_episodes - 1 or # Last game
            self.episode_count % 50 == 0 or                 # Regular milestone
            random.random() < 0.05 or                       # Random 5% sample
            episode_length > 30 or                          # Long games
            episode_length < 15                             # Short games
        )
        
        if should_save_game:
            from connect4.data.data_manager import save_game_moves
            save_game_moves(self.job_id, self.episode_count + 1, move_data, winner_str, episode_length)
            debug.info(f"Saved game moves for episode {self.episode_count + 1}", "training")
        
        return total_reward, episode_length, winner, losses

    def train(self, episodes: int = 1000, 
            log_interval: int = 10, 
            save_interval: int = 100,
            evaluation_interval: int = 50):
        """
        Train the agent through self-play.
        
        Args:
            episodes: Number of episodes to train for
            log_interval: How often to log progress
            save_interval: How often to save models
            evaluation_interval: How often to evaluate against a fixed opponent
        """
        debug.info(f"Starting self-play training for {episodes} episodes", "training")
        
        start_time = time.time()
        
        # Update the job with total episodes
        job_params = {
            'episodes': episodes,
            'log_interval': log_interval, 
            'save_interval': save_interval,
            'evaluation_interval': evaluation_interval
        }
        update_job_progress(self.job_id, 0)  # Reset progress
        
        # Add job_id to stats for episode logging
        self.stats.job_id = self.job_id
        self.total_episodes = episodes
        
        # Add tracking for dangerous loss values
        max_loss_threshold = 1000.0  # Set a reasonable threshold
        high_loss_count = 0
        
        for episode in range(1, episodes + 1):
            debug.debug(f"Starting episode {episode}/{episodes}", "training")
            
            # Reset epsilon periodically to encourage exploration
            if episode % 5000 == 0 and episode > 0:
                old_epsilon = self.agent.epsilon
                self.agent.epsilon = max(0.3, self.agent.epsilon)  # Reset to at least 30% exploration
                debug.info(f"Reset epsilon from {old_epsilon:.3f} to {self.agent.epsilon:.3f} to encourage exploration", "training")
            
            # Play a self-play episode and train
            total_reward, episode_length, winner, losses = self._play_episode(training=True)
            
            # Track statistics
            avg_loss = np.mean(losses) if losses else None
            self.stats.add_episode(
                total_reward, episode_length, winner, avg_loss, self.agent.epsilon
            )

            # Monitor for high loss values
            if losses and avg_loss is not None and avg_loss > max_loss_threshold:
                high_loss_count += 1
                debug.warning(f"High loss detected: {avg_loss}, count: {high_loss_count}", "training")
                
                # If loss values stay high, take corrective action
                if high_loss_count > 10:
                    debug.warning("Persistent high losses! Temporarily reducing learning rate", "training")
                    current_lr = self.agent.optimizer.param_groups[0]['lr']
                    reduced_lr = current_lr * 0.5  # Cut learning rate in half
                    for param_group in self.agent.optimizer.param_groups:
                        param_group['lr'] = max(reduced_lr, 0.00001)  # Don't go too low
                    high_loss_count = 0  # Reset counter
                    
            else:
                high_loss_count = max(0, high_loss_count - 1)  # Decrease counter if loss is normal

            # Update job progress
            update_job_progress(self.job_id, episode)
            
            # Log progress
            if episode % log_interval == 0:
                stats = self.stats.get_summary()
                elapsed = time.time() - start_time
                
                print(f"Episode {episode}/{episodes} "
                    f"[{elapsed:.1f}s] - "
                    f"Reward: {total_reward:.2f}, "
                    f"Length: {episode_length}, "
                    f"Winner: {winner}, "
                    f"P1 Win Rate: {stats['win_rate_p1']:.2f}, "
                    f"P2 Win Rate: {stats['win_rate_p2']:.2f}, "
                    f"Draw Rate: {stats['draw_rate']:.2f}, "
                    f"Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model periodically
            if episode % save_interval == 0:
                self._save_checkpoint(episode)
            
            # Evaluate against a fixed opponent periodically
            if episode % evaluation_interval == 0:
                self._evaluate(5)  # Play 5 evaluation games
            
            self.episode_count += 1
        
        # Save final model and statistics
        self._save_checkpoint(episodes, final=True)
        
        # Mark job as complete
        complete_job(self.job_id)
        
        debug.info(f"Training completed after {episodes} episodes", "training")
        
        # Return final statistics
        return self.stats.get_summary()
   
    def _save_checkpoint(self, episode: int, final: bool = False):
        """
        Save a model checkpoint.
        
        Args:
            episode: Current episode number
            final: Whether this is the final model
        """
        if final:
            model_name = f"final_model_{self.start_time}"
        else:
            model_name = f"model_{self.start_time}_ep{episode}"
        
        model_path = os.path.join(self.model_dir, model_name)
        self.agent.save(model_path)
        
        # Save stats with the same name
        stats_path = self.stats.save(f"{model_name}_stats.json")
        
        # Register model in data manager
        register_model(self.job_id, episode, model_path, is_final=final)
        
        debug.info(f"Saved checkpoint to {model_path}", "training")

    def _evaluate(self, num_games: int = 10):
        """
        Evaluate the current agent by playing against a fixed opponent.
        
        Args:
            num_games: Number of games to play for evaluation
        """
        debug.info(f"Evaluating agent over {num_games} games", "training")
        
        # For now, evaluate against random play
        wins = 0
        draws = 0
        player_one_wins = 0
        player_two_wins = 0
        for game_idx in range(num_games):
            # Alternate playing as player 1 and 2
            # Play without training (evaluation mode)
            total_reward, episode_length, winner, _ = self._play_episode(training=False)
            
            if winner == Player.ONE:
                wins += 1
                player_one_wins += 1
            elif winner == Player.TWO:
                wins += 1  # Count as win for the agent regardless of which player
                player_two_wins += 1
            else:
                draws += 1
        
        p1_win_rate = player_one_wins / num_games
        draw_rate = draws / num_games
        p2_win_rate = player_two_wins / num_games
        
        print(f"P1 Win Rate: {p1_win_rate:.2f}, "
             f"Draw Rate: {draw_rate:.2f}, "
             f"P2 Win Rate: {p2_win_rate:.2f}")