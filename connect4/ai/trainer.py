"""
trainer.py - Unified training framework for Connect Four AI
This module provides a flexible trainer that can train the DQN agent against
different types of opponents: minimax, self-play, or other saved models.
"""
import os
import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from enum import Enum
from connect4.debug import debug, DebugLevel
from connect4.utils import ROWS, COLS, Player, GameResult
from connect4.game.rules import ConnectFourGame
from connect4.game.board import Board
from connect4.ai.dqn import DQNAgent, DQNModel
from connect4.ai.replay_buffer import ReplayBuffer
# --- Curriculum helpers ---
def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Conservative lower bound of a binomial proportion (Wilson score interval)."""
    if trials <= 0:
        return 0.0
    phat = successes / trials
    denom = 1.0 + (z*z)/trials
    center = phat + (z*z)/(2.0*trials)
    margin = z * ((phat*(1.0-phat) + (z*z)/(4.0*trials)) / trials) ** 0.5
    return max(0.0, (center - margin) / denom)
class DualReplayBuffer:
    """A replay buffer wrapper that mixes recent and global experience.
    This is useful in curriculum training: global prevents forgetting, recent
    helps adapt quickly to the current stage distribution.
    """
    def __init__(self, global_buffer: ReplayBuffer, recent_capacity: int = 10000, recent_fraction: float = 0.5):
        self.global_buffer = global_buffer
        self.recent_buffer = ReplayBuffer(capacity=max(1000, int(recent_capacity)))
        self.recent_fraction = float(max(0.0, min(1.0, recent_fraction)))
    def add(self, *args, **kwargs):
        self.global_buffer.add(*args, **kwargs)
        self.recent_buffer.add(*args, **kwargs)
    def sample(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []
        n_recent = int(round(batch_size * self.recent_fraction))
        n_global = batch_size - n_recent
        out = []
        # Guard: if a buffer is too small, fall back to the other.
        if n_recent > 0 and len(self.recent_buffer) >= n_recent:
            out.extend(self.recent_buffer.sample(n_recent))
        else:
            n_global += n_recent
            n_recent = 0
        if n_global > 0 and len(self.global_buffer) >= n_global:
            out.extend(self.global_buffer.sample(n_global))
        elif n_global > 0 and len(self.recent_buffer) >= n_global:
            out.extend(self.recent_buffer.sample(n_global))
        return out
    def __len__(self):
        return len(self.global_buffer)
from connect4.ai.minimax import MinimaxPlayer
from connect4.ai.utils import board_to_state, state_to_tensor
from connect4.data.data_manager import (
    create_job, update_job_progress, add_episode_log,
    register_model, complete_job, save_game_moves
)
class OpponentType(Enum):
    """Types of opponents the agent can train against."""
    MINIMAX = "minimax"
    SELF = "self"
    MODEL = "model"
    RANDOM = "random"
    MIXED = "mixed"  # Mix of random and minimax for curriculum transition
class Trainer:
    """
    Unified trainer for Connect Four AI.
    
    Supports training against multiple opponent types:
    - minimax: Algorithmic opponent using minimax with alpha-beta pruning
    - self: Self-play where agent plays both sides
    - model: Play against a saved model
    - random: Play against random moves (for baseline testing)
    """
    def __init__(
        self,
        agent: Optional[DQNAgent] = None,
        opponent_type: OpponentType = OpponentType.MINIMAX,
        opponent_model_path: Optional[str] = None,
        minimax_depth: int = 5,
        mixed_random_prob: float = 0.5,  # For MIXED type: probability of random vs minimax
        model_dir: str = 'models',
        batch_size: int = 64,
        replay_buffer_size: int = 50000,
        replay_buffer: Optional[ReplayBuffer] = None,
        # Reward settings (simplified)
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_draw: float = 0.0,
        # Epsilon settings - FIXED for proper exploration/exploitation
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 2000,  # Reach epsilon_end after this many episodes
    ):
        """
        Initialize the trainer.
        
        Args:
            agent: Pre-initialized DQN agent (creates new if None)
            opponent_type: Type of opponent to train against
            opponent_model_path: Path to opponent model (for MODEL type)
            minimax_depth: Search depth for minimax opponent
            model_dir: Directory to save models
            batch_size: Batch size for training
            replay_buffer_size: Size of replay buffer
            replay_buffer: Optional shared replay buffer instance (if None, a new buffer is created)
            reward_win: Reward for winning
            reward_loss: Reward for losing
            reward_draw: Reward for draw
            epsilon_start: Starting exploration rate
            epsilon_end: Final exploration rate  
            epsilon_decay_episodes: Episodes over which to decay epsilon
        """
        debug.info(f"Initializing Trainer with opponent: {opponent_type.value}", "training")
        
        # Create or use provided agent
        self.agent = agent if agent is not None else DQNAgent()
        
        # Override agent's epsilon settings with our calculated decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.agent.epsilon = epsilon_start
        
        # Opponent configuration
        self.opponent_type = opponent_type
        self.minimax_depth = minimax_depth
        self.minimax_depth_range = (max(2, minimax_depth - 2), minimax_depth + 1)  # Vary depth
        self.mixed_random_prob = mixed_random_prob  # For MIXED opponent type
        
        # Initialize opponent based on type
        self._init_opponent(opponent_model_path)
        
        # Training infrastructure
        # If a replay buffer is provided (e.g., curriculum training), reuse it to preserve experience
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(capacity=replay_buffer_size)
        self.batch_size = batch_size
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Simplified rewards
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.reward_draw = reward_draw
        
        # Training state
        self.episode_count = 0
        self.total_episodes = 0
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_id = None
        
        # Curriculum tracking (set by CurriculumTrainer if applicable)
        self.curriculum_stage = None  # Current stage index (0-based)
        self.curriculum_total_stages = None  # Total number of stages
        self.curriculum_stage_name = None  # Human-readable stage name
        
        # Statistics
        self.stats = {
            'wins': [],
            'losses': [],
            'draws': [],
            'episode_lengths': [],
            'losses_values': [],
            'epsilons': []
        }
    
    def _init_opponent(self, opponent_model_path: Optional[str] = None):
        """Initialize the opponent based on type."""
        if self.opponent_type == OpponentType.MINIMAX:
            self.opponent = MinimaxPlayer(depth=self.minimax_depth)
            debug.info(f"Initialized Minimax opponent with depth {self.minimax_depth}", "training")
        
        elif self.opponent_type == OpponentType.MODEL:
            if opponent_model_path is None:
                raise ValueError("opponent_model_path required for MODEL opponent type")
            self.opponent = DQNAgent.load(opponent_model_path)
            debug.info(f"Loaded opponent model from {opponent_model_path}", "training")
        
        elif self.opponent_type == OpponentType.SELF:
            self.opponent = None  # Will use self.agent
            debug.info("Using self-play mode", "training")
        
        elif self.opponent_type == OpponentType.RANDOM:
            self.opponent = None  # Will use random selection
            debug.info("Using random opponent", "training")
        
        elif self.opponent_type == OpponentType.MIXED:
            # Mixed uses both random and minimax
            self.opponent = MinimaxPlayer(depth=self.minimax_depth)
            debug.info(f"Initialized Mixed opponent ({self.mixed_random_prob:.0%} random, "
                      f"{1-self.mixed_random_prob:.0%} minimax depth {self.minimax_depth})", "training")
    
    def _get_opponent_move(self, game: ConnectFourGame, valid_moves: List[int]) -> Tuple[int, str]:
        """
        Get a move from the opponent.
        
        Returns:
            Tuple of (action, action_source) where action_source describes
            what type of opponent made the move for replay clarity.
        """
        if self.opponent_type == OpponentType.MINIMAX:
            # Occasionally vary the minimax depth for diversity
            if random.random() < 0.1:  # 10% of moves use different depth
                varied_depth = random.randint(*self.minimax_depth_range)
                temp_player = MinimaxPlayer(depth=varied_depth)
                return temp_player.get_move(game.board), f'opponent_minimax_d{varied_depth}'
            return self.opponent.get_move(game.board), f'opponent_minimax_d{self.minimax_depth}'
        
        elif self.opponent_type == OpponentType.MODEL:
            current_player = game.get_current_player()
            return self.opponent.get_action(
                game.board,
                valid_moves,
                current_player=current_player,
                training=False
            ), 'opponent_model'
        
        elif self.opponent_type == OpponentType.SELF:
            current_player = game.get_current_player()
            return self.agent.get_action(
                game.board,
                valid_moves,
                current_player=current_player,
                training=True
            ), 'opponent_self'
        
        elif self.opponent_type == OpponentType.RANDOM:
            return random.choice(valid_moves), 'opponent_random'
        
        elif self.opponent_type == OpponentType.MIXED:
            # Randomly choose between random and minimax
            if random.random() < self.mixed_random_prob:
                return random.choice(valid_moves), 'opponent_random'
            else:
                return self.opponent.get_move(game.board), f'opponent_minimax_d{self.minimax_depth}'
        
        return random.choice(valid_moves), 'opponent_unknown'
    
    def _update_epsilon(self):
        """Update epsilon based on episode count with linear decay."""
        if self.episode_count < self.epsilon_decay_episodes:
            # Linear decay from start to end over decay_episodes
            progress = self.episode_count / self.epsilon_decay_episodes
            self.agent.epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
        else:
            self.agent.epsilon = self.epsilon_end
    def _play_episode(
        self,
        training: bool = True,
        agent_plays_first: bool = True,
        random_opening_moves: int = 0,
        save_game: bool = False
    ) -> Tuple[float, int, Optional[Player], List[float]]:
        """
        Play a single episode.
        
        Args:
            training: Whether to train during this episode
            agent_plays_first: Whether the learning agent plays as Player ONE
            random_opening_moves: Number of random moves to start with
            save_game: Whether to save this game for replay
            
        Returns:
            Tuple of (total_reward, episode_length, winner, losses)
        """
        game = ConnectFourGame()
        losses = []
        episode_length = 0
        experiences = []  # Store experiences for end-of-episode processing
        move_data = []  # Store move data for game saving
        
        # Determine which player the agent is
        agent_player = Player.ONE if agent_plays_first else Player.TWO
        
        # Apply random opening moves for variety
        for _ in range(random_opening_moves):
            valid_moves = game.get_valid_moves()
            if valid_moves and not game.is_game_over():
                move = random.choice(valid_moves)
                game.make_move(move)
                episode_length += 1
                if save_game:
                    move_data.append({
                        'column': move,
                        'player': 'X' if game.get_current_player() == Player.TWO else 'O',  # Previous player
                        'action_source': 'random_opening'
                    })
        
        # Main game loop
        while not game.is_game_over():
            current_player = game.get_current_player()
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                break
            
            # Include side-to-move in the state representation (critical for learning)
            state = board_to_state(game.board.grid, current_player)
            
            # Determine whose turn it is
            is_agent_turn = (current_player == agent_player)
            
            if is_agent_turn:
                # Agent's turn - check for exploration vs exploitation
                is_random = False
                if training and random.random() < self.agent.epsilon:
                    action = random.choice(valid_moves)
                    is_random = True
                else:
                    action = self.agent.get_action(
                        game.board,
                        valid_moves,
                        current_player=current_player,
                        training=False
                    )
                action_source = 'agent_random' if is_random else 'agent_model'
            else:
                # Opponent's turn - get both action and descriptive source
                action, action_source = self._get_opponent_move(game, valid_moves)
            
            # Make the move
            game.make_move(action)
            episode_length += 1
            
            # Store move data for replay
            if save_game:
                move_info = {
                    'column': action,
                    'player': 'X' if current_player == Player.ONE else 'O',
                    'action_source': action_source,
                    'epsilon': self.agent.epsilon if is_agent_turn else None
                }
                # Add curriculum stage info if available
                if self.curriculum_stage is not None:
                    move_info['curriculum_stage'] = self.curriculum_stage + 1  # 1-indexed
                    move_info['curriculum_stage_name'] = self.curriculum_stage_name
                move_data.append(move_info)
            
            # Store experience for agent's moves only
            if is_agent_turn and training:
                # After making a move, the current player has switched
                next_player = game.get_current_player()
                next_state = board_to_state(game.board.grid, next_player)
                done = game.is_game_over()
                experiences.append((state, action, next_state, done, current_player))
        
        # Calculate final reward based on game outcome
        winner = game.get_winner()
        
        if winner == agent_player:
            final_reward = self.reward_win
            outcome = 'win'
        elif winner is None:
            final_reward = self.reward_draw
            outcome = 'draw'
        else:
            final_reward = self.reward_loss
            outcome = 'loss'
        
        # Process experiences with final reward
        # Assign rewards: final move gets the outcome reward, others get small step penalty
        if training and experiences:
            num_experiences = len(experiences)
            for i, (state, action, next_state, done, player) in enumerate(experiences):
                if i == num_experiences - 1:
                    # Last move gets the final reward
                    reward = final_reward
                else:
                    # Intermediate moves get small penalty to encourage faster wins
                    reward = -0.01
                
                self.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train on batch if buffer is large enough
            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.agent.train(batch)
                losses.append(loss)
        
        # Save game for replay if flagged
        if save_game and self.job_id is not None:
            winner_str = 'X' if winner == Player.ONE else 'O' if winner == Player.TWO else None
            save_game_moves(self.job_id, self.episode_count, move_data, winner_str, episode_length)
        
        # Update statistics
        self.stats['wins'].append(1 if outcome == 'win' else 0)
        self.stats['losses'].append(1 if outcome == 'loss' else 0)
        self.stats['draws'].append(1 if outcome == 'draw' else 0)
        self.stats['episode_lengths'].append(episode_length)
        self.stats['epsilons'].append(self.agent.epsilon)
        if losses:
            self.stats['losses_values'].append(np.mean(losses))
        
        return final_reward, episode_length, winner, losses
    
    def train(
        self,
        episodes: int = 1000,
        log_interval: int = 100,
        save_interval: int = 500,
        eval_interval: int = 250,
        eval_games: int = 20,
        randomize_first_player: bool = True,
        random_opening_probability: float = 0.2,
        max_random_opening_moves: int = 4,
        save_game_interval: int = 50,  # Save a game every N episodes for replay
    ) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            episodes: Number of episodes to train
            log_interval: How often to log progress
            save_interval: How often to save checkpoints
            eval_interval: How often to run evaluation
            eval_games: Number of games for evaluation
            randomize_first_player: Whether to randomize who plays first
            random_opening_probability: Probability of starting with random moves
            max_random_opening_moves: Maximum random opening moves
            save_game_interval: How often to save games for replay
            
        Returns:
            Final training statistics
        """
        debug.info(f"Starting training for {episodes} episodes against {self.opponent_type.value}", "training")
        
        self.total_episodes = episodes
        
        # Adjust epsilon decay to the number of episodes
        # Decay over 40% of total episodes, or epsilon_decay_episodes, whichever is smaller
        self.epsilon_decay_episodes = min(self.epsilon_decay_episodes, int(episodes * 0.4))
        
        start_time = time.time()
        
        # Create job for tracking
        training_params = {
            'episodes': episodes,
            'opponent_type': self.opponent_type.value,
            'minimax_depth': self.minimax_depth if self.opponent_type == OpponentType.MINIMAX else None,
            'batch_size': self.batch_size,
            'model_dir': self.model_dir,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_episodes': self.epsilon_decay_episodes
        }
        self.job_id = create_job(training_params)
        debug.info(f"Created training job with ID: {self.job_id}", "training")
        
        for episode in range(1, episodes + 1):
            self.episode_count = episode
            
            # Update epsilon with linear decay
            self._update_epsilon()
            
            # Randomize who plays first
            agent_plays_first = True
            if randomize_first_player:
                agent_plays_first = random.random() < 0.5
            
            # Occasionally start with random opening moves
            random_opening = 0
            if random.random() < random_opening_probability:
                random_opening = random.randint(1, max_random_opening_moves)
            
            # Decide whether to save this game
            save_this_game = (
                episode == 1 or  # First game
                episode == episodes or  # Last game
                episode % save_game_interval == 0 or  # Regular interval
                random.random() < 0.02  # 2% random chance
            )
            
            # Play episode
            reward, length, winner, losses = self._play_episode(
                training=True,
                agent_plays_first=agent_plays_first,
                random_opening_moves=random_opening,
                save_game=save_this_game
            )
            
            # Update job progress
            update_job_progress(self.job_id, episode)
            
            # Log progress
            if episode % log_interval == 0:
                self._log_progress(episode, episodes, start_time)
            
            # Save checkpoint
            if episode % save_interval == 0:
                self._save_checkpoint(episode)
            
            # Evaluation
            if episode % eval_interval == 0:
                self._evaluate(eval_games)
        
        # Final save
        self._save_checkpoint(episodes, final=True)
        complete_job(self.job_id)
        
        elapsed = time.time() - start_time
        debug.info(f"Training completed in {elapsed:.1f}s", "training")
        
        return self._get_summary()
    
    def _log_progress(self, episode: int, total: int, start_time: float):
        """Log training progress."""
        window = min(100, episode)
        
        win_rate = np.mean(self.stats['wins'][-window:])
        loss_rate = np.mean(self.stats['losses'][-window:])
        draw_rate = np.mean(self.stats['draws'][-window:])
        avg_length = np.mean(self.stats['episode_lengths'][-window:])
        epsilon = self.agent.epsilon
        
        elapsed = time.time() - start_time
        eps_per_sec = episode / elapsed if elapsed > 0 else 0
        
        avg_loss = 0
        if self.stats['losses_values']:
            recent_losses = self.stats['losses_values'][-window:]
            avg_loss = np.mean(recent_losses) if recent_losses else 0
        
        # Build curriculum prefix if in curriculum mode
        curriculum_prefix = ""
        if self.curriculum_stage is not None and self.curriculum_total_stages is not None:
            curriculum_prefix = f"[Stage {self.curriculum_stage + 1}/{self.curriculum_total_stages}] "
        
        print(f"{curriculum_prefix}Episode {episode}/{total} [{elapsed:.0f}s, {eps_per_sec:.1f} ep/s] | "
              f"Win: {win_rate:.1%}, Loss: {loss_rate:.1%}, Draw: {draw_rate:.1%} | "
              f"Len: {avg_length:.1f}, ε: {epsilon:.3f}, Loss: {avg_loss:.4f}")
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save a model checkpoint."""
        if final:
            model_name = f"final_model_{self.start_time}"
        else:
            model_name = f"model_{self.start_time}_ep{episode}"
        
        model_path = os.path.join(self.model_dir, model_name)
        self.agent.save(model_path)
        
        # Register in data manager
        register_model(self.job_id, episode, model_path, is_final=final)
        
        debug.info(f"Saved checkpoint: {model_path}", "training")
    
    def _evaluate(self, num_games: int = 20) -> Dict[str, float]:
        """
        Evaluate the agent against the current opponent.
        
        Args:
            num_games: Number of evaluation games
        """
        wins = 0
        losses = 0
        draws = 0
        
        for i in range(num_games):
            # Alternate who plays first
            agent_first = (i % 2 == 0)
            agent_player = Player.ONE if agent_first else Player.TWO
            
            reward, length, winner, _ = self._play_episode(
                training=False,
                agent_plays_first=agent_first,
                random_opening_moves=0
            )
            
            if winner == agent_player:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1
        
        win_rate = wins / num_games
        draw_rate = draws / num_games
        loss_rate = losses / num_games
        
        print(f"  [EVAL] {num_games} games: Win {win_rate:.1%}, Draw {draw_rate:.1%}, Loss {loss_rate:.1%}")
        return {
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate,
        }
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        window = min(100, len(self.stats['wins']))
        
        return {
            'total_episodes': self.episode_count,
            'final_win_rate': np.mean(self.stats['wins'][-window:]) if self.stats['wins'] else 0,
            'final_loss_rate': np.mean(self.stats['losses'][-window:]) if self.stats['losses'] else 0,
            'final_draw_rate': np.mean(self.stats['draws'][-window:]) if self.stats['draws'] else 0,
            'avg_episode_length': np.mean(self.stats['episode_lengths'][-window:]) if self.stats['episode_lengths'] else 0,
            'final_epsilon': self.agent.epsilon
        }
def train_against_minimax(
    agent: Optional[DQNAgent] = None,
    episodes: int = 1000,
    depth: int = 5,
    log_interval: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train against minimax.
    
    Args:
        agent: Pre-existing agent to continue training (None for new)
        episodes: Number of training episodes
        depth: Minimax search depth
        log_interval: Logging interval
        **kwargs: Additional arguments for Trainer
        
    Returns:
        Training statistics
    """
    trainer = Trainer(
        agent=agent,
        opponent_type=OpponentType.MINIMAX,
        minimax_depth=depth,
        **kwargs
    )
    return trainer.train(episodes=episodes, log_interval=log_interval)
def train_self_play(
    agent: Optional[DQNAgent] = None,
    episodes: int = 1000,
    log_interval: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for self-play training.
    
    Args:
        agent: Pre-existing agent to continue training (None for new)
        episodes: Number of training episodes
        log_interval: Logging interval
        **kwargs: Additional arguments for Trainer
        
    Returns:
        Training statistics
    """
    trainer = Trainer(
        agent=agent,
        opponent_type=OpponentType.SELF,
        **kwargs
    )
    return trainer.train(episodes=episodes, log_interval=log_interval)
class CurriculumStage:
    """Defines a stage in curriculum training."""
    
    def __init__(
        self,
        name: str,
        opponent_type: OpponentType,
        episodes: int,
        minimax_depth: int = 1,
        mixed_random_prob: float = 0.5,  # For MIXED type: probability of random moves
        promotion_win_rate: float = 0.5,
        promotion_window: int = 100,
        max_attempts: int = 3
    ):
        """
        Initialize a curriculum stage.
        
        Args:
            name: Human-readable name for this stage
            opponent_type: Type of opponent for this stage
            episodes: Number of episodes for this stage
            minimax_depth: Depth for minimax opponent (if applicable)
            mixed_random_prob: For MIXED opponent, probability of random vs minimax
            promotion_win_rate: Win rate required to advance to next stage
            promotion_window: Number of recent episodes to consider for promotion
            max_attempts: Maximum times to repeat this stage if promotion threshold not met
        """
        self.name = name
        self.opponent_type = opponent_type
        self.episodes = episodes
        self.minimax_depth = minimax_depth
        self.mixed_random_prob = mixed_random_prob
        self.promotion_win_rate = promotion_win_rate
        self.promotion_window = promotion_window
        self.max_attempts = max_attempts
class CurriculumTrainer:
    """
    Curriculum-based trainer that progressively increases opponent difficulty.
    
    The agent starts against easier opponents and graduates to harder ones
    as it improves, allowing it to learn winning strategies before facing
    opponents that punish every mistake.
    """
    
    # Default curriculum stages - smoother progression with mixed stages
    DEFAULT_CURRICULUM = [
        CurriculumStage(
            name="Random Opponent",
            opponent_type=OpponentType.RANDOM,
            episodes=1000,
            promotion_win_rate=0.75,
            promotion_window=100,
            max_attempts=2
        ),
        CurriculumStage(
            name="Mixed (70% Random, 30% Minimax D1)",
            opponent_type=OpponentType.MIXED,
            episodes=1500,
            minimax_depth=1,
            mixed_random_prob=0.70,
            promotion_win_rate=0.65,
            promotion_window=100,
            max_attempts=2
        ),
        CurriculumStage(
            name="Mixed (50% Random, 50% Minimax D1)",
            opponent_type=OpponentType.MIXED,
            episodes=1500,
            minimax_depth=1,
            mixed_random_prob=0.50,
            promotion_win_rate=0.55,
            promotion_window=100,
            max_attempts=2
        ),
        CurriculumStage(
            name="Mixed (30% Random, 70% Minimax D1)",
            opponent_type=OpponentType.MIXED,
            episodes=1500,
            minimax_depth=1,
            mixed_random_prob=0.30,
            promotion_win_rate=0.50,
            promotion_window=100,
            max_attempts=2
        ),
        CurriculumStage(
            name="Minimax Depth 1",
            opponent_type=OpponentType.MINIMAX,
            episodes=2000,
            minimax_depth=1,
            promotion_win_rate=0.45,
            promotion_window=100,
            max_attempts=3
        ),
        CurriculumStage(
            name="Minimax Depth 2",
            opponent_type=OpponentType.MINIMAX,
            episodes=2500,
            minimax_depth=2,
            promotion_win_rate=0.35,
            promotion_window=100,
            max_attempts=3
        ),
        CurriculumStage(
            name="Minimax Depth 3",
            opponent_type=OpponentType.MINIMAX,
            episodes=3000,
            minimax_depth=3,
            promotion_win_rate=0.25,
            promotion_window=100,
            max_attempts=3
        ),
        CurriculumStage(
            name="Minimax Depth 4",
            opponent_type=OpponentType.MINIMAX,
            episodes=4000,
            minimax_depth=4,
            promotion_win_rate=0.15,
            promotion_window=100,
            max_attempts=3
        ),
        CurriculumStage(
            name="Minimax Depth 5",
            opponent_type=OpponentType.MINIMAX,
            episodes=5000,
            minimax_depth=5,
            promotion_win_rate=0.10,
            promotion_window=100,
            max_attempts=3
        ),
    ]
    
    def __init__(
        self,
        agent: Optional[DQNAgent] = None,
        stages: Optional[List[CurriculumStage]] = None,
        model_dir: str = 'models',
        batch_size: int = 64,
        replay_buffer_size: int = 50000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
    ):
        """
        Initialize the curriculum trainer.
        
        Args:
            agent: Pre-initialized DQN agent (creates new if None)
            stages: List of curriculum stages (uses default if None)
            model_dir: Directory to save models
            batch_size: Batch size for training
            replay_buffer_size: Size of replay buffer
            replay_buffer: Optional shared replay buffer instance (if None, a new buffer is created)
            epsilon_start: Starting exploration rate (for first stage)
            epsilon_end: Final exploration rate
        """
        self.agent = agent if agent is not None else DQNAgent()
        self.stages = stages if stages is not None else self.DEFAULT_CURRICULUM
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        
        self.current_stage_index = 0
        self.stage_attempts = {}  # Track attempts per stage
        self.stage_results = []  # Store results from each stage
        
        os.makedirs(model_dir, exist_ok=True)
    
    def _calculate_stage_epsilon(self, stage_index: int) -> Tuple[float, float]:
        """
        Calculate epsilon start/end for a stage.
        
        Early stages get more exploration, later stages exploit more.
        """
        total_stages = len(self.stages)
        
        # Gradually decrease exploration across stages
        # First stage: full epsilon range
        # Last stage: mostly exploitation
        stage_progress = stage_index / max(1, total_stages - 1)
        
        # Epsilon start decreases as we progress through stages
        stage_epsilon_start = self.epsilon_start - (stage_progress * 0.5 * (self.epsilon_start - self.epsilon_end))
        
        # Epsilon end is always the global minimum for the last stage,
        # but higher for earlier stages to allow continued exploration
        stage_epsilon_end = self.epsilon_end + ((1 - stage_progress) * 0.15)
        
        return max(stage_epsilon_end, stage_epsilon_start), stage_epsilon_end
    
    def train(
        self,
        log_interval: int = 100,
        save_interval: int = 500,
        eval_interval: int = 250,
        eval_games: int = 20,
        verbose: bool = True,
        # New controls
        patience: int = 5,
        backoff_factor: float = 0.5,
        promotion_strategy: str = 'hybrid',  # '2of3', 'wilson', 'hybrid'
        promotion_batches: int = 3,
        promotion_eval_games: int = 120,
        wilson_z: float = 1.96,
        dual_replay: bool = False,
        dual_recent_capacity: int = 10000,
        dual_recent_fraction: float = 0.5,
    ) -> Dict[str, Any]:
        """Run curriculum training forever.
        - Infinite curriculum loop (never stops).
        - Patience/backoff for MIXED stages by relaxing toward previous stage.
        - For MINIMAX depth stages, backoff trains on the previous stage for a few attempts,
          then retries the harder stage.
        - Promotion stabilized via 2-of-3 batches and/or Wilson lower bound.
        - Optional dual replay sampling (recent+global) while keeping persistence across stages.
        """
        start_time = time.time()
        patience = max(1, int(patience))
        backoff_factor = float(max(0.01, min(0.99, backoff_factor)))
        promotion_strategy = str(promotion_strategy).lower().strip()
        promotion_batches = max(1, int(promotion_batches))
        promotion_eval_games = max(30, int(promotion_eval_games))
        if verbose:
            print("=" * 60)
            print("CURRICULUM TRAINING")
            print("=" * 60)
            print(f"Stages: {len(self.stages)}")
            for i, stage in enumerate(self.stages):
                depth_str = f" (depth {stage.minimax_depth})" if stage.opponent_type == OpponentType.MINIMAX else ""
                print(
                    f"  {i+1}. {stage.name}{depth_str}: {stage.episodes} episodes, "
                    f"promotion at {stage.promotion_win_rate:.0%} win rate"
                )
            print("=" * 60)
            print()
        shared_replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)
        training_replay = shared_replay_buffer
        if dual_replay:
            training_replay = DualReplayBuffer(
                global_buffer=shared_replay_buffer,
                recent_capacity=dual_recent_capacity,
                recent_fraction=dual_recent_fraction,
            )
        # Adaptive difficulty state
        mixed_effective_prob = {}  # stage_index -> current random prob
        failures_since_backoff = {}  # stage_index -> count
        # Recovery mode for depth stages
        recovery_target = None  # (hard_stage_index)
        recovery_attempts_left = 0
        total_episodes = 0
        stage_index = 0
        def run_promotion_eval(trainer: Trainer, threshold: float):
            """Return (promoted:bool, details:dict)."""
            per_batch = max(20, promotion_eval_games // promotion_batches)
            batch_rates = []
            wins_total = 0
            games_total = 0
            passed_batches = 0
            for _ in range(promotion_batches):
                ev = trainer._evaluate(per_batch)
                wr = float(ev.get('win_rate', 0.0))
                batch_rates.append(wr)
                # Convert to wins conservatively
                wins = int(round(wr * per_batch))
                wins_total += wins
                games_total += per_batch
                if wr >= threshold:
                    passed_batches += 1
            overall_wr = (wins_total / games_total) if games_total else 0.0
            wlb = wilson_lower_bound(wins_total, games_total, z=wilson_z)
            pass_2of3 = passed_batches >= max(1, (promotion_batches * 2 + 2) // 3)  # ceil(2/3)
            pass_wilson = wlb >= threshold
            if promotion_strategy == '2of3':
                promoted = pass_2of3
            elif promotion_strategy == 'wilson':
                promoted = pass_wilson
            else:
                promoted = pass_2of3 or pass_wilson
            return promoted, {
                'batch_win_rates': batch_rates,
                'passed_batches': passed_batches,
                'promotion_eval_games': games_total,
                'promotion_win_rate': overall_wr,
                'wilson_lb': wlb,
                'strategy': promotion_strategy,
            }
        while True:
            stage = self.stages[stage_index]
            # Determine effective opponent parameters
            eff_opponent_type = stage.opponent_type
            eff_minimax_depth = stage.minimax_depth
            eff_mixed_prob = stage.mixed_random_prob
            if stage.opponent_type == OpponentType.MIXED:
                if stage_index not in mixed_effective_prob:
                    mixed_effective_prob[stage_index] = float(stage.mixed_random_prob)
                eff_mixed_prob = mixed_effective_prob[stage_index]
            # If we're in recovery mode, train on the previous stage instead
            if recovery_target is not None and recovery_attempts_left > 0:
                recover_idx = max(0, recovery_target - 1)
                stage = self.stages[recover_idx]
                eff_opponent_type = stage.opponent_type
                eff_minimax_depth = stage.minimax_depth
                eff_mixed_prob = stage.mixed_random_prob
                stage_index_effective = recover_idx
                recovery_attempts_left -= 1
            else:
                stage_index_effective = stage_index
                if recovery_target is not None and recovery_attempts_left <= 0:
                    # Recovery done; go back to hard stage
                    stage_index_effective = recovery_target
                    stage = self.stages[stage_index_effective]
                    eff_opponent_type = stage.opponent_type
                    eff_minimax_depth = stage.minimax_depth
                    eff_mixed_prob = stage.mixed_random_prob
                    recovery_target = None
            # Accounting
            self.current_stage_index = stage_index_effective
            attempts = self.stage_attempts.get(stage_index_effective, 0) + 1
            self.stage_attempts[stage_index_effective] = attempts
            if verbose:
                print()
                print('-' * 60)
                print(f"STAGE {stage_index_effective + 1}/{len(self.stages)}: {stage.name}")
                if dual_replay and attempts == 1 and stage_index_effective == 0:
                    print(f"  Using dual replay sampling (recent_capacity={dual_recent_capacity}, recent_fraction={dual_recent_fraction:.2f})")
                print('-' * 60)
            # Epsilon scheduling
            eps_start, eps_end = self._calculate_stage_epsilon(stage_index_effective)
            if stage_index_effective == 0 and attempts == 1:
                self.agent.epsilon = eps_start
            elif attempts == 1:
                self.agent.epsilon = min(self.agent.epsilon + 0.1, eps_start * 0.5)
            if verbose:
                print(f"  Epsilon: {self.agent.epsilon:.3f} -> {eps_end:.3f}")
            trainer = Trainer(
                agent=self.agent,
                opponent_type=eff_opponent_type,
                minimax_depth=eff_minimax_depth,
                mixed_random_prob=eff_mixed_prob,
                model_dir=self.model_dir,
                batch_size=self.batch_size,
                replay_buffer_size=self.replay_buffer_size,
                replay_buffer=training_replay,
                epsilon_start=self.agent.epsilon,
                epsilon_end=eps_end,
                epsilon_decay_episodes=int(stage.episodes * 0.5),
            )
            trainer.curriculum_stage = stage_index_effective
            trainer.curriculum_total_stages = len(self.stages)
            trainer.curriculum_stage_name = stage.name
            stats = trainer.train(
                episodes=stage.episodes,
                log_interval=log_interval,
                save_interval=save_interval,
                eval_interval=eval_interval,
                eval_games=eval_games,
            )
            total_episodes += stage.episodes
            promoted, promo = run_promotion_eval(trainer, stage.promotion_win_rate)
            self.stage_results.append({
                'stage': stage_index_effective,
                'name': stage.name,
                'attempt': attempts,
                'episodes': stage.episodes,
                'final_win_rate': stats.get('final_win_rate', 0),
                **promo,
                'promoted': bool(promoted),
            })
            if verbose:
                bw = ", ".join([f"{w:.1%}" for w in promo.get('batch_win_rates', [])])
                print(f"  [PROMO] batches=[{bw}] passed={promo.get('passed_batches',0)}/{promotion_batches} overall={promo.get('promotion_win_rate',0.0):.1%} wilson_lb={promo.get('wilson_lb',0.0):.1%}")
            if promoted:
                if verbose:
                    thr = float(stage.promotion_win_rate)
                    passed = int(promo.get('passed_batches', 0))
                    wilson_lb = float(promo.get('wilson_lb', 0.0))
                    overall = float(promo.get('promotion_win_rate', 0.0))
                    need_batches = (promotion_batches * 2 + 2) // 3  # ceil(2/3 * promotion_batches)

                    strategy = promo.get('strategy', 'hybrid')
                    # Determine which criterion actually triggered promotion (for accurate logs)
                    if strategy == '2of3':
                        via = '2-of-3 batches'
                    elif strategy == 'wilson':
                        via = 'Wilson lower bound'
                    else:
                        if passed >= need_batches:
                            via = '2-of-3 batches'
                        elif wilson_lb >= thr:
                            via = 'Wilson lower bound'
                        else:
                            via = 'hybrid'

                    print(
                        f"\n✓ PROMOTED! (strategy={strategy}, via={via}) "
                        f"overall {overall:.1%}; "
                        f"batches {passed}/{promotion_batches} (need {need_batches} at >= {thr:.0%}); "
                        f"wilson_lb {wilson_lb:.1%} (need {thr:.0%})"
                    )
                failures_since_backoff[stage_index_effective] = 0
                stage_model_name = f"curriculum_stage{stage_index_effective + 1}_{stage.name.replace(' ', '_').lower()}"
                stage_model_path = os.path.join(self.model_dir, stage_model_name)
                self.agent.save(stage_model_path)
                if verbose:
                    print(f"  Saved stage checkpoint: {stage_model_path}")
                # Advance if we were training the main track
                if stage_index_effective == stage_index:
                    stage_index = min(stage_index + 1, len(self.stages) - 1)
                continue
            # Not promoted
            failures_since_backoff[stage_index_effective] = failures_since_backoff.get(stage_index_effective, 0) + 1
            if verbose:
                print(f"\n✗ Not promoted. failures_since_backoff={failures_since_backoff[stage_index_effective]}/{patience}")
            if failures_since_backoff[stage_index_effective] >= patience:
                failures_since_backoff[stage_index_effective] = 0
                # Mixed stages: relax toward previous stage by backoff_factor
                if stage.opponent_type == OpponentType.MIXED and stage_index_effective > 0:
                    prev = self.stages[stage_index_effective - 1]
                    prev_p = float(getattr(prev, 'mixed_random_prob', 1.0))
                    cur_p = float(mixed_effective_prob.get(stage_index_effective, stage.mixed_random_prob))
                    new_p = cur_p + backoff_factor * (prev_p - cur_p)
                    mixed_effective_prob[stage_index_effective] = max(0.0, min(prev_p, new_p))
                    if verbose:
                        print(f"  [BACKOFF] Mixed stage random_prob -> {mixed_effective_prob[stage_index_effective]:.3f} (toward prev {prev_p:.3f})")
                # Depth stages: temporarily recover on previous stage for `patience` attempts
                elif stage.opponent_type == OpponentType.MINIMAX and stage_index_effective > 0:
                    recovery_target = stage_index_effective
                    recovery_attempts_left = patience
                    if verbose:
                        print(f"  [BACKOFF] Depth stage: training previous stage for {patience} attempts, then retry")
            # Keep looping forever
            continue
        # Final summary
        if verbose:
            print()
            print("=" * 60)
            print("CURRICULUM TRAINING COMPLETE")
            print("=" * 60)
            print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
            print(f"Total episodes: {total_episodes}")
            print()
            print("Stage Results:")
            for result in self.stage_results:
                status = "✓" if result['promoted'] else "✗"
                promo_wr = result.get('promotion_win_rate', result.get('final_win_rate', 0.0))
                promo_n = int(result.get('promotion_eval_games', 0))
                extra = f" (promotion eval {promo_wr:.1%} over {promo_n} games)" if promo_n else ""
                print(
                    f"  {status} Stage {result['stage'] + 1} ({result['name']}): "
                    f"attempt {result['attempt']}" + extra
                )
            print()
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, f"curriculum_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.agent.save(final_model_path)
        if verbose:
            print(f"Final model saved: {final_model_path}")
        
        stages_completed = sum(1 for r in self.stage_results if r.get('promoted'))
        return {
            'total_episodes': total_episodes,
            'total_time': elapsed,
            'stages_completed': stages_completed,
            'stage_results': self.stage_results,
            'final_model_path': final_model_path
        }
def train_curriculum(
    agent: Optional[DQNAgent] = None,
    log_interval: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for curriculum training.
    
    Args:
        agent: Pre-existing agent to continue training (None for new)
        log_interval: Logging interval
        **kwargs: Additional arguments for CurriculumTrainer
        
    Returns:
        Training statistics
    """
    trainer = CurriculumTrainer(agent=agent, **kwargs)
    return trainer.train(log_interval=log_interval)
if __name__ == "__main__":
    # Quick test
    from connect4.debug import debug, DebugLevel
    debug.configure(level=DebugLevel.INFO)
    
    print("Testing Trainer with Minimax opponent...")
    print("=" * 50)
    
    trainer = Trainer(
        opponent_type=OpponentType.MINIMAX,
        minimax_depth=4,
        batch_size=32,
        replay_buffer_size=10000
    )
    
    # Train for a few episodes
    stats = trainer.train(
        episodes=100,
        log_interval=10,
        save_interval=50,
        eval_interval=50,
        eval_games=10
    )
    
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
