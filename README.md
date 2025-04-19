# AI Learning Connect Four

A Connect Four AI learning system using reinforcement learning. This project implements a reinforcement learning agent that improves through self-play using Deep Q-Networks (DQN).

## Project Goals
- Implement Connect Four game environment ✓
- Create reinforcement learning agent that improves through self-play ✓
- Track and analyze training progress ✓
- Observe and replay AI games ✓
- Develop web interface to visualize learning and play against the AI (planned)
- Ensure compatibility with both Raspberry Pi and desktop platforms ✓

## Technologies
- Python 3
- PyTorch (neural networks and training)
- Gymnasium (reinforcement learning environment)
- Flask (future web interface)

## Project Structure

```
ai-learning-connect4/
├── README.md
├── run.py                      # Main entry point
├── setup.py                    # Installation file
├── data/                       # Data storage directory
│   ├── jobs.json               # Training jobs tracking
│   ├── models.json             # Model registry
│   ├── games/                  # Saved games for replay
│   └── logs/                   # Episode logs directory
├── connect4/                   # Core package
│   ├── __init__.py
│   ├── debug.py                # Debug and logging system
│   ├── utils.py                # Common utilities
│   ├── game/                   # Game core module
│   │   ├── __init__.py
│   │   ├── board.py            # Game board representation
│   │   └── rules.py            # Game rules and Gymnasium environment
│   ├── ai/                     # AI module
│   │   ├── __init__.py
│   │   ├── dqn.py              # DQN implementation
│   │   ├── replay_buffer.py    # Experience replay buffer
│   │   ├── training.py         # Self-play training framework
│   │   └── utils.py            # AI-specific utilities
│   ├── interfaces/             # User interfaces
│   │   ├── __init__.py
│   │   └── cli.py              # Command-line interface
│   └── data/                   # Data management module
│       ├── __init__.py
│       └── data_manager.py     # Training data management
├── models/                     # Saved model checkpoints
└── tests/                      # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-learning-connect4.git
cd ai-learning-connect4
```

2. Install dependencies:
```bash
pip install -e .
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Command-Line Interface

The project provides a comprehensive command-line interface through `run.py` for interacting with the Connect Four game and AI.

### Game Component

#### Play the Game

Play Connect Four against a random AI:
```bash
python run.py game play
```

Play with two human players:
```bash
python run.py game play --ai none
```

#### Test Game Logic

Test a specific board position:
```bash
python run.py game test --position 0,0,0,0,1,1,1,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

Run all validation tests:
```bash
python run.py game test_all
```

#### Benchmark Performance

Run performance tests:
```bash
python run.py game benchmark --iterations 5000
```

### AI Component

#### Model Management

Initialize a new model:
```bash
python run.py ai init --hidden_size 128
```

The `--hidden_size` parameter determines the neural network complexity:
- Higher values (256, 512) give more learning capacity but train slower
- Lower values (64, 128) train faster but may have less potential
- Default is 128, which works well for Connect Four

Play against a trained model:
```bash
python run.py ai play --model models/final_model_TIMESTAMP
```

Replace `TIMESTAMP` with the actual timestamp from your saved model.

#### Training

Train the AI through self-play:
```bash
python run.py ai train --episodes 1000 --debug_level info
```

Training options:
- `--episodes`: Number of games to play (higher = better learning but more time)
- `--model`: Continue training from an existing model
- `--hidden_size`: Neural network size if creating a new model
- `--log_interval`: How often to print updates (default: episodes/100)
- `--debug_level`: Control logging verbosity

Training guidance:
- 1,000-5,000 episodes: Basic learning (beginner level)
- 10,000-50,000 episodes: Intermediate strategies
- 100,000+ episodes: Advanced play (significant training time)

Continue training from an existing model:
```bash
python run.py ai train --episodes 2000 --model models/final_model_TIMESTAMP
```

#### Job Management

View all training jobs:
```bash
python run.py ai jobs
```

View details of a specific job:
```bash
python run.py ai jobs --job_id 1
```

Purge all data (requires confirmation):
```bash
python run.py ai purge --confirm
```

#### Game Replay

List all saved games:
```bash
python run.py ai games --list
```

List games for a specific job:
```bash
python run.py ai games --list --job_id 1
```

Replay a specific game:
```bash
python run.py ai games --replay 0
```

Control playback speed:
```bash
python run.py ai games --replay 5 --delay 1.0
```

### Debug Levels

The system has several debug levels for controlling log output:

```bash
python run.py ai train --debug_level [level]
```

Available levels:
- `none`: No output (completely silent)
- `error`: Only error messages
- `warning`: Errors and warnings
- `info`: Informational messages (default)
- `debug`: Detailed debugging information
- `trace`: Most verbose output

Or use the shorthand:

```bash
python run.py ai train --debug
```

This enables full debug output (equivalent to `--debug_level debug`).

## Data Management

The system tracks training data in JSON files:

1. **Jobs**: Each training run is tracked as a job with parameters and progress
2. **Episode Logs**: Training progress is stored in two formats:
   - Recent logs: Full details for the last 1000 episodes
   - Historical logs: Sampled data (1 in 50) for older episodes
3. **Game Replays**: Move sequences from interesting games are saved for replay
4. **Model Registry**: Tracks all saved model checkpoints

All data is stored in the `data` directory, making it portable and human-readable.

### Training Statistics

During training, the system tracks:
- Win rates for Player 1 and Player 2
- Draw rates
- Average game length
- Exploration rate (epsilon)
- Training loss values

These metrics help evaluate when the model has reached diminishing returns, which typically occurs when:
- Win rates stabilize over many episodes
- Draw rates increase and plateau
- Loss values flatten at a low level

### Game Replay System

The system intelligently saves game replays for:
- First and last games of training
- Regular milestone games (every 50 episodes)
- Random sampling (~5% of games)
- Unusually short or long games
- Games with interesting patterns

This allows you to observe how the AI's strategy evolves throughout training without excessive storage requirements.

### Managing Models

Models are saved to the `models` directory with timestamps in their filenames:
- Periodic checkpoints during training (e.g., `model_TIMESTAMP_ep100.pt`)
- Final model after training completes (e.g., `final_model_TIMESTAMP.pt`)

To start fresh with no previous models:

```bash
python run.py ai purge --confirm
python run.py ai init --hidden_size 128
python run.py ai train --episodes 1000
```

## Reinforcement Learning Approach

The AI uses a Deep Q-Network (DQN) approach for reinforcement learning:

1. **Neural Network Architecture**:
   - Input: 3-channel representation of the board state
   - Convolutional layers to capture spatial patterns
   - Fully connected hidden layer (configurable size)
   - Output: Q-values for each possible column

2. **Training Components**:
   - Experience Replay: Stores and randomly samples past experiences
   - Target Network: Stabilizes learning with delayed updates
   - Epsilon-Greedy Strategy: Balances exploration vs. exploitation

3. **Self-Play Training Loop**:
   - Agent plays against itself, improving with each game
   - Both sides share the same neural network
   - Learning from wins, losses, and draws

4. **Reward System**:
   - +1.0 for winning a game
   - -1.0 for losing a game
   - +0.1 for a draw
   - -0.01 small penalty per move to encourage efficient play

## Future Work

- Web interface with Flask for visualizing training progress
- Configurable neural network architecture
- Advanced opponent models for improved training
- Model evaluation and comparison tools
- Training visualization and analysis
- Neural network architecture exploration

## Troubleshooting

**Common issues:**

1. **"ModuleNotFoundError"**: Make sure you've installed the package with `pip install -e .`

2. **PyTorch errors**: Ensure you have PyTorch installed correctly for your platform

3. **"No module named 'gymnasium'"**: Install the Gymnasium package with `pip install gymnasium`

4. **"No module named 'filelock'"**: Install the filelock package with `pip install filelock`

5. **File permission errors**: Ensure the data directory has write permissions

6. **Empty job listings**: Make sure at least one training job has been run

7. **Model loading errors**: Verify you're using the correct model path

## License

MIT License

## Acknowledgments

- The OpenAI Gymnasium framework for reinforcement learning environments
- PyTorch for neural network implementation