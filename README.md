# AI Learning Connect Four

A Connect Four AI learning system using reinforcement learning. This project implements a reinforcement learning agent that improves through self-play using Deep Q-Networks (DQN).

## Project Goals
- Implement Connect Four game environment ✓
- Create reinforcement learning agent that improves through self-play ✓
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

## Usage

The project provides a command-line interface through `run.py` for interacting with the Connect Four game and AI.

### Playing the Game

To play Connect Four against a random AI:

```bash
python run.py game play --ai random
```

### AI Commands

#### Initialize a new DQN model

```bash
python run.py ai init --hidden_size 128
```

This creates a new model and saves it to the `models` directory.

#### Train the AI through self-play

```bash
python run.py ai train --episodes 1000 --debug_level info
```

Optional parameters:
- `--model models/model_name`: Continue training from an existing model
- `--hidden_size 128`: Specify hidden layer size for new models
- `--debug_level`: Control logging verbosity
- `--log_interval`: Specify how often to print updates (default: episodes/100)

#### Play against a trained AI model

```bash
python run.py ai play --model models/final_model_TIMESTAMP
```

Replace `TIMESTAMP` with the actual timestamp from your saved model.

#### View training jobs

List all training jobs:
```bash
python run.py ai jobs
```

View details of a specific job:
```bash
python run.py ai jobs --job_id 1
```

This shows job parameters, progress, and recent episode results.

#### Purge all data

Reset all jobs, logs, and training data (requires confirmation):
```bash
python run.py ai purge --confirm
```

This resets all tracking data while preserving code and model files.

### Data Management

The system tracks training data in JSON files:

1. **Jobs**: Each training run is tracked as a job with parameters and progress
2. **Episode Logs**: Detailed logs of training episodes are stored in two formats:
   - Recent logs: Full details for the last 1000 episodes
   - Historical logs: Sampled data (1 in 50) for older episodes
3. **Model Registry**: Tracks all saved model checkpoints

All data is stored in the `data` directory, making it portable and human-readable.

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

### Training Statistics

During training, statistics are automatically saved in the following locations:
- `data/jobs.json`: Overall job information
- `data/logs/job_X_recent.json`: Detailed logs for recent episodes
- `data/logs/job_X_history.json`: Sampled logs for older episodes

Statistics include:
- Win rates for Player 1 and Player 2
- Draw rates
- Average game length
- Exploration rate (epsilon)
- Training loss values

### Managing Models

Models are saved to the `models` directory with timestamps in their filenames. The system creates:
- Periodic checkpoints during training (e.g., `model_TIMESTAMP_ep100.pt`)
- Final model after training completes (e.g., `final_model_TIMESTAMP.pt`)

To start fresh with no previous models:

```bash
python run.py ai purge --confirm
python run.py ai init --hidden_size 128
python run.py ai train --episodes 1000
```

### Reinforcement Learning Approach

The AI uses the following reinforcement learning components:

1. **Deep Q-Network (DQN)**: A neural network that predicts the value of each possible move
2. **Experience Replay**: Stores game states and outcomes for efficient learning
3. **Self-Play**: The agent plays against itself to continuously improve
4. **Epsilon-Greedy Exploration**: Balance between exploration and exploitation during training

### Reward System

The AI learns through a reward system:
- +1.0 for winning a game
- -1.0 for losing a game
- +0.1 for a draw
- -0.01 small penalty per move to encourage efficient play

## Future Work

- Web interface with Flask for visualizing training progress
- Configurable neural network architecture
- Advanced opponent models for improved training
- Model evaluation and comparison tools

## Troubleshooting

**Common issues:**

1. **"ModuleNotFoundError"**: Make sure you've installed the package with `pip install -e .`

2. **PyTorch errors**: Ensure you have PyTorch installed correctly for your platform

3. **"No module named 'gymnasium'"**: Install the Gymnasium package with `pip install gymnasium`

4. **"No module named 'filelock'"**: Install the filelock package with `pip install filelock`

5. **File permission errors**: Ensure the data directory has write permissions

6. **Empty job listings**: Make sure at least one training job has been run

## License

MIT License

## Acknowledgments

- The OpenAI Gymnasium framework for reinforcement learning environments
- PyTorch for neural network implementation