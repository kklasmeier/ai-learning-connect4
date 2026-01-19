"""
data_manager.py - Data management for Connect Four AI training

This module handles storage and retrieval of training data, job information,
and model registrations using JSON files stored in the data directory.
"""

import os
import json
import time
import filelock
import datetime
from typing import Dict, List, Any, Optional, Union
import shutil

from connect4.debug import debug, DebugLevel

# Define paths to data files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
JOBS_FILE = os.path.join(DATA_DIR, 'jobs.json')
MODELS_FILE = os.path.join(DATA_DIR, 'models.json')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
GAMES_DIR = os.path.join(DATA_DIR, 'games')
os.makedirs(GAMES_DIR, exist_ok=True)

# Constants
MAX_RECENT_LOGS = 1000
HISTORY_SAMPLE_RATE = 50

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def safe_read_json(file_path: str) -> List[Dict]:
    """
    Safely read a JSON file with file locking.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data (empty list if file doesn't exist)
    """
    if not os.path.exists(file_path):
        return []
    
    lock_path = f"{file_path}.lock"
    with filelock.FileLock(lock_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            debug.error(f"Error decoding JSON from {file_path}", "data")
            return []
            
def safe_write_json(file_path: str, data: Any) -> bool:
    """
    Safely write data to a JSON file with atomic updates.
    
    Args:
        file_path: Path to JSON file
        data: Data to write
        
    Returns:
        True if successful, False otherwise
    """
    lock_path = f"{file_path}.lock"
    with filelock.FileLock(lock_path):
        try:
            temp_file = f"{file_path}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            shutil.move(temp_file, file_path)
            return True
        except Exception as e:
            debug.error(f"Error writing to {file_path}: {e}", "data")
            return False

def get_new_job_id() -> int:
    """Generate a new unique job ID."""
    jobs = safe_read_json(JOBS_FILE)
    if not jobs:
        return 1
    return max(job['job_id'] for job in jobs) + 1

def create_job(parameters: Dict[str, Any]) -> int:
    """Create a new job entry."""
    job_id = get_new_job_id()
    start_time = datetime.datetime.now().isoformat()
    
    job_data = {
        "job_id": job_id,
        "start_time": start_time,
        "end_time": None,
        "total_episodes": parameters.get('episodes', 0),
        "episodes_completed": 0,
        "model_dir": f"models/job_{job_id}",
        "status": "running",
        "parameters": parameters
    }
    
    jobs = safe_read_json(JOBS_FILE)
    jobs.append(job_data)
    
    if safe_write_json(JOBS_FILE, jobs):
        debug.info(f"Created new job with ID {job_id}", "data")
        
        recent_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_recent.json")
        history_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_history.json")
        
        safe_write_json(recent_log_file, [])
        safe_write_json(history_log_file, [])
        
        return job_id
    else:
        debug.error(f"Failed to create job", "data")
        return -1

def update_job_progress(job_id: int, episodes_completed: int) -> bool:
    """Update job progress."""
    jobs = safe_read_json(JOBS_FILE)
    
    for job in jobs:
        if job['job_id'] == job_id:
            job['episodes_completed'] = episodes_completed
            
            if safe_write_json(JOBS_FILE, jobs):
                debug.trace(f"Updated job {job_id} progress to {episodes_completed}", "data")
                return True
            else:
                debug.error(f"Failed to update job {job_id} progress", "data")
                return False
    
    debug.error(f"Job {job_id} not found", "data")
    return False

def complete_job(job_id: int) -> bool:
    """Mark a job as complete."""
    jobs = safe_read_json(JOBS_FILE)
    
    for job in jobs:
        if job['job_id'] == job_id:
            job['end_time'] = datetime.datetime.now().isoformat()
            job['status'] = "completed"
            
            if safe_write_json(JOBS_FILE, jobs):
                debug.info(f"Marked job {job_id} as completed", "data")
                process_historical_data(job_id)
                return True
            else:
                debug.error(f"Failed to complete job {job_id}", "data")
                return False
    
    debug.error(f"Job {job_id} not found", "data")
    return False

def add_episode_log(job_id: int, episode_data: Dict[str, Any]) -> bool:
    """Add episode data to log."""
    if 'timestamp' not in episode_data:
        episode_data['timestamp'] = datetime.datetime.now().isoformat()
    
    episode_data['job_id'] = job_id
    
    recent_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_recent.json")
    
    logs = safe_read_json(recent_log_file)
    logs.append(episode_data)
    
    if len(logs) > MAX_RECENT_LOGS:
        if episode_data['episode'] % HISTORY_SAMPLE_RATE == 0:
            add_to_history(job_id, episode_data)
        logs = logs[-MAX_RECENT_LOGS:]
    
    if safe_write_json(recent_log_file, logs):
        debug.trace(f"Added episode {episode_data['episode']} log for job {job_id}", "data")
        return True
    else:
        debug.error(f"Failed to add episode log for job {job_id}", "data")
        return False

def add_to_history(job_id: int, episode_data: Dict[str, Any]) -> bool:
    """Add episode data to historical log."""
    history_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_history.json")
    history_logs = safe_read_json(history_log_file)
    
    history_logs.append(episode_data)
    
    if safe_write_json(history_log_file, history_logs):
        debug.trace(f"Added episode {episode_data['episode']} to history for job {job_id}", "data")
        return True
    else:
        debug.error(f"Failed to add episode to history for job {job_id}", "data")
        return False

def process_historical_data(job_id: int) -> bool:
    """Process historical data for a completed job."""
    recent_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_recent.json")
    history_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_history.json")
    
    recent_logs = safe_read_json(recent_log_file)
    history_logs = safe_read_json(history_log_file)
    
    history_episodes = set(log['episode'] for log in history_logs)
    
    for log in recent_logs:
        episode = log['episode']
        if episode % HISTORY_SAMPLE_RATE == 0 and episode not in history_episodes:
            history_logs.append(log)
            history_episodes.add(episode)
    
    history_logs.sort(key=lambda x: x['episode'])
    
    if safe_write_json(history_log_file, history_logs):
        debug.info(f"Processed historical data for job {job_id}", "data")
        return True
    else:
        debug.error(f"Failed to process historical data for job {job_id}", "data")
        return False

def register_model(job_id: int, episode: int, path: str, is_final: bool = False) -> bool:
    """Register a saved model."""
    models = safe_read_json(MODELS_FILE)
    
    model_id = len(models) + 1
    timestamp = datetime.datetime.now().isoformat()
    
    model_data = {
        "model_id": model_id,
        "job_id": job_id,
        "episode": episode,
        "path": path,
        "timestamp": timestamp,
        "is_final": is_final
    }
    
    models.append(model_data)
    
    if safe_write_json(MODELS_FILE, models):
        debug.info(f"Registered model for job {job_id}, episode {episode}", "data")
        return True
    else:
        debug.error(f"Failed to register model for job {job_id}", "data")
        return False

def get_job_data(job_id: Optional[int] = None) -> Union[Dict, List[Dict]]:
    """Get job data."""
    jobs = safe_read_json(JOBS_FILE)
    
    if job_id is None:
        return jobs
    
    for job in jobs:
        if job['job_id'] == job_id:
            return job
    
    debug.warning(f"Job {job_id} not found", "data")
    return {}

def get_all_jobs() -> List[Dict]:
    """Get all jobs."""
    return safe_read_json(JOBS_FILE)

def get_job_info(job_id: int) -> Optional[Dict]:
    """Get info for a specific job."""
    jobs = safe_read_json(JOBS_FILE)
    for job in jobs:
        if job['job_id'] == job_id:
            return job
    return None

def get_episode_logs(job_id: int, recent: bool = True) -> List[Dict]:
    """Get episode logs for a job."""
    log_type = "recent" if recent else "history"
    log_file = os.path.join(LOGS_DIR, f"job_{job_id}_{log_type}.json")
    
    logs = safe_read_json(log_file)
    return logs

def get_registered_models(job_id: Optional[int] = None) -> List[Dict]:
    """Get registered models."""
    models = safe_read_json(MODELS_FILE)
    
    if job_id is None:
        return models
    
    return [model for model in models if model['job_id'] == job_id]

def save_game_moves(job_id: int, episode: int, move_data: List[Dict], winner: Optional[str], 
                   game_length: int) -> bool:
    """
    Save a game's move history and statistics for later replay.
    
    Args:
        job_id: Training job ID
        episode: Episode number in training
        move_data: List of dictionaries with move details and stats
        winner: Winner of the game ('X', 'O', or None for draw)
        game_length: Number of moves in the game
        
    Returns:
        True if successful, False otherwise
    """
    game_data = {
        "job_id": job_id,
        "episode": episode,
        "moves": move_data,
        "winner": winner,
        "game_length": game_length,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    games_file = os.path.join(GAMES_DIR, f"job_{job_id}_games.json")
    
    games = safe_read_json(games_file)
    games.append(game_data)
    
    if safe_write_json(games_file, games):
        debug.trace(f"Saved moves for job {job_id}, episode {episode}", "data")
        return True
    else:
        debug.error(f"Failed to save moves for job {job_id}, episode {episode}", "data")
        return False

def get_saved_games(job_id: Optional[int] = None) -> List[Dict]:
    """Get saved games for replay, sorted by timestamp."""
    if job_id is not None:
        games_file = os.path.join(GAMES_DIR, f"job_{job_id}_games.json")
        if os.path.exists(games_file):
            return safe_read_json(games_file)
        return []
    else:
        all_games = []
        for file in os.listdir(GAMES_DIR):
            if file.startswith("job_") and file.endswith("_games.json"):
                games_file = os.path.join(GAMES_DIR, file)
                all_games.extend(safe_read_json(games_file))
        # Sort by timestamp so games[-1] is truly the latest
        all_games.sort(key=lambda g: g.get('timestamp', ''))
        return all_games
    

def get_latest_game_id(job_id: Optional[int] = None) -> Optional[int]:
    """Get the ID of the latest saved game."""
    games = get_saved_games(job_id)
    if not games:
        return None
    
    return len(games) - 1

def purge_all_data() -> bool:
    """
    Purge all training data, logs, games, and model registrations.
    
    This removes:
    - All job records
    - All episode logs
    - All saved games
    - All model registrations
    
    Note: This does NOT delete the actual model files in the models/ directory.
    
    Returns:
        True if successful, False otherwise
    """
    success = True
    
    # Clear jobs file
    if os.path.exists(JOBS_FILE):
        if not safe_write_json(JOBS_FILE, []):
            debug.error("Failed to clear jobs file", "data")
            success = False
        else:
            debug.info("Cleared jobs file", "data")
    
    # Clear models file
    if os.path.exists(MODELS_FILE):
        if not safe_write_json(MODELS_FILE, []):
            debug.error("Failed to clear models file", "data")
            success = False
        else:
            debug.info("Cleared models registry", "data")
    
    # Clear logs directory
    if os.path.exists(LOGS_DIR):
        for file in os.listdir(LOGS_DIR):
            file_path = os.path.join(LOGS_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    debug.trace(f"Removed log file: {file}", "data")
            except Exception as e:
                debug.error(f"Failed to remove {file}: {e}", "data")
                success = False
        debug.info("Cleared logs directory", "data")
    
    # Clear games directory
    if os.path.exists(GAMES_DIR):
        for file in os.listdir(GAMES_DIR):
            file_path = os.path.join(GAMES_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    debug.trace(f"Removed game file: {file}", "data")
            except Exception as e:
                debug.error(f"Failed to remove {file}: {e}", "data")
                success = False
        debug.info("Cleared games directory", "data")
    
    if success:
        print("All training data purged successfully.")
        print("Note: Model files in models/ directory were NOT deleted.")
        print("To delete models, manually remove files from the models/ directory.")
    
    return success