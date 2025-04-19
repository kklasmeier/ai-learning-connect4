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

# Constants
MAX_RECENT_LOGS = 1000  # Maximum number of recent episode logs to keep
HISTORY_SAMPLE_RATE = 50  # Keep 1 in every N episodes for historical data

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# File utility functions
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
            # Write to a temporary file first
            temp_file = f"{file_path}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Replace the original file (atomic operation)
            shutil.move(temp_file, file_path)
            return True
        except Exception as e:
            debug.error(f"Error writing to {file_path}: {e}", "data")
            return False

# Job management functions
def get_new_job_id() -> int:
    """
    Generate a new unique job ID.
    
    Returns:
        New job ID
    """
    jobs = safe_read_json(JOBS_FILE)
    if not jobs:
        return 1
    return max(job['job_id'] for job in jobs) + 1

def create_job(parameters: Dict[str, Any]) -> int:
    """
    Create a new job entry.
    
    Args:
        parameters: Training parameters
        
    Returns:
        New job ID
    """
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
        
        # Initialize log files
        recent_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_recent.json")
        history_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_history.json")
        
        safe_write_json(recent_log_file, [])
        safe_write_json(history_log_file, [])
        
        return job_id
    else:
        debug.error(f"Failed to create job", "data")
        return -1

def update_job_progress(job_id: int, episodes_completed: int) -> bool:
    """
    Update job progress.
    
    Args:
        job_id: Job ID
        episodes_completed: Number of completed episodes
        
    Returns:
        True if successful, False otherwise
    """
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
    """
    Mark a job as complete.
    
    Args:
        job_id: Job ID
        
    Returns:
        True if successful, False otherwise
    """
    jobs = safe_read_json(JOBS_FILE)
    
    for job in jobs:
        if job['job_id'] == job_id:
            job['end_time'] = datetime.datetime.now().isoformat()
            job['status'] = "completed"
            
            if safe_write_json(JOBS_FILE, jobs):
                debug.info(f"Marked job {job_id} as completed", "data")
                
                # Process historical data
                process_historical_data(job_id)
                
                return True
            else:
                debug.error(f"Failed to complete job {job_id}", "data")
                return False
    
    debug.error(f"Job {job_id} not found", "data")
    return False

# Episode log management
def add_episode_log(job_id: int, episode_data: Dict[str, Any]) -> bool:
    """
    Add episode data to log.
    
    Args:
        job_id: Job ID
        episode_data: Episode data to log
        
    Returns:
        True if successful, False otherwise
    """
    # Add timestamp if not present
    if 'timestamp' not in episode_data:
        episode_data['timestamp'] = datetime.datetime.now().isoformat()
    
    # Ensure job_id is included
    episode_data['job_id'] = job_id
    
    # Path to recent logs file
    recent_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_recent.json")
    
    # Read existing logs
    logs = safe_read_json(recent_log_file)
    
    # Add new log
    logs.append(episode_data)
    
    # Keep only the most recent MAX_RECENT_LOGS
    if len(logs) > MAX_RECENT_LOGS:
        # Before discarding, check if we should add to history
        if episode_data['episode'] % HISTORY_SAMPLE_RATE == 0:
            add_to_history(job_id, episode_data)
        
        # Keep only the most recent logs
        logs = logs[-MAX_RECENT_LOGS:]
    
    # Save updated logs
    if safe_write_json(recent_log_file, logs):
        debug.trace(f"Added episode {episode_data['episode']} log for job {job_id}", "data")
        return True
    else:
        debug.error(f"Failed to add episode log for job {job_id}", "data")
        return False

def add_to_history(job_id: int, episode_data: Dict[str, Any]) -> bool:
    """
    Add episode data to historical log.
    
    Args:
        job_id: Job ID
        episode_data: Episode data to log
        
    Returns:
        True if successful, False otherwise
    """
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
    """
    Process historical data for a completed job.
    Ensures that we have proper sampling of historical data.
    
    Args:
        job_id: Job ID
        
    Returns:
        True if successful, False otherwise
    """
    recent_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_recent.json")
    history_log_file = os.path.join(LOGS_DIR, f"job_{job_id}_history.json")
    
    recent_logs = safe_read_json(recent_log_file)
    history_logs = safe_read_json(history_log_file)
    
    # Check which episodes we already have in history
    history_episodes = set(log['episode'] for log in history_logs)
    
    # Add episodes to history that should be sampled but aren't yet
    for log in recent_logs:
        episode = log['episode']
        if episode % HISTORY_SAMPLE_RATE == 0 and episode not in history_episodes:
            history_logs.append(log)
            history_episodes.add(episode)
    
    # Sort by episode number
    history_logs.sort(key=lambda x: x['episode'])
    
    if safe_write_json(history_log_file, history_logs):
        debug.info(f"Processed historical data for job {job_id}", "data")
        return True
    else:
        debug.error(f"Failed to process historical data for job {job_id}", "data")
        return False

# Model registration
def register_model(job_id: int, episode: int, path: str, is_final: bool = False) -> bool:
    """
    Register a saved model.
    
    Args:
        job_id: Job ID
        episode: Episode number
        path: Path to model file
        is_final: Whether this is the final model
        
    Returns:
        True if successful, False otherwise
    """
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

# Data retrieval functions
def get_job_data(job_id: Optional[int] = None) -> Union[Dict, List[Dict]]:
    """
    Get job data.
    
    Args:
        job_id: Specific job ID to get, or None for all jobs
        
    Returns:
        Job data or list of all jobs
    """
    jobs = safe_read_json(JOBS_FILE)
    
    if job_id is None:
        return jobs
    
    for job in jobs:
        if job['job_id'] == job_id:
            return job
    
    debug.warning(f"Job {job_id} not found", "data")
    return {}

def get_episode_logs(job_id: int, recent: bool = True) -> List[Dict]:
    """
    Get episode logs for a job.
    
    Args:
        job_id: Job ID
        recent: Whether to get recent logs (True) or historical logs (False)
        
    Returns:
        Episode logs
    """
    log_type = "recent" if recent else "history"
    log_file = os.path.join(LOGS_DIR, f"job_{job_id}_{log_type}.json")
    
    logs = safe_read_json(log_file)
    return logs

def get_registered_models(job_id: Optional[int] = None) -> List[Dict]:
    """
    Get registered models.
    
    Args:
        job_id: Specific job ID to filter by, or None for all models
        
    Returns:
        List of registered models
    """
    models = safe_read_json(MODELS_FILE)
    
    if job_id is None:
        return models
    
    return [model for model in models if model['job_id'] == job_id]