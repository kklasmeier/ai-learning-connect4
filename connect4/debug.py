"""
debug.py - Debug and logging functionality for Connect Four implementation

This module provides centralized debug and logging capabilities with configurable levels
and output options to facilitate development and troubleshooting.
"""

import logging
import os
import sys
import time
from enum import Enum
from typing import List, Optional, Dict, Any, Set

# Define debug levels as an Enum for type checking
class DebugLevel(Enum):
    NONE = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5

# Mapping to standard logging levels
LEVEL_MAP = {
    DebugLevel.NONE: logging.NOTSET,
    DebugLevel.ERROR: logging.ERROR,
    DebugLevel.WARNING: logging.WARNING,
    DebugLevel.INFO: logging.INFO,
    DebugLevel.DEBUG: logging.DEBUG,
    DebugLevel.TRACE: logging.DEBUG + 1  # Python logging doesn't have TRACE
}

# ANSI color codes for terminal output
COLORS = {
    DebugLevel.ERROR: "\033[31m",  # Red
    DebugLevel.WARNING: "\033[33m",  # Yellow
    DebugLevel.INFO: "\033[32m",  # Green
    DebugLevel.DEBUG: "\033[36m",  # Cyan
    DebugLevel.TRACE: "\033[35m",  # Magenta
    "RESET": "\033[0m"
}

class DebugManager:
    """Manages debug and logging functionality for the Connect Four game."""
    
    def __init__(self):
        self._level = DebugLevel.INFO
        self._enabled = True
        self._log_to_file = False
        self._log_file = None
        self._enabled_components: Set[str] = set()  # Empty set means all components
        self._logger = self._setup_logger()
        self._performance_markers: Dict[str, float] = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance."""
        logger = logging.getLogger("connect4")
        logger.setLevel(LEVEL_MAP[self._level])
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def configure(self, level: DebugLevel = None, 
                 enabled: bool = None,
                 log_file: str = None,
                 components: List[str] = None):
        """
        Configure the debug manager settings.
        
        Args:
            level: Debug level to set
            enabled: Whether debugging is enabled
            log_file: Path to log file (None for no file logging)
            components: List of components to enable debugging for (empty for all)
        """
        if level is not None:
            self._level = level
            self._logger.setLevel(LEVEL_MAP[level])
        
        if enabled is not None:
            self._enabled = enabled
        
        if log_file is not None:
            self._log_file = log_file
            self._log_to_file = bool(log_file)
            
            # Remove existing file handlers
            for handler in self._logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self._logger.removeHandler(handler)
            
            # Add new file handler if needed
            if self._log_to_file:
                file_handler = logging.FileHandler(log_file)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)
        
        if components is not None:
            self._enabled_components = set(components)
    
    def _should_log(self, level: DebugLevel, component: str = None) -> bool:
        """Determine if a message should be logged based on settings."""
        if not self._enabled:
            return False
        
        if level.value > self._level.value:
            return False
        
        if component and self._enabled_components and component not in self._enabled_components:
            return False
        
        return True
    
    def log(self, level: DebugLevel, message: str, component: str = None):
        """
        Log a message at the specified level.
        
        Args:
            level: Debug level for the message
            message: The message to log
            component: Optional component name for filtering
        """
        if not self._should_log(level, component):
            return
        
        # Format with component if provided
        formatted_message = message
        if component:
            formatted_message = f"[{component}] {message}"
        
        # Use the logger with the appropriate level
        if level == DebugLevel.ERROR:
            self._logger.error(formatted_message)
        elif level == DebugLevel.WARNING:
            self._logger.warning(formatted_message)
        elif level == DebugLevel.INFO:
            self._logger.info(formatted_message)
        elif level == DebugLevel.DEBUG:
            self._logger.debug(formatted_message)
        elif level == DebugLevel.TRACE:
            # Python's logging doesn't have TRACE, so we use DEBUG
            self._logger.debug(f"TRACE: {formatted_message}")
    
    # Convenience methods for each level
    def error(self, message: str, component: str = None):
        """Log an error message."""
        self.log(DebugLevel.ERROR, message, component)
    
    def warning(self, message: str, component: str = None):
        """Log a warning message."""
        self.log(DebugLevel.WARNING, message, component)
    
    def info(self, message: str, component: str = None):
        """Log an info message."""
        self.log(DebugLevel.INFO, message, component)
    
    def debug(self, message: str, component: str = None):
        """Log a debug message."""
        self.log(DebugLevel.DEBUG, message, component)
    
    def trace(self, message: str, component: str = None):
        """Log a trace message."""
        self.log(DebugLevel.TRACE, message, component)
    
    # Performance tracking methods
    def start_timer(self, marker_name: str):
        """Start a timer for performance tracking."""
        self._performance_markers[marker_name] = time.time()
    
    def end_timer(self, marker_name: str, component: str = None) -> Optional[float]:
        """
        End a timer and log the elapsed time.
        
        Args:
            marker_name: Name of the marker to end
            component: Optional component name for the log entry
            
        Returns:
            Elapsed time in seconds, or None if marker not found
        """
        if marker_name not in self._performance_markers:
            self.warning(f"Timer '{marker_name}' not started", "debug")
            return None
        
        elapsed = time.time() - self._performance_markers[marker_name]
        self.debug(f"Performance [{marker_name}]: {elapsed:.6f} seconds", component)
        del self._performance_markers[marker_name]
        return elapsed
    
    def set_from_string(self, level_str: str):
        """Set debug level from a string (for command line arguments)."""
        level_map = {
            "none": DebugLevel.NONE,
            "error": DebugLevel.ERROR,
            "warning": DebugLevel.WARNING,
            "info": DebugLevel.INFO,
            "debug": DebugLevel.DEBUG,
            "trace": DebugLevel.TRACE
        }
        
        level_str = level_str.lower()
        if level_str in level_map:
            self.configure(level=level_map[level_str])
            self.info(f"Debug level set to {level_str.upper()}")
        else:
            self.warning(f"Unknown debug level: {level_str}")


# Create a singleton instance
debug = DebugManager()

# Example usage
if __name__ == "__main__":
    # Set debug level
    debug.configure(level=DebugLevel.DEBUG)
    
    # Log messages at different levels
    debug.error("This is an error message", "test")
    debug.warning("This is a warning message", "test")
    debug.info("This is an info message", "test")
    debug.debug("This is a debug message", "test")
    debug.trace("This is a trace message", "test")
    
    # Test component filtering
    debug.configure(components=["test"])
    debug.info("This should appear (component: test)", "test")
    debug.info("This should not appear (component: other)", "other")
    
    # Test performance tracking
    debug.start_timer("test_operation")
    time.sleep(0.1)  # Simulate work
    debug.end_timer("test_operation", "test")