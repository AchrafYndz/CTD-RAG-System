import logging
import sys
import time
from datetime import datetime
from pathlib import Path


def setup_logging(level: int = logging.INFO, module_name: str = None) -> logging.Logger:
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent.parent
    logs_dir = root_dir / "logs"
    
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"rag_system_{timestamp}.log"
    log_filepath = logs_dir / log_filename
    
    root_logger = logging.getLogger()
    
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    root_logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    
    if module_name is None:
        frame = sys._getframe(1)
        module_name = frame.f_globals.get('__name__', 'unknown')
    
    logger = logging.getLogger(module_name)
    
    logger.info(f"Logging initialized - File: {log_filepath}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    if name is None:
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


def set_log_level(level: int):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers:
        handler.setLevel(level)


def clean_old_logs(days_to_keep: int = 1):
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    logs_dir = root_dir / "logs"
    
    if not logs_dir.exists():
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
    
    deleted_count = 0
    for log_file in logs_dir.glob("rag_system_*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            deleted_count += 1
    
    if deleted_count > 0:
        logger = get_logger(__name__)
        logger.info(f"Cleaned up {deleted_count} old log files") 