import logging
import pdb
from typing_extensions import List, Tuple, Dict, Union, Any, Literal
import os
from logging.handlers import RotatingFileHandler


class CameraLogFilter(logging.Filter):
    def __init__(self, enabled_cameras: List[str]):
        super().__init__()
        self.enabled_cameras = enabled_cameras

    def filter(self, record):
        # Only allow logs from enabled cameras
        return any(camera_id in record.name for camera_id in self.enabled_cameras)


def setup_logging(log_dir='logs', log_file='app.log', level=logging.DEBUG, enabled_cameras=None):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.log_dir = log_dir
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    if enabled_cameras:
        camera_filter = CameraLogFilter(enabled_cameras)
        for handler in root_logger.handlers:
            handler.addFilter(camera_filter)