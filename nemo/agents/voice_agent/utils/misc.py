# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(log_file: str = "bot_server.log", log_level: str = "DEBUG", rotation: str = "1 day"):
    """Configure loguru to emit to stderr and a rotating log file."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSSS}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        level=log_level,
    )

    logger.add(log_file, rotation=rotation, level=log_level)


def setup_rotating_log(
    log_file: str,
    log_level: str = "DEBUG",
    create_new_log: bool = False,
    overwrite_existing: bool = False,
    rotation: str = "1 day",
) -> None:
    """Roll an existing log file aside (or delete it), then call ``setup_logging``.

    If ``create_new_log`` is True and the file exists, it is either removed
    (``overwrite_existing=True``) or renamed with a timestamp suffix. Used by
    bot server scripts that want a fresh log on every run.
    """
    if create_new_log and os.path.exists(log_file):
        if overwrite_existing:
            os.remove(log_file)
            logger.info(f"Removed existing log file: {log_file}")
        else:
            new_log_file = log_file.replace(".log", f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            os.rename(log_file, new_log_file)
            logger.info(f"Renamed existing log file: {log_file} to {new_log_file}")

    setup_logging(log_file=log_file, log_level=log_level, rotation=rotation)


class FileLogger:
    """Simple file+stdout logger with caller location tracking."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file

    def _get_caller_location(self) -> str:
        """Return file:function:line of the caller, skipping frames inside FileLogger."""
        logger_methods = {"log", "info", "error", "warning", "debug", "__call__", "_get_caller_location"}
        for frame_info in inspect.stack():
            if frame_info.function not in logger_methods:
                path = Path(frame_info.filename).resolve()
                return f"{path.name}:{frame_info.function}:{frame_info.lineno}"
        return "unknown"

    def log(self, message: str, include_caller: bool = True):
        """Write a timestamped line to the log file (if set) and stdout."""
        if include_caller:
            message = f"{self._get_caller_location()} | {message}"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{timestamp} | {message}"
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        print(message, flush=True)

    def __call__(self, message: str, include_caller: bool = True):
        """Allow calling the logger instance directly like a function."""
        self.log(message, include_caller=include_caller)

    def info(self, message: str, include_caller: bool = True):
        """Log at INFO level."""
        self.log(f"[INFO]: {message}", include_caller=include_caller)

    def error(self, message: str, include_caller: bool = True):
        """Log at ERROR level."""
        self.log(f"[ERROR]: {message}", include_caller=include_caller)

    def warning(self, message: str, include_caller: bool = True):
        """Log at WARNING level."""
        self.log(f"[WARNING]: {message}", include_caller=include_caller)

    def debug(self, message: str, include_caller: bool = True):
        """Log at DEBUG level."""
        self.log(f"[DEBUG]: {message}", include_caller=include_caller)
