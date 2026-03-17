# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


import enum
import logging as _logging
import sys
import threading
import warnings
from contextlib import contextmanager
from logging.handlers import MemoryHandler

from nemo.constants import NEMO_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR, NEMO_ENV_VARNAME_TESTING
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.formatters.base import BaseNeMoFormatter, DebugNeMoFormatter
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.metaclasses import Singleton

__all__ = ["Logger", "LogMode"]


class LogMode(enum.IntEnum):
    """Enum to control how many times to log messages in NeMo logging"""

    EACH = 0  # Log the message each time
    ONCE = 1  # Log the message only once. The same message will not be logged again.


class Logger(metaclass=Singleton):
    """NeMo's logging class. Makes some changes on top of python's logging module to aid model devs."""

    # Level 0
    NOTSET = _logging.NOTSET

    # Level 10
    DEBUG = _logging.DEBUG

    # Level 20
    INFO = _logging.INFO

    # Level 30
    WARNING = _logging.WARNING

    # Level 40
    ERROR = _logging.ERROR

    # Level 50
    CRITICAL = _logging.CRITICAL

    _level_names = {
        0: "NOTSET",
        10: "DEBUG",
        20: "INFO",
        30: "WARNING",
        40: "ERROR",
        50: "CRITICAL",
    }

    def __init__(self, capture_warnings=True):

        self._logger = None
        # Multi-GPU runs run in separate processes, thread locks shouldn't be needed
        self._logger_lock = threading.Lock()
        self._handlers = dict()
        self.old_warnings_showwarning = None
        self._define_logger(capture_warnings)
        self.once_logged = set()
        self.rank = 0 if is_global_rank_zero() else "UNK"

    def _define_logger(self, capture_warnings=True):
        """Creates the logger if not already created. Called in init"""

        # Use double-checked locking to avoid taking lock unnecessarily.
        if self._logger is not None:
            return self._logger

        with self._logger_lock:
            try:
                self._logger = _logging.getLogger("nemo_logger")
                # By default, silence all loggers except the logger for rank 0
                self.remove_stream_handlers()
                # If NEMO_TESTING is set, add a streamhandler to all ranks
                if get_envbool(NEMO_ENV_VARNAME_TESTING, False):
                    old_factory = _logging.getLogRecordFactory()

                    def record_factory(*args, **kwargs):
                        record = old_factory(*args, **kwargs)
                        record.rank = self.rank
                        return record

                    _logging.setLogRecordFactory(record_factory)
                    self.add_stream_handlers(formatter=DebugNeMoFormatter)
                elif is_global_rank_zero():
                    self.add_stream_handlers()

                # Add memoryhandlers, essentially buffers. They are used to save messages that we will flush to file
                # once the appropriate file handlers are added.
                if is_global_rank_zero():
                    # Add a memoryhandler for error messages. Only logged on rank 0
                    self._handlers["memory_err"] = MemoryHandler(-1)
                    self._handlers["memory_err"].addFilter(lambda record: record.levelno > _logging.INFO)
                    formatter = BaseNeMoFormatter
                    self._handlers["memory_err"].setFormatter(formatter())
                    self._logger.addHandler(self._handlers["memory_err"])
                # Add a memoryhandler for all messages on all ranks
                self._handlers["memory_all"] = MemoryHandler(-1)
                formatter = BaseNeMoFormatter
                self._handlers["memory_all"].setFormatter(formatter())
                self._logger.addHandler(self._handlers["memory_all"])

            finally:
                level = Logger.INFO
                if get_envbool(NEMO_ENV_VARNAME_TESTING, False):
                    level = Logger.DEBUG
                self.set_verbosity(verbosity_level=level)
                self.captureWarnings(capture_warnings)

        self._logger.propagate = False

    def remove_stream_handlers(self):
        """Removes StreamHandler that log to stdout and stderr from the logger."""
        if self._logger is None:
            raise RuntimeError("Impossible to set handlers if the Logger is not predefined")

        # ======== Remove Handler if already existing ========

        try:
            self._logger.removeHandler(self._handlers["stream_stdout"])
            del self._handlers["stream_stdout"]
        except KeyError:
            pass

        try:
            self._logger.removeHandler(self._handlers["stream_stderr"])
            del self._handlers["stream_stderr"]
        except KeyError:
            pass

    def add_stream_handlers(self, formatter=BaseNeMoFormatter):
        """Add StreamHandler that log to stdout and stderr to the logger. INFO and lower logs are streamed to stdout
        while WARNING and higher are streamed to stderr. If the NEMO_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR environment
        variable is set, all logs are sent to stderr instead.
        """
        if self._logger is None:
            raise RuntimeError("Impossible to set handlers if the Logger is not predefined")

        # Add the output handler.
        if get_envbool(NEMO_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR, False):
            self._handlers["stream_stdout"] = _logging.StreamHandler(sys.stderr)

        else:
            self._handlers["stream_stdout"] = _logging.StreamHandler(sys.stdout)
            self._handlers["stream_stdout"].addFilter(lambda record: record.levelno <= _logging.INFO)

            self._handlers["stream_stderr"] = _logging.StreamHandler(sys.stderr)
            self._handlers["stream_stderr"].addFilter(lambda record: record.levelno > _logging.INFO)

        self._handlers["stream_stdout"].setFormatter(formatter())
        self._logger.addHandler(self._handlers["stream_stdout"])

        try:
            self._handlers["stream_stderr"].setFormatter(formatter())
            self._logger.addHandler(self._handlers["stream_stderr"])
        except KeyError:
            pass

    def reset_stream_handler(self, formatter=BaseNeMoFormatter):
        """Removes then adds stream handlers."""
        self.remove_stream_handlers()
        self.add_stream_handlers(formatter=formatter)

    def add_file_handler(self, log_file):
        """Add a FileHandler to logger that logs all messages to a file. If the logger had a MemoryHandler at
        self._handlers["memory_all"], those buffered messages are flushed to the new file, and the MemoryHandler is
        closed."""
        if self._logger is None:
            raise RuntimeError("Impossible to set handlers if the Logger is not predefined")

        self._handlers["file"] = _logging.FileHandler(log_file)
        formatter = BaseNeMoFormatter
        self._handlers["file"].setFormatter(formatter())
        self._logger.addHandler(self._handlers["file"])

        if self._handlers.get("memory_all", None):
            self._handlers["memory_all"].setTarget(self._handlers["file"])
            self._handlers["memory_all"].close()  # flush and remove
            del self._handlers["memory_all"]

    def add_err_file_handler(self, log_file):
        """Add a FileHandler to logger that logs all WARNING and higher messages to a file. If the logger had a
        MemoryHandler at self._handlers["memory_err"], those buffered messages are flushed to the new file, and the
        MemoryHandler is closed."""
        if self._logger is None:
            raise RuntimeError("Impossible to set handlers if the Logger is not predefined")

        self._handlers["file_err"] = _logging.FileHandler(log_file)
        self._handlers["file_err"].addFilter(lambda record: record.levelno > _logging.INFO)

        formatter = BaseNeMoFormatter
        self._handlers["file_err"].setFormatter(formatter())
        self._logger.addHandler(self._handlers["file_err"])

        if self._handlers.get("memory_err", None):
            self._handlers["memory_err"].setTarget(self._handlers["file_err"])
            self._handlers["memory_err"].close()  # flush and remove
            del self._handlers["memory_err"]

    def getEffectiveLevel(self):
        """Return how much logging output will be produced."""
        if self._logger is not None:
            return self._logger.getEffectiveLevel()

    def get_verbosity(self):
        """See getEffectiveLevel"""
        return self.getEffectiveLevel()

    def setLevel(self, verbosity_level):
        """Sets the threshold for what messages will be logged."""
        if self._logger is not None:
            self._logger.setLevel(verbosity_level)

            for handler in self._logger.handlers:
                handler.setLevel(verbosity_level)

    def set_verbosity(self, verbosity_level):
        """See setLevel"""
        self.setLevel(verbosity_level)

    @contextmanager
    def patch_stderr_handler(self, stream):
        """Sends messages that should log to stderr to stream instead. Useful for unittests"""
        if self._logger is not None:
            try:
                old_stream = self._handlers["stream_stderr"].stream
                if old_stream is None:
                    raise ValueError

                # Port backwards set_stream() from python 3.7
                self._handlers["stream_stderr"].acquire()
                try:
                    self._handlers["stream_stderr"].flush()
                    self._handlers["stream_stderr"].stream = stream
                finally:
                    self._handlers["stream_stderr"].release()

                yield stream
            except (KeyError, ValueError):
                raise RuntimeError("Impossible to patch logging handlers if handler does not exist")
            finally:
                # Port backwards set_stream() from python 3.7
                self._handlers["stream_stderr"].acquire()
                try:
                    self._handlers["stream_stderr"].flush()
                    self._handlers["stream_stderr"].stream = old_stream
                finally:
                    self._handlers["stream_stderr"].release()

        else:
            raise RuntimeError("Impossible to patch logging handlers if handler does not exist")

    @contextmanager
    def patch_stdout_handler(self, stream):
        """Sends messages that should log to stdout to stream instead. Useful for unittests"""
        if self._logger is not None:
            try:
                old_stream = self._handlers["stream_stdout"].stream
                if old_stream is None:
                    raise ValueError

                # Port backwards set_stream() from python 3.7
                self._handlers["stream_stdout"].acquire()
                try:
                    self._handlers["stream_stdout"].flush()
                    self._handlers["stream_stdout"].stream = stream
                finally:
                    self._handlers["stream_stdout"].release()

                yield stream
            except (KeyError, ValueError):
                raise RuntimeError("Impossible to patch logging handlers if handler does not exist")
            finally:
                # Port backwards set_stream() from python 3.7
                self._handlers["stream_stdout"].acquire()
                try:
                    self._handlers["stream_stdout"].flush()
                    self._handlers["stream_stdout"].stream = old_stream
                finally:
                    self._handlers["stream_stdout"].release()

        else:
            raise RuntimeError("Impossible to patch logging handlers if handler does not exist")

    @contextmanager
    def temp_verbosity(self, verbosity_level):
        """Sets the a temporary threshold for what messages will be logged."""

        if self._logger is not None:

            old_verbosity = self.get_verbosity()

            try:
                self.set_verbosity(verbosity_level)
                yield

            finally:
                self.set_verbosity(old_verbosity)

        else:
            try:
                yield

            finally:
                pass

    def captureWarnings(self, capture):
        """
        If capture is true, redirect all warnings to the logging package.
        If capture is False, ensure that warnings are not redirected to logging
        but to their original destinations.
        """

        if self._logger is not None:

            if capture and self.old_warnings_showwarning is None:
                # Backup Method
                self.old_warnings_showwarning = warnings.showwarning
                warnings.showwarning = self._showwarning

            elif not capture and self.old_warnings_showwarning is not None:
                # Restore Method
                warnings.showwarning = self.old_warnings_showwarning
                self.old_warnings_showwarning = None

    def _warning_is_ignored(self, category):
        from warnings import filters

        # Search the filters
        for action, msg, cat, mod, ln in filters:
            # least-common demoninator if multiple filters for the same class.
            if cat == category and action == 'ignore':
                return True
        return False

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        """
        Implementation of showwarnings which redirects to logging.
        It will call warnings.formatwarning and will log the resulting string
        with level logging.WARNING.
        """
        s = warnings.formatwarning(message, category, filename, lineno, line)
        if self._warning_is_ignored(category):
            return
        self.warning("%s", s)

    def _logged_once(self, msg, mode):
        PREFIX_LEN = 12
        if mode == LogMode.ONCE:
            if msg[PREFIX_LEN:] in self.once_logged:
                return True
            self.once_logged.add(msg[PREFIX_LEN:])
        return False

    def debug(self, msg, *args, mode=LogMode.EACH, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.DEBUG) and not self._logged_once(msg, mode):
            self._logger._log(Logger.DEBUG, msg, args, **kwargs, stacklevel=2)

    def info(self, msg, *args, mode=LogMode.EACH, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.INFO) and not self._logged_once(msg, mode):
            self._logger._log(Logger.INFO, msg, args, **kwargs, stacklevel=2)

    def warning(self, msg, *args, mode=LogMode.EACH, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.WARNING) and not self._logged_once(msg, mode):
            self._logger._log(Logger.WARNING, msg, args, **kwargs, stacklevel=2)

    def error(self, msg, *args, mode=LogMode.EACH, **kwargs):
        """
        Log 'msg % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.ERROR) and not self._logged_once(msg, mode):
            self._logger._log(Logger.ERROR, msg, args, **kwargs, stacklevel=2)

    def critical(self, msg, *args, mode=LogMode.EACH, **kwargs):
        """
        Log 'msg % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
        if (
            self._logger is not None
            and self._logger.isEnabledFor(Logger.CRITICAL)
            and not self._logged_once(msg, mode)
        ):
            self._logger._log(Logger.CRITICAL, msg, args, **kwargs, stacklevel=2)
