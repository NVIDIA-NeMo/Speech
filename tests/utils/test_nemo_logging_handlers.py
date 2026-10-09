# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from logging.handlers import MemoryHandler

from nemo.utils.nemo_logging import Logger


def _memory_handlers(logger):
    return [h for h in logger.handlers if isinstance(h, MemoryHandler)]


def test_memory_handlers_come_off_the_logger_once_the_files_are_added(tmp_path):
    """The buffers are a bridge until the log files exist, not a permanent sink.

    MemoryHandler.flush() only empties its buffer when it has a target, and
    close() clears the target, so a handler left attached keeps every record
    for the life of the process.
    """
    nemo_logger = Logger()
    nemo_logger.add_file_handler(str(tmp_path / "all.log"))
    nemo_logger.add_err_file_handler(str(tmp_path / "err.log"))

    assert _memory_handlers(nemo_logger._logger) == []

    nemo_logger.info("after the files are in place")
    nemo_logger.error("and an error too")

    assert all(len(handler.buffer) == 0 for handler in _memory_handlers(nemo_logger._logger))
