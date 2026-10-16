# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own RAM-backed reward WAVs without importing the training runtime in the cleanup process."""

import os
import select
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from typing import Iterator

_RAM_AUDIO_ROOT = '/dev/shm'


@contextmanager
def ram_audio_directory(rank: int) -> Iterator[str]:
    """Create a private reward directory in Linux shared memory and remove it after use.

    A separate process also removes the directory if its owner exits without
    running Python cleanup (including SIGTERM, SIGKILL, and ``os._exit``). The
    watcher must remain alive; killing the entire job/container can kill both
    processes and requires cleanup by the job/container runtime.

    Raises:
        RuntimeError: If writable ``/dev/shm`` is unavailable or the cleanup
            process does not become ready. Audio is never spilled to disk.
    """
    if not os.path.isdir(_RAM_AUDIO_ROOT) or not os.access(_RAM_AUDIO_ROOT, os.W_OK):
        raise RuntimeError('Reward audio requires writable /dev/shm; disk fallback is disabled.')
    directory = os.path.join(_RAM_AUDIO_ROOT, f'easy_magpie_rewards_rank{rank}_{os.getpid()}_{uuid.uuid4().hex}')
    os.mkdir(directory, mode=0o700)
    read_fd = write_fd = None
    watcher = None
    try:
        read_fd, write_fd = os.pipe()
        # Execute this stdlib-only file directly: importing NeMo in the watcher
        # would initialize another training runtime and potentially CUDA.
        watcher = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), str(read_fd), str(os.getpid()), directory],
            pass_fds=(read_fd,),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        os.close(read_fd)
        read_fd = None
        with watcher.stdout:
            if not select.select([watcher.stdout], [], [], 10)[0] or watcher.stdout.readline() != b'ready\n':
                raise RuntimeError('Reward audio cleanup process did not become ready.')
        yield directory
    finally:
        try:
            _remove_directory(directory)
        finally:
            for fd in (read_fd, write_fd):
                if fd is not None:
                    os.close(fd)
            if watcher is not None:
                try:
                    watcher.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    watcher.kill()
                    watcher.wait()
                    raise RuntimeError('Reward audio cleanup process did not exit.') from None
                if watcher.returncode != 0:
                    raise RuntimeError(f'Reward audio cleanup process failed with exit code {watcher.returncode}.')


def _remove_directory(directory: str) -> None:
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        pass


def _watch_owner(read_fd: int, owner_pid: int, directory: str) -> None:
    try:
        os.write(sys.stdout.fileno(), b'ready\n')
        # EOF handles normal scope exit. Checking the parent also handles
        # forked reward workers inheriting a copy of the pipe's write end.
        while os.getppid() == owner_pid:
            if select.select([read_fd], [], [], 1)[0] and not os.read(read_fd, 1):
                break
    finally:
        os.close(read_fd)
        _remove_directory(directory)


if __name__ == '__main__':
    _watch_owner(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
