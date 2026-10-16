# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo.collections.tts.parts.utils import reward_audio

pytestmark = pytest.mark.unit


def test_ram_audio_directory_requires_shared_memory(tmp_path, monkeypatch):
    missing = tmp_path / 'missing-shared-memory'
    monkeypatch.setattr(reward_audio, '_RAM_AUDIO_ROOT', str(missing))
    with pytest.raises(RuntimeError, match='/dev/shm'):
        with reward_audio.ram_audio_directory(rank=0):
            pytest.fail('Must not fall back to a disk directory')
    assert not missing.exists()


@pytest.mark.parametrize('fail', [False, True])
def test_ram_audio_directory_cleans_and_reaps_watcher(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(reward_audio, '_RAM_AUDIO_ROOT', str(tmp_path))
    processes = []
    original_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(reward_audio.subprocess, 'Popen', popen)
    try:
        with reward_audio.ram_audio_directory(rank=3) as directory:
            path = Path(directory)
            assert path.parent == tmp_path
            assert path.stat().st_mode & 0o777 == 0o700
            (path / 'audio.wav').write_bytes(b'audio')
            if fail:
                raise ValueError('reward failed')
    except ValueError:
        assert fail
    assert not path.exists()
    assert len(processes) == 1
    assert processes[0].returncode == 0


def test_ram_audio_directory_cleans_after_watcher_start_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(reward_audio, '_RAM_AUDIO_ROOT', str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError('cannot start watcher')

    monkeypatch.setattr(reward_audio.subprocess, 'Popen', fail)
    with pytest.raises(OSError, match='cannot start watcher'):
        with reward_audio.ram_audio_directory(rank=0):
            pytest.fail('Must not generate audio without a watcher')
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not Path('/dev/shm').is_dir(), reason='Requires Linux shared memory')
def test_ram_audio_directory_uses_dev_shm():
    with reward_audio.ram_audio_directory(rank=0) as directory:
        assert Path(directory).parent == Path('/dev/shm')
        (Path(directory) / 'audio.wav').write_bytes(b'audio')
    assert not Path(directory).exists()


@pytest.mark.parametrize('termination', ['exit', 'exception', 'sigterm', 'sigkill', 'sigkill-inherited-pipe'])
def test_ram_audio_directory_cleans_after_owner_dies(tmp_path, termination):
    script = '''
import importlib.util
import os
from pathlib import Path
import sys
import time
spec = importlib.util.spec_from_file_location('reward_audio', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._RAM_AUDIO_ROOT = sys.argv[2]
with module.ram_audio_directory(rank=0) as directory:
    (Path(directory) / 'audio.wav').write_bytes(b'audio')
    if sys.argv[3] == 'sigkill-inherited-pipe':
        worker_pid = os.fork()
        if worker_pid == 0:
            for fd in (0, 1, 2):
                os.close(fd)
            time.sleep(30)
            os._exit(0)
    print(directory, flush=True)
    if sys.argv[3] == 'sigkill-inherited-pipe':
        print(worker_pid, flush=True)
    if sys.argv[3] == 'exit':
        os._exit(0)
    if sys.argv[3] == 'exception':
        raise RuntimeError('owner failed')
    time.sleep(60)
'''
    process = subprocess.Popen(
        [sys.executable, '-c', script, reward_audio.__file__, str(tmp_path), termination],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker_pid = None
    try:
        assert select.select([process.stdout], [], [], 10)[0], 'Owner did not start'
        directory = process.stdout.readline().strip()
        assert directory, 'Owner failed before creating audio'
        if termination == 'sigkill-inherited-pipe':
            worker_pid = int(process.stdout.readline())
        if termination in ('sigterm', 'sigkill', 'sigkill-inherited-pipe'):
            assert (Path(directory) / 'audio.wav').exists()
            process.send_signal(signal.SIGTERM if termination == 'sigterm' else signal.SIGKILL)
        process.communicate(timeout=10)
        deadline = time.monotonic() + 10
        while Path(directory).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(directory).exists()
        assert not list(tmp_path.iterdir())
    finally:
        if worker_pid is not None:
            os.kill(worker_pid, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
