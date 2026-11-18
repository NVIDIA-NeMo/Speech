# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
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

"""CPU regressions for nightly model-support test inputs and checkpoint preparation."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.unit
@pytest.mark.parametrize("audio_locator_id,pad_id", [(1, 0), (0, 1), (2, 0)])
def test_canary_training_batch_has_one_placeholder(audio_locator_id, pad_id, monkeypatch):
    from tests.e2e_nightly import test_model_support_nvidia__canary_qwen_2_5b as canary

    class Model:
        sampling_rate = 16
        audio_locator_tag_id = audio_locator_id
        text_pad_id = pad_id
        text_vocab_size = 8

        def train(self):
            pass

        def training_step(self, batch, batch_idx):
            input_ids = batch["input_ids"]
            assert (input_ids == self.audio_locator_tag_id).sum().item() == batch["audios"].shape[0]
            assert not (input_ids == self.text_pad_id).any()
            assert ((input_ids >= 0) & (input_ids < self.text_vocab_size)).all()
            assert batch["loss_mask"].any()
            return {"loss": torch.tensor(1.0, requires_grad=True)}

    monkeypatch.setattr(canary, "_load_model", Model)
    monkeypatch.setattr(canary, "_DEVICE", torch.device("cpu"))
    canary.test_model_training_step()


@pytest.mark.unit
def test_magpie_prefetch_downloads_each_release(monkeypatch):
    checkpoints = importlib.import_module("tests.e2e_nightly.magpie_release_checkpoints")
    calls = []
    monkeypatch.setattr(checkpoints, "hf_hub_download", lambda **kwargs: calls.append(kwargs))

    checkpoints.prefetch_release_checkpoints()

    assert calls == [
        {
            "repo_id": "nvidia/magpie_tts_multilingual_357m",
            "filename": "magpie_tts_multilingual_357m.nemo",
            "revision": revision,
        }
        for revision in ("v2512", "v2602", "v2607")
    ]


@pytest.mark.unit
def test_magpie_prefetch_propagates_download_failure(monkeypatch):
    checkpoints = importlib.import_module("tests.e2e_nightly.magpie_release_checkpoints")
    calls = []

    def fail_download(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("download failed")

    monkeypatch.setattr(checkpoints, "hf_hub_download", fail_download)
    with pytest.raises(RuntimeError, match="download failed"):
        checkpoints.prefetch_release_checkpoints()
    assert len(calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("download_exit_code", [0, 7])
def test_magpie_wrapper_prefetches_online_before_offline_tests(tmp_path, download_exit_code):
    # Stub the external commands to exercise the real wrapper without downloading models.
    command = tmp_path / "record_command"
    command.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "name = Path(sys.argv[0]).name\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps({'name': name, 'args': sys.argv[1:], "
        "'offline': os.environ['HF_HUB_OFFLINE'], 'hf_home': os.environ['HF_HOME']}) + '\\n')\n"
        "sys.exit(int(os.environ['DOWNLOAD_EXIT_CODE']) if name == 'python' else 0)\n"
    )
    command.chmod(0o755)
    for name in ("python", "coverage"):
        (tmp_path / name).symlink_to(command)
    log = tmp_path / "commands.jsonl"
    cache = str(tmp_path / "hf-cache")
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        COMMAND_LOG=str(log),
        DOWNLOAD_EXIT_CODE=str(download_exit_code),
        HF_HOME=cache,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    root = Path(__file__).resolve().parents[3]
    wrapper = root / "tests/e2e_nightly/L2_Model_Support_nvidia__magpie_tts_multilingual_357m.sh"
    result = subprocess.run(["bash", str(wrapper)], cwd=root, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]

    assert calls[0] == {
        "name": "python",
        "args": ["-m", "tests.e2e_nightly.magpie_release_checkpoints"],
        "offline": "0",
        "hf_home": cache,
    }
    assert result.returncode == download_exit_code, result.stderr
    if download_exit_code:
        assert len(calls) == 1  # Failed preparation must stop before pytest.
    else:
        assert len(calls) == 2
        assert calls[1]["name"] == "coverage"
        assert calls[1]["offline"] == "1"
        assert calls[1]["hf_home"] == cache
        assert "--with_downloads" in calls[1]["args"]
