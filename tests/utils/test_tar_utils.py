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

import ast
import io
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.parts.mixins.mixins import ASRBPEMixin
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import load_config, save_artifacts
from nemo.utils.notebook_utils import download_an4
from nemo.utils.tar_utils import TarPathTraversalError

REPO_ROOT = Path(__file__).parents[2]
RAW_TAR_EXTRACTION_ALLOWLIST = {REPO_ROOT / "nemo/utils/tar_utils.py"}


def _is_tarfile_open_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tarfile"
    )


def _add_tar_file(tar, name, data=b""):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_load_config_rejects_traversal_model_config(tmp_path):
    nemo_path = tmp_path / "malicious.nemo"
    with tarfile.open(nemo_path, "w:") as tar:
        _add_tar_file(tar, "../model_config.yaml", b"model:\n  value: 1\n")

    with pytest.raises(TarPathTraversalError):
        load_config(str(nemo_path))


def test_save_artifacts_rejects_traversal_artifact_path(tmp_path):
    nemo_path = tmp_path / "malicious.nemo"
    with tarfile.open(nemo_path, "w:") as tar:
        _add_tar_file(tar, "model_config.yaml", b"model:\n  value: 1\n")
        _add_tar_file(tar, "../tokenizer.model", b"payload")

    model = SimpleNamespace(
        cfg=OmegaConf.create({"tokenizer": {"library": "sentencepiece"}}),
        artifacts={"tokenizer.model": SimpleNamespace(path="nemo:../tokenizer.model")},
    )
    AppState().model_restore_path = str(nemo_path)

    with pytest.raises(TarPathTraversalError):
        save_artifacts(model, str(tmp_path / "output"))


def test_save_artifacts_rejects_absolute_artifact_path_before_rename(tmp_path):
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("sensitive")
    nemo_path = tmp_path / "malicious.nemo"
    with tarfile.open(nemo_path, "w:") as tar:
        _add_tar_file(tar, "nested/model_config.yaml", b"model:\n  value: 1\n")
        _add_tar_file(tar, f"nested/{secret_path.as_posix()}", b"payload")

    model = SimpleNamespace(
        cfg=OmegaConf.create({"tokenizer": {"library": "sentencepiece"}}),
        artifacts={"tokenizer.model": SimpleNamespace(path=f"nemo:{secret_path.as_posix()}")},
    )
    output_dir = tmp_path / "output"
    AppState().model_restore_path = str(nemo_path)

    with pytest.raises(TarPathTraversalError):
        save_artifacts(model, str(output_dir))

    assert secret_path.read_text() == "sensitive"
    assert not (output_dir / "tokenizer.model").exists()


def test_asr_tokenizer_rename_uses_member_basename():
    assert ASRBPEMixin._get_extracted_tokenizer_name("prefix_/tmp/tokenizer_vocab") == "vocab"


def test_download_an4_rejects_traversal_archive(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    an4_path = data_dir / "an4_sphere.tar.gz"
    with tarfile.open(an4_path, "w:gz") as tar:
        _add_tar_file(tar, "an4/etc/an4_train.transcription")
        _add_tar_file(tar, "an4/etc/an4_test.transcription")
        _add_tar_file(tar, "../nemo_poc_traversal_test", b"payload")

    with pytest.raises(TarPathTraversalError):
        download_an4(str(data_dir))

    assert not (tmp_path / "nemo_poc_traversal_test").exists()


def test_no_raw_tar_extraction_outside_safe_helper():
    raw_extractions = []
    python_files = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for python_file in python_files:
        if not python_file:
            continue
        path = REPO_ROOT / python_file.decode()
        if path in RAW_TAR_EXTRACTION_ALLOWLIST:
            continue

        tar_vars = set()
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _is_tarfile_open_call(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tar_vars.add(target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if _is_tarfile_open_call(item.context_expr) and isinstance(item.optional_vars, ast.Name):
                        tar_vars.add(item.optional_vars.id)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                node.func.attr in {"extract", "extractall"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in tar_vars
            ):
                raw_extractions.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert raw_extractions == []
