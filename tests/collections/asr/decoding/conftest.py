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

from pathlib import Path

import pytest

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest


@pytest.fixture(scope="session")
def an4_val_manifest_corrected(tmp_path_factory, test_data_dir):
    """
    Correct an4_val manifest audio filepaths, e.g.,
    "tests/data/asr/test/an4/wav/an440-mjgm-b.wav" -> test_data_dir / "test/an4/wav/an440-mjgm-b.wav"
    """
    an4_val_manifest_orig_path = Path(test_data_dir) / "asr/an4_val.json"
    an4_val_manifest_corrected_path = tmp_path_factory.mktemp("manifests") / "an4_val_corrected.json"
    an4_val_records = read_manifest(an4_val_manifest_orig_path)
    for record in an4_val_records:
        record["audio_filepath"] = record["audio_filepath"].replace(
            "tests/data/asr", str(an4_val_manifest_orig_path.resolve().parent)
        )
    write_manifest(an4_val_manifest_corrected_path, an4_val_records)
    return an4_val_manifest_corrected_path


@pytest.fixture(scope="session")
def an4_train_manifest_corrected(tmp_path_factory, test_data_dir):
    """
    Correct an4_train manifest audio filepaths, e.g.,
    "tests/data/asr/test/an4/wav/an440-mjgm-b.wav" -> test_data_dir / "test/an4/wav/an440-mjgm-b.wav"
    """
    an4_train_manifest_orig_path = Path(test_data_dir) / "asr/an4_train.json"
    an4_train_manifest_corrected_path = tmp_path_factory.mktemp("manifests") / "an4_train_corrected.json"
    an4_train_records = read_manifest(an4_train_manifest_orig_path)
    for record in an4_train_records:
        record["audio_filepath"] = record["audio_filepath"].replace(
            "tests/data/asr", str(an4_train_manifest_orig_path.resolve().parent)
        )
    write_manifest(an4_train_manifest_corrected_path, an4_train_records)
    return an4_train_manifest_corrected_path
