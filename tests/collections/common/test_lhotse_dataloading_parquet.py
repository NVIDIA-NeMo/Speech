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

import io
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

# We need this try/except because the test environment might not have pyarrow/pandas
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAVE_PARQUET = True
except ImportError:
    HAVE_PARQUET = False

from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config


@pytest.fixture
def parquet_dataset(tmp_path):
    """
    Creates a dummy Parquet dataset with 5 audio files embedded as bytes.
    Returns the path to the .parquet file.
    """
    if not HAVE_PARQUET:
        pytest.skip("PyArrow not installed, skipping Parquet tests.")

    data = []
    sr = 16000
    for i in range(5):
        # Generate 1 second of random noise
        audio = np.random.uniform(-1, 1, sr).astype(np.float32)

        # Write to bytes using SoundFile
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        audio_bytes = buf.getvalue()

        data.append(
            {
                "id": f"audio_{i}",
                "audio_col": audio_bytes,  # Custom column name to test mapping
                "text_col": f"This is sentence {i}",
                "duration": 1.0,
                "lang": "en",
            }
        )

    df = pd.DataFrame(data)
    path = tmp_path / "test_data.parquet"

    # Save using pyarrow engine
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

    return str(path)


@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
def test_read_parquet_manifest(parquet_dataset):
    """
    Test that we can read the parquet file using the registry mechanism.
    """
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "parquet",
                    "manifest_filepath": parquet_dataset,
                    "audio_field": "audio_col",
                    "text_field": "text_col",
                }
            ],
            "batch_size": 2,
            "force_finite": True,
        }
    )

    # This calls read_dataset_config internally
    cuts, is_tarred = read_cutset_from_config(config)

    assert is_tarred is True

    # Iterate and check data
    cuts_list = list(cuts)
    assert len(cuts_list) == 5

    first_cut = cuts_list[0]
    assert first_cut.id == "audio_0"
    assert first_cut.supervisions[0].text == "This is sentence 0"

    # Check if we can actually decode audio
    audio = first_cut.load_audio()
    assert audio.shape[1] == 16000  # 1 second at 16khz


@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
def test_read_parquet_sharding(parquet_dataset):
    """
    Test that we can handle a list of files (sharding) using LazyIteratorChain.
    """
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "parquet",
                    "manifest_filepath": [parquet_dataset, parquet_dataset],
                    "audio_field": "audio_col",
                    "text_field": "text_col",
                    "shuffle": True,
                    "shard_seed": 42,
                }
            ],
            "force_finite": True,
        }
    )

    cuts, is_tarred = read_cutset_from_config(config)
    cuts_list = list(cuts)

    # 5 items per file * 2 files = 10 items
    assert len(cuts_list) == 10
