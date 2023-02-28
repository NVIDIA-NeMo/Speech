# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os

import pytest
import torch

from nemo.collections.asr.models.msdd_models import NeuralDiarizer


class TestNeuralDiarizerInference:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "device",
        [
            torch.device("cpu"),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for test.',),
            ),
        ],
    )
    @pytest.mark.parametrize("num_speakers", [None, 1])
    @pytest.mark.parametrize("max_num_speakers", [4])
    def test_diar_inference(self, tmpdir, test_data_dir, device, num_speakers, max_num_speakers):
        """
        Test to ensure diarization inference works correctly.
            - Ensures multiple audio files can be diarized sequentially
            - Ensures both CPU/CUDA is supported
            - Ensures that max speaker and num speaker are set correctly
            - Ensures temporary directory is emptied at the end of diarization
            - Sanity check to ensure outputs from diarization are reasonable
        """
        audio_filenames = ['an22-flrp-b.wav', 'an90-fbbh-b.wav']
        audio_paths = [os.path.join(test_data_dir, "asr", "train", "an4", "wav", fp) for fp in audio_filenames]

        diarizer = NeuralDiarizer.from_pretrained(model_name='diar_msdd_telephonic').to(device)

        out_dir = os.path.join(tmpdir, 'diarize_inference/')

        assert diarizer.msdd_model.device.type == device.type
        assert diarizer._speaker_model.device.type == device.type
        for audio_path in audio_paths:
            annotation = diarizer(
                audio_path, num_speakers=num_speakers, max_speakers=max_num_speakers, out_dir=out_dir
            )

            # assert max speakers has been set up correctly
            assert diarizer.clustering_embedding.clus_diar_model._cluster_params.max_num_speakers == max_num_speakers

            if num_speakers:
                assert diarizer._cfg.diarizer.clustering.parameters.oracle_num_speakers

            # assert all temporary files are cleaned up
            assert len(os.listdir(out_dir)) == 0

            # assert only 1 speaker & segment
            assert len(annotation.labels()) == 1
            assert len(list(annotation.itersegments())) == 1
