# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import pytest
import torch

from nemo.collections.asr.models import ASRModel


class TestASRSubsamplingConvChunking:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_forward(self):
        asr_model = ASRModel.from_pretrained("stt_en_fastconformer_ctc_large")
        asr_model = asr_model.eval()
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        len = 512

        input_signal_batch1 = torch.randn(size=(1, len), device=asr_model.device)
        length_batch1 = torch.randint(low=161, high=500, size=[1], device=asr_model.device)

        input_signal_batch4 = torch.randn(size=(4, len), device=asr_model.device)
        length_batch4 = torch.randint(low=161, high=500, size=[4], device=asr_model.device)

        with torch.no_grad():
            # regular inference
            logprobs_batch1_nosplit, _, _ = asr_model.forward(
                input_signal=input_signal_batch1, input_signal_length=length_batch1
            )
            logprobs_batch4_nosplit, _, _ = asr_model.forward(
                input_signal=input_signal_batch4, input_signal_length=length_batch4
            )

            # force chunking to 2
            asr_model.change_subsampling_conv_chunking_factor(subsampling_conv_chunking_factor=2)

            # chunked inference by channels as batch is 1
            logprobs_batch1_split, _, _ = asr_model.forward(
                input_signal=input_signal_batch1, input_signal_length=length_batch1
            )
            # chunked inference by batch as it is 4 [> 1]
            logprobs_batch4_split, _, _ = asr_model.forward(
                input_signal=input_signal_batch4, input_signal_length=length_batch4
            )

        diff = torch.mean(torch.abs(logprobs_batch1_split - logprobs_batch1_nosplit))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logprobs_batch4_split - logprobs_batch4_nosplit))
        assert diff <= 1e-6
