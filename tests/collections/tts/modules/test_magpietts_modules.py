# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
import torch

from nemo.collections.tts.modules.magpietts_modules import CodecHelper, FeatureMasking


def _set_seed():
    torch.manual_seed(42)


class TestFeatureMasking:

    @pytest.mark.unit
    @pytest.mark.parametrize('batch_size', [2, 5])
    @pytest.mark.parametrize('time', [5, 10])
    @pytest.mark.parametrize('hidden_size', [128, 256])
    def test_feature_masking_infer(self, batch_size, time, hidden_size):
        _set_seed()
        feature_masking = FeatureMasking(hidden_size=hidden_size)
        masked_emb = feature_masking.masked_emb[0, 0]
        inputs = torch.randn(size=(batch_size, time, hidden_size), dtype=torch.float32)
        mask = torch.randint(low=0, high=2, size=(batch_size, time), dtype=torch.bool)
        masked_tensor = feature_masking.infer(inputs=inputs, mask=mask)
        for i in range(batch_size):
            for j in range(time):
                output_val = masked_tensor[i, j]
                if mask[i, j]:
                    torch.testing.assert_close(actual=output_val, expected=masked_emb)
                else:
                    torch.testing.assert_close(actual=output_val, expected=inputs[i, j])

    @pytest.mark.unit
    def test_feature_masking_forward(self):
        _set_seed()
        hidden_size = 8
        feature_masking = FeatureMasking(hidden_size=hidden_size, mask_min=0.5, mask_max=0.5)
        torch.nn.init.constant_(feature_masking.masked_emb, 7.0)
        inputs = torch.randn(size=(4, 10, hidden_size), dtype=torch.float32)
        input_len = torch.tensor([10, 7, 2, 1])

        # Training: half of the valid timesteps (rounded down, at least one) are replaced by the mask embedding.
        feature_masking.train()
        outputs = feature_masking(inputs=inputs, input_len=input_len)
        masked = (outputs == 7.0).all(dim=-1)
        for i, length in enumerate(input_len.tolist()):
            assert not masked[i, length:].any()
            assert masked[i, :length].sum().item() == max(length // 2, 1)
        torch.testing.assert_close(actual=outputs[~masked], expected=inputs[~masked])

        # Eval: inputs pass through unchanged.
        feature_masking.eval()
        torch.testing.assert_close(actual=feature_masking(inputs=inputs, input_len=input_len), expected=inputs)


@pytest.mark.unit
@pytest.mark.parametrize('decode_batch_size', [0, 1, 2, 8])
@pytest.mark.parametrize('convert', [False, True])
def test_codec_decode_batches_preserve_waveforms_codes_and_order(decode_batch_size, convert):
    class Codec:
        def __init__(self):
            self.calls = []

        def eval(self):
            return self

        def decode(self, tokens, tokens_len):
            assert torch.get_default_dtype() == torch.float32
            self.calls.append(tuple(tokens.shape))
            audio = tokens[:, 0].float().repeat_interleave(2, dim=1)
            lengths = tokens_len * 2
            mask = torch.arange(audio.size(1))[None, :] < lengths[:, None]
            return audio * mask, lengths

    class Converter:
        def __init__(self):
            self.calls = 0

        def convert_new_to_original(self, audio_tokens, audio_lens):
            self.calls += 1
            return audio_tokens + 10

    codes = torch.arange(35).reshape(5, 1, 7)
    lengths = torch.tensor([3, 1, 0, 0, 7])
    converter = Converter() if convert else None
    codec = Codec()
    helper = CodecHelper(codec, converter, force_fp32=True, decode_batch_size=decode_batch_size)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        audio, audio_len, decoded_codes = helper.codes_to_audio(codes, lengths)
        assert torch.get_default_dtype() == torch.float16
    finally:
        torch.set_default_dtype(previous_dtype)

    expected_codes = codes + 10 if convert else codes
    expected_audio, expected_lens = Codec().decode(expected_codes, lengths)
    torch.testing.assert_close(audio, expected_audio)
    torch.testing.assert_close(audio_len, expected_lens)
    torch.testing.assert_close(decoded_codes, expected_codes)
    assert audio.dtype == torch.float32
    if converter is not None:
        assert converter.calls == 1
    if decode_batch_size == 1:
        assert codec.calls == [(1, 1, 3), (1, 1, 1), (1, 1, 7)]
    elif decode_batch_size == 2:
        assert codec.calls == [(2, 1, 3), (1, 1, 7)]
    else:
        assert codec.calls == [(5, 1, 7)]


@pytest.mark.unit
def test_codec_decode_batches_skip_empty_chunks_and_reject_negative_limit():
    class Codec:
        def eval(self):
            return self

        def decode(self, **kwargs):
            pytest.fail('Zero-length sub-batches must not reach the codec decoder')

    helper = CodecHelper(Codec(), decode_batch_size=2)
    audio, lengths, codes = helper.codes_to_audio(
        torch.zeros(3, 1, 4, dtype=torch.long), torch.zeros(3, dtype=torch.long)
    )
    assert audio.shape == (3, 0) and audio.dtype == torch.float32
    assert lengths.tolist() == [0, 0, 0] and codes.shape == (3, 1, 4)
    with pytest.raises(ValueError, match='non-negative'):
        CodecHelper(Codec(), decode_batch_size=-1)
