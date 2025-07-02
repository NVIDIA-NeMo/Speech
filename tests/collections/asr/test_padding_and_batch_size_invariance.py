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
import pytest
import torch.testing
from lhotse.testing.random import deterministic_rng

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder
from nemo.collections.asr.parts.preprocessing import FilterbankFeatures


@pytest.mark.parametrize("length", list(range(15950, 16050, 3)))
def test_preprocessor_invariant_to_padding(deterministic_rng, length):
    # Settings corresponding to Canary-1B features
    f = FilterbankFeatures(n_window_size=400, nfilt=128, pad_to=0).eval()

    # Test data:
    # * a1: 1s "audio"
    # * a2: 1s "audio" + 1s padding, keep length tensor unchanged
    a1 = torch.arange(0, length).unsqueeze(0) / 16000
    a1l = torch.tensor([length])

    a2 = torch.cat([a1, torch.zeros(1, 16000)], dim=1)
    a2l = a1l.clone()

    mels1, mels1l = f(a1, a1l)
    mels2, mels2l = f(a2, a2l)

    # Ideally, we'd have strictly identical results.
    # However, we observed depending on PyTorch build and environment,
    # Mel-spectrogram normalization tends to yield non-deterministic results;
    # specifically, in the computation of numerator in
    # nemo.collections.asr.parts.preprocessing.features.normalize_batch
    # where identical inputs lead up to +/- 2e-3 numerical differences.
    torch.testing.assert_close(mels1[..., :mels1l], mels2[..., :mels1l], atol=5e-2, rtol=0)


@pytest.mark.parametrize("length", [16000])
def test_canary_encoder_invariant_to_padding(deterministic_rng, length):
    preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=16000,
        normalize="per_feature",
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        features=128,
        n_fft=512,
        log=True,
        frame_splicing=1,
        dither=1e-5,
        pad_to=0,
        pad_value=0.0,
    ).eval()
    encoder = ConformerEncoder(
        feat_in=128,
        feat_out=-1,
        n_layers=17,
        d_model=512,
        subsampling="dw_striding",
        subsampling_factor=8,
        subsampling_conv_channels=256,
        causal_downsampling=True,
        reduction=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model="rel_pos",
        n_heads=8,
        att_context_size=[-1, -1],
        xscaling=False,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=9,
        conv_norm_type="batch_norm",
        conv_context_size=None,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.0,
        dropout_att=0.1,
    ).eval()

    # Test data:
    # * a1: 1s "audio"
    # * a2: 1s "audio" + 1s padding, keep length tensor unchanged
    a1 = torch.arange(0, length).unsqueeze(0) / 16000
    a1l = torch.tensor([length])

    a2 = torch.cat([a1, torch.zeros(1, 16000)], dim=1)
    a2l = a1l.clone()

    mels1, mels1l = preprocessor(input_signal=a1, length=a1l)
    mels2, mels2l = preprocessor(input_signal=a2, length=a2l)

    torch.testing.assert_close(mels1[..., :mels1l], mels2[..., :mels1l], atol=5e-4, rtol=0)

    # SUBSAMPLING MODULE NOT MISMATCHING
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = torch.tensor(output.detach().tolist())

        return hook

    for i, layer in enumerate(encoder.pre_encode.conv):
        if "ReLU" in str(layer):
            continue
        layer.register_forward_hook(get_activation(f"{i}:{layer}"))
    h1, h1l = encoder.pre_encode(mels1.transpose(1, 2), mels1l)
    inner1 = activation.copy()
    h2, h2l = encoder.pre_encode(mels2.transpose(1, 2), mels2l)
    inner2 = activation
    for k in inner1:
        torch.testing.assert_close(inner1[k], inner2[k][:, :, : inner1[k].shape[2]], atol=5e-5, rtol=0)

    torch.testing.assert_close(h1[:, :h1l], h2[:, :h1l])

    h1, h1l = encoder(audio_signal=mels1, length=mels1l)
    h2, h2l = encoder(audio_signal=mels2, length=mels2l)

    torch.testing.assert_close(h1[..., :h1l], h2[..., :h1l])


def test_conformer_inference_invariant_to_batch_size(deterministic_rng):
    model = ConformerEncoder(feat_in=128, n_layers=2, d_model=128, feat_out=128)
    model = model.eval()

    audio_signal_bs1, length_bs1 = model.input_example()
    h_bs1, h_length_bs1 = model(audio_signal=audio_signal_bs1, length=length_bs1)

    audio_signal_bs2 = audio_signal_bs1.repeat(2, 1, 1)
    length_bs2 = length_bs1.repeat(2)
    h_bs2, h_length_bs2 = model(audio_signal=audio_signal_bs2, length=length_bs2)

    torch.testing.assert_close(h_bs1, h_bs2[:1])
    torch.testing.assert_close(h_bs1, h_bs2[1:])
