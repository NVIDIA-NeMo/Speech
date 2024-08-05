# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.modules.audio_codec_modules import (
    CodecActivation,
    Conv1dNorm,
    ConvTranspose1dNorm,
    FiniteScalarQuantizer,
    GroupFiniteScalarQuantizer,
    HiFiGANDecoder,
    MelSpectrogramProcessor,
    MultiBandMelEncoder,
    ResidualBlock,
    ResNetEncoder,
    get_down_sample_padding,
)
from nemo.collections.tts.modules.encodec_modules import GroupResidualVectorQuantizer, ResidualVectorQuantizer


class TestAudioCodecModules:
    def setup_class(self):
        self.in_channels = 8
        self.out_channels = 16
        self.filters = 32
        self.batch_size = 2
        self.len1 = 4
        self.len2 = 8
        self.max_len = 10
        self.kernel_size = 3

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d(self):
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([self.len1, self.len2], dtype=torch.int32)

        conv = Conv1dNorm(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size)
        out = conv(inputs=inputs, input_len=lengths)

        assert out.shape == (self.batch_size, self.out_channels, self.max_len)
        assert torch.all(out[0, :, : self.len1] != 0.0)
        assert torch.all(out[0, :, self.len1 :] == 0.0)
        assert torch.all(out[1, :, : self.len2] != 0.0)
        assert torch.all(out[1, :, self.len2 :] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d_downsample(self):
        stride = 2
        out_len = self.max_len // stride
        out_len_1 = self.len1 // stride
        out_len_2 = self.len2 // stride
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([out_len_1, out_len_2], dtype=torch.int32)

        padding = get_down_sample_padding(kernel_size=self.kernel_size, stride=stride)
        conv = Conv1dNorm(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
        )
        out = conv(inputs=inputs, input_len=lengths)

        assert out.shape == (self.batch_size, self.out_channels, out_len)
        assert torch.all(out[0, :, :out_len_1] != 0.0)
        assert torch.all(out[0, :, out_len_1:] == 0.0)
        assert torch.all(out[1, :, :out_len_2] != 0.0)
        assert torch.all(out[1, :, out_len_2:] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d_transpose_upsample(self):
        stride = 2
        out_len = self.max_len * stride
        out_len_1 = self.len1 * stride
        out_len_2 = self.len2 * stride
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([out_len_1, out_len_2], dtype=torch.int32)

        conv = ConvTranspose1dNorm(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=stride
        )
        out = conv(inputs=inputs, input_len=lengths)

        assert out.shape == (self.batch_size, self.out_channels, out_len)
        assert torch.all(out[0, :, :out_len_1] != 0.0)
        assert torch.all(out[0, :, out_len_1:] == 0.0)
        assert torch.all(out[1, :, :out_len_2] != 0.0)
        assert torch.all(out[1, :, out_len_2:] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_residual_block(self):
        lengths = torch.tensor([self.len1, self.len2], dtype=torch.int32)
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        inputs = mask_sequence_tensor(tensor=inputs, lengths=lengths)

        res_block = ResidualBlock(channels=self.in_channels, filters=self.filters)
        out = res_block(inputs=inputs, input_len=lengths)

        assert out.shape == (self.batch_size, self.in_channels, self.max_len)
        assert torch.all(out[0, :, : self.len1] != 0.0)
        assert torch.all(out[0, :, self.len1 :] == 0.0)
        assert torch.all(out[1, :, : self.len2] != 0.0)
        assert torch.all(out[1, :, self.len2 :] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_hifigan_decoder(self):
        up_sample_rates = [4, 4, 2, 2]
        up_sample_total = 64
        lengths = torch.tensor([self.len1, self.len2], dtype=torch.int32)
        out_len_1 = self.len1 * up_sample_total
        out_len_2 = self.len2 * up_sample_total
        out_len_max = self.max_len * up_sample_total

        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        inputs = mask_sequence_tensor(tensor=inputs, lengths=lengths)

        decoder = HiFiGANDecoder(
            input_dim=self.in_channels, base_channels=self.filters, up_sample_rates=up_sample_rates
        )
        out, out_len = decoder(inputs=inputs, input_len=lengths)

        assert out_len[0] == out_len_1
        assert out_len[1] == out_len_2
        assert out.shape == (self.batch_size, out_len_max)
        assert torch.all(out[0, :out_len_1] != 0.0)
        assert torch.all(out[0, out_len_1:] == 0.0)
        assert torch.all(out[1, :out_len_2] != 0.0)
        assert torch.all(out[1, out_len_2:] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_resnet_encoder(self):
        lengths = torch.tensor([self.len1, self.len2], dtype=torch.int32)
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        inputs = mask_sequence_tensor(tensor=inputs, lengths=lengths)

        res_net = ResNetEncoder(in_channels=self.in_channels, out_channels=self.out_channels)
        out = res_net(inputs=inputs, input_len=lengths)

        assert out.shape == (self.batch_size, self.out_channels, self.max_len)
        assert torch.all(out[0, :, : self.len1] != 0.0)
        assert torch.all(out[0, :, self.len1 :] == 0.0)
        assert torch.all(out[1, :, : self.len2] != 0.0)
        assert torch.all(out[1, :, self.len2 :] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_multiband_mel_encoder(self):
        mel_dim = 10
        win_length = 16
        hop_length = 10
        mel_bands = [(0, 4), (4, 7), (7, 10)]
        max_len = 100
        len1 = 40
        len2 = 80
        out_dim = len(mel_bands) * self.out_channels
        lengths = torch.tensor([len1, len2], dtype=torch.int32)
        out_len_1 = len1 // hop_length
        out_len_2 = len2 // hop_length
        out_len_max = max_len // hop_length

        audio = torch.rand([self.batch_size, max_len])
        audio = mask_sequence_tensor(tensor=audio, lengths=lengths)

        mel_processor = MelSpectrogramProcessor(
            mel_dim=mel_dim, sample_rate=100, win_length=win_length, hop_length=hop_length
        )
        encoder = MultiBandMelEncoder(
            mel_bands=mel_bands,
            mel_processor=mel_processor,
            out_channels=self.out_channels,
            hidden_channels=self.filters,
            filters=self.filters,
        )
        out, out_len = encoder(audio=audio, audio_len=lengths)

        assert out_len[0] == out_len_1
        assert out_len[1] == out_len_2
        assert out.shape == (self.batch_size, out_dim, out_len_max)
        assert torch.all(out[0, :, :out_len_1] != 0.0)
        assert torch.all(out[0, :, out_len_1:] == 0.0)
        assert torch.all(out[1, :, :out_len_2] != 0.0)
        assert torch.all(out[1, :, out_len_2:] == 0.0)


class TestResidualVectorQuantizer:
    def setup_class(self):
        """Setup common members"""
        self.batch_size = 2
        self.max_len = 20
        self.codebook_size = 256
        self.codebook_dim = 64
        self.num_examples = 10

    @pytest.mark.unit
    @pytest.mark.parametrize('num_codebooks', [1, 4])
    def test_rvq_eval(self, num_codebooks: int):
        """Simple test to confirm that the RVQ module can be instantiated and run,
        and that forward produces the same result as encode-decode.
        """
        # instantiate and set in eval mode
        rvq = ResidualVectorQuantizer(num_codebooks=num_codebooks, codebook_dim=self.codebook_dim)
        rvq.eval()

        for i in range(self.num_examples):
            inputs = torch.randn([self.batch_size, self.codebook_dim, self.max_len])
            input_len = torch.tensor([self.max_len] * self.batch_size, dtype=torch.int32)

            # apply forward
            dequantized_fw, indices_fw, commit_loss = rvq(inputs=inputs, input_len=input_len)

            # make sure the commit loss is zero
            assert commit_loss == 0.0, f'example {i}: commit_loss is {commit_loss}, expected 0.0'

            # encode-decode
            indices_enc = rvq.encode(inputs=inputs, input_len=input_len)
            dequantized_dec = rvq.decode(indices=indices_enc, input_len=input_len)

            # make sure the results are the same
            torch.testing.assert_close(indices_enc, indices_fw, msg=f'example {i}: indices mismatch')
            torch.testing.assert_close(dequantized_dec, dequantized_fw, msg=f'example {i}: dequantized mismatch')

    @pytest.mark.unit
    @pytest.mark.parametrize('num_groups', [1, 2, 4])
    @pytest.mark.parametrize('num_codebooks', [1, 4])
    def test_group_rvq_eval(self, num_groups: int, num_codebooks: int):
        """Simple test to confirm that the group RVQ module can be instantiated and run,
        and that forward produces the same result as encode-decode.
        """
        if num_groups > num_codebooks:
            # Expected to fail if num_groups is lager than the total number of codebooks
            with pytest.raises(ValueError):
                _ = GroupResidualVectorQuantizer(
                    num_codebooks=num_codebooks, num_groups=num_groups, codebook_dim=self.codebook_dim
                )
        else:
            # Test inference with group RVQ
            # instantiate and set in eval mode
            grvq = GroupResidualVectorQuantizer(
                num_codebooks=num_codebooks, num_groups=num_groups, codebook_dim=self.codebook_dim
            )
            grvq.eval()

            for i in range(self.num_examples):
                inputs = torch.randn([self.batch_size, self.codebook_dim, self.max_len])
                input_len = torch.tensor([self.max_len] * self.batch_size, dtype=torch.int32)

                # apply forward
                dequantized_fw, indices_fw, commit_loss = grvq(inputs=inputs, input_len=input_len)

                # make sure the commit loss is zero
                assert commit_loss == 0.0, f'example {i}: commit_loss is {commit_loss}, expected 0.0'

                # encode-decode
                indices_enc = grvq.encode(inputs=inputs, input_len=input_len)
                dequantized_dec = grvq.decode(indices=indices_enc, input_len=input_len)

                # make sure the results are the same
                torch.testing.assert_close(indices_enc, indices_fw, msg=f'example {i}: indices mismatch')
                torch.testing.assert_close(dequantized_dec, dequantized_fw, msg=f'example {i}: dequantized mismatch')

                # apply individual RVQs and make sure the results are the same
                inputs_grouped = inputs.chunk(num_groups, dim=1)
                dequantized_fw_grouped = dequantized_fw.chunk(num_groups, dim=1)
                indices_fw_grouped = indices_fw.chunk(num_groups, dim=0)

                for g in range(num_groups):
                    dequantized, indices, _ = grvq.rvqs[g](inputs=inputs_grouped[g], input_len=input_len)
                    torch.testing.assert_close(
                        dequantized, dequantized_fw_grouped[g], msg=f'example {i}: dequantized mismatch for group {g}'
                    )
                    torch.testing.assert_close(
                        indices, indices_fw_grouped[g], msg=f'example {i}: indices mismatch for group {g}'
                    )


class TestCodecActivation:
    def setup_class(self):
        self.batch_size = 2
        self.in_channels = 4
        self.max_len = 4

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_snake(self):
        """
        Test for snake activation function execution.
        """
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        snake = CodecActivation('snake', channels=self.in_channels)
        out = snake(x=inputs)
        assert out.shape == (self.batch_size, self.in_channels, self.max_len)


class TestFiniteScalarQuantizer:
    def setup_class(self):
        """Setup common members"""
        self.batch_size = 2
        self.max_len = 20
        self.num_examples = 10

    @pytest.mark.unit
    @pytest.mark.parametrize('num_levels', [[2, 3], [8, 5, 5]])
    def test_fsq_eval(self, num_levels: list):
        """Simple test to confirm that the FSQ module can be instantiated and run,
        and that forward produces the same result as encode-decode.
        """
        fsq = FiniteScalarQuantizer(num_levels=num_levels)

        for i in range(self.num_examples):
            inputs = torch.randn([self.batch_size, fsq.codebook_dim, self.max_len])
            input_len = torch.tensor([self.max_len] * self.batch_size, dtype=torch.int32)

            # apply forward
            dequantized_fw, indices_fw = fsq(inputs=inputs, input_len=input_len)

            assert dequantized_fw.max() <= 1.0, f'example {i}: dequantized_fw.max() is {dequantized_fw.max()}'
            assert dequantized_fw.min() >= -1.0, f'example {i}: dequantized_fw.min() is {dequantized_fw.min()}'

            # encode-decode
            indices_enc = fsq.encode(inputs=inputs, input_len=input_len)
            dequantized_dec = fsq.decode(indices=indices_enc, input_len=input_len)

            # make sure the results are the same
            torch.testing.assert_close(indices_enc, indices_fw, msg=f'example {i}: indices mismatch')
            torch.testing.assert_close(dequantized_dec, dequantized_fw, msg=f'example {i}: dequantized mismatch')

    @pytest.mark.unit
    def test_fsq_output(self):
        """Simple test to make sure the output of FSQ is correct for a single setup.

        To re-generate test vectors:
        ```
        num_examples, max_len = 5, 8
        inputs = torch.randn([num_examples, fsq.codebook_dim, max_len])
        input_len = torch.tensor([max_len] * num_examples, dtype=torch.int32)
        dequantized, indices = fsq(inputs=inputs, input_len=input_len)
        ```
        """
        num_levels = [3, 4]
        fsq = FiniteScalarQuantizer(num_levels=num_levels)

        # inputs
        inputs = torch.tensor(
            [
                [
                    [0.1483, -0.3855, -0.3715, -0.5913, -0.2212, -0.4226, -0.4864, -1.6069],
                    [-0.5519, -0.5307, -0.5995, -1.9675, -0.4439, 0.3938, -0.5636, -0.3655],
                ],
                [
                    [0.5184, 1.4028, 0.1553, -0.2324, 1.0363, -0.4981, -0.1203, -1.0335],
                    [-0.1567, -0.2274, 0.0424, -0.0819, -0.2122, -2.1851, -1.5035, -1.2237],
                ],
                [
                    [0.9497, 0.8510, -1.2021, 0.3299, -0.2388, 0.8445, 2.2129, -2.3383],
                    [1.5331, 0.0399, -0.7676, -0.4715, -0.5713, 0.8761, -0.9755, -0.7479],
                ],
                [
                    [1.7243, -1.2146, -0.1969, 1.9261, 0.1109, 0.4028, 0.1240, -0.0994],
                    [-0.3304, 2.1239, 0.1004, -1.4060, 1.1463, -0.0557, -0.5856, -1.2441],
                ],
                [
                    [2.3743, -0.1421, -0.4548, 0.6320, -0.2640, -0.3967, -2.5694, 0.0493],
                    [0.3409, 0.2366, -0.0309, -0.7652, 0.3484, -0.8419, 0.9079, -0.9929],
                ],
            ]
        )

        input_len = torch.tensor([8, 8, 8, 8, 8], dtype=torch.int32)

        # expected output
        dequantized_expected = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000, -1.0000, 0.0000, 0.0000, 0.0000, -1.0000],
                    [-0.5000, -0.5000, -0.5000, -1.0000, -0.5000, 0.0000, -0.5000, -0.5000],
                ],
                [
                    [0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, -1.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.0000, -1.0000, -1.0000],
                ],
                [
                    [1.0000, 1.0000, -1.0000, 0.0000, 0.0000, 1.0000, 1.0000, -1.0000],
                    [0.5000, 0.0000, -0.5000, -0.5000, -0.5000, 0.5000, -0.5000, -0.5000],
                ],
                [
                    [1.0000, -1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.5000, 0.0000, -1.0000, 0.5000, 0.0000, -0.5000, -1.0000],
                ],
                [
                    [1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, -1.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, -0.5000, 0.0000, -0.5000, 0.5000, -0.5000],
                ],
            ]
        )

        indices_expected = torch.tensor(
            [
                [
                    [4, 4, 4, 0, 4, 7, 4, 3],
                    [7, 8, 7, 7, 8, 1, 1, 0],
                    [11, 8, 3, 4, 4, 11, 5, 3],
                    [8, 9, 7, 2, 10, 7, 4, 1],
                    [8, 7, 7, 5, 7, 4, 9, 4],
                ]
            ],
            dtype=torch.int32,
        )

        # test
        dequantized, indices = fsq(inputs=inputs, input_len=input_len)
        torch.testing.assert_close(dequantized, dequantized_expected, msg=f'dequantized mismatch')
        torch.testing.assert_close(indices, indices_expected, msg=f'indices mismatch')

    @pytest.mark.unit
    @pytest.mark.parametrize('num_groups', [1, 2, 4])
    @pytest.mark.parametrize('num_levels_per_group', [[2, 3], [8, 5, 5]])
    def test_group_fsq_eval(self, num_groups: int, num_levels_per_group: int):
        """Simple test to confirm that the group FSQ module can be instantiated and run,
        and that forward produces the same result as encode-decode.
        """
        # Test inference with group FSQ
        # instantiate
        gfsq = GroupFiniteScalarQuantizer(num_groups=num_groups, num_levels_per_group=num_levels_per_group)

        for i in range(self.num_examples):
            inputs = torch.randn([self.batch_size, gfsq.codebook_dim, self.max_len])
            input_len = torch.tensor([self.max_len] * self.batch_size, dtype=torch.int32)

            # apply forward
            dequantized_fw, indices_fw = gfsq(inputs=inputs, input_len=input_len)

            # encode-decode
            indices_enc = gfsq.encode(inputs=inputs, input_len=input_len)
            dequantized_dec = gfsq.decode(indices=indices_enc, input_len=input_len)

            # make sure the results are the same
            torch.testing.assert_close(indices_enc, indices_fw, msg=f'example {i}: indices mismatch')
            torch.testing.assert_close(dequantized_dec, dequantized_fw, msg=f'example {i}: dequantized mismatch')

            # apply individual FSQs and make sure the results are the same
            inputs_grouped = inputs.chunk(num_groups, dim=1)
            dequantized_fw_grouped = dequantized_fw.chunk(num_groups, dim=1)
            indices_fw_grouped = indices_fw.chunk(num_groups, dim=0)

            for g in range(num_groups):
                dequantized, indices = gfsq.fsqs[g](inputs=inputs_grouped[g], input_len=input_len)
                torch.testing.assert_close(
                    dequantized, dequantized_fw_grouped[g], msg=f'example {i}: dequantized mismatch for group {g}'
                )
                torch.testing.assert_close(
                    indices, indices_fw_grouped[g], msg=f'example {i}: indices mismatch for group {g}'
                )
