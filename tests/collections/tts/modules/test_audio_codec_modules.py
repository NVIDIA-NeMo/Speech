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

from nemo.collections.tts.modules.audio_codec_modules import Conv1dNorm, ConvTranspose1dNorm, get_down_sample_padding
from nemo.collections.tts.modules.encodec_modules import GroupResidualVectorQuantizer, ResidualVectorQuantizer


class TestAudioCodecModules:
    def setup_class(self):
        self.in_channels = 8
        self.out_channels = 16
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


class TestResidualVectorQuantizer:
    def setup_class(self):
        """Setup common members
        """
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
