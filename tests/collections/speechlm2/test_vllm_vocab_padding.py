# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vocabulary padding must support both checkpoint and IPC-refit tensor devices."""

import unittest

import torch


class TestVocabPadding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from nemo.collections.speechlm2.vllm.salm.audio import _pad_to_vocab_size
        except ImportError as exc:
            raise unittest.SkipTest(f"Speech vLLM dependencies unavailable: {exc}") from exc
        cls.pad = staticmethod(_pad_to_vocab_size)

    def test_cpu_values_and_dtype(self):
        for dtype in (torch.float32, torch.bfloat16):
            for shape in ((2,), (2, 3)):
                with self.subTest(dtype=dtype, shape=shape):
                    source = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
                    result = self.pad(source, 4)
                    self.assertEqual(result.shape, (4, *shape[1:]))
                    self.assertEqual(result.dtype, source.dtype)
                    self.assertEqual(result.device, source.device)
                    torch.testing.assert_close(result[:2], source, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(result[2:]).item(), 0)

    def test_non_cpu_device_preserved(self):
        for dtype in (torch.float32, torch.bfloat16):
            source = torch.empty((2, 3), device="meta", dtype=dtype)
            result = self.pad(source, 4)
            self.assertEqual(result.device, source.device)
            self.assertEqual(result.dtype, source.dtype)
            self.assertEqual(result.shape, (4, 3))

    def test_no_padding_returns_original_tensor(self):
        for device in ("cpu", "meta"):
            source = torch.empty((2, 3), device=device)
            for target_vocab in (1, 2):
                self.assertIs(self.pad(source, target_vocab), source)

    def test_cpu_preserved_under_other_default_device(self):
        source = torch.ones((2, 3), device="cpu")
        with torch.device("meta"):
            result = self.pad(source, 4)
        self.assertEqual(result.device, source.device)
        torch.testing.assert_close(result[:2], source, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(result[2:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
