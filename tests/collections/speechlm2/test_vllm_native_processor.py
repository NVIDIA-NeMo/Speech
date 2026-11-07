# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise SALM through vLLM's native processor, without GPU or model weights."""

import unittest
from types import SimpleNamespace

import torch

try:
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache
    from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
    from vllm.multimodal.processing import BaseMultiModalProcessor, ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
except ImportError as exc:
    raise unittest.SkipTest("vLLM processor dependencies are not installed") from exc

from nemo.collections.speechlm2.vllm.salm.audio import (
    NeMoSpeechLMDummyInputsBuilder,
    NeMoSpeechLMMultiModalProcessor,
    NeMoSpeechLMProcessingInfo,
)


@unittest.skipIf(hasattr(BaseMultiModalProcessor, "_call_hf_processor"), "Tests the vLLM 0.29 native hook")
class TestNativeProcessor(unittest.TestCase):
    def setUp(self):
        mm_config = SimpleNamespace(
            enable_mm_embeds=False,
            mm_hasher_algorithm="sha256",
            mm_processor_cache_gb=0.01,
            get_limit_per_prompt=lambda modality: 8,
        )
        self.model_config = SimpleNamespace(
            model="salm-processor-contract-test",
            get_multimodal_config=lambda: mm_config,
        )
        ctx = SimpleNamespace(
            model_config=self.model_config,
            get_mm_config=lambda: mm_config,
            get_hf_config=lambda: SimpleNamespace(encoder_chunk_size_seconds=None),
            get_tokenizer=_Tokenizer,
        )
        self.info = NeMoSpeechLMProcessingInfo(ctx)

    def processor(self, cache=None):
        return NeMoSpeechLMMultiModalProcessor(self.info, NeMoSpeechLMDummyInputsBuilder(self.info), cache=cache)

    def apply(self, processor, lengths, ids=None, capture=None):
        items = MultiModalDataItems({"audio": AudioProcessorItems([torch.zeros(n) for n in lengths])})
        kwargs = {} if capture is None else {"capture_ctc_timestamps": capture}
        inputs = ProcessorInputs([101, 102, 42, 103, 42, 104], items, {"audio": ids} if ids else None, kwargs)
        return processor.apply(inputs, TimingContext(enabled=False))

    def test_native_audio_packing_and_ctc_opt_in(self):
        for capture in (None, False, True):
            with self.subTest(capture=capture):
                items = MultiModalDataItems({"audio": AudioProcessorItems([torch.zeros(16000), torch.ones(32000)])})
                kwargs = {} if capture is None else {"capture_ctc_timestamps": capture}
                result = self.processor()._apply_hf_processor_main(items, kwargs)
                self.assertNotIn("input_ids", result)
                self.assertEqual(result["audio_signal_length"].tolist(), [16000, 32000])
                self.assertEqual(result["capture_ctc_timestamps"].tolist(), [bool(capture), bool(capture)])
                self.assertTrue(torch.equal(result["audio_signal"][1], torch.ones(32000)))

    def test_empty_audio_preserves_passthrough(self):
        sentinel = torch.tensor([7])
        items = MultiModalDataItems({"audio": AudioProcessorItems([]), "extra": _PassthroughItems(sentinel)})
        result = self.processor()._apply_hf_processor_main(items, {})
        self.assertEqual(set(result), {"sentinel"})
        self.assertIs(result["sentinel"], sentinel)

    def test_full_apply_preserves_text_before_audio_and_exact_spans(self):
        result = self.apply(self.processor(), [16000, 32000])
        self.assertEqual(result["prompt_token_ids"], [101, 102] + [42] * 13 + [103] + [42] * 25 + [104])
        self.assertEqual([(p.offset, p.length) for p in result["mm_placeholders"]["audio"]], [(2, 13), (16, 25)])
        self.assertEqual(
            [item["audio_signal_length"].data.item() for item in result["mm_kwargs"]["audio"]], [16000, 32000]
        )

    def test_text_only_full_apply(self):
        inputs = ProcessorInputs([101, 102, 103], MultiModalDataItems({}))
        result = self.processor().apply(inputs, TimingContext(enabled=False))
        self.assertEqual(result["prompt_token_ids"], [101, 102, 103])
        self.assertFalse(result["mm_kwargs"])
        self.assertFalse(result["mm_placeholders"])

    def test_full_cache_hits_and_missing_subset_preserve_order(self):
        processor = self.processor(MultiModalProcessorOnlyCache(self.model_config))
        first = self.apply(processor, [16000, 32000], ["first", "second"])
        cached = self.apply(processor, [16000, 32000], ["first", "second"])
        self.assertEqual(cached["prompt_token_ids"], first["prompt_token_ids"])
        partial = self.apply(processor, [32000, 48000], ["second", "third"])
        self.assertEqual(partial["prompt_token_ids"], [101, 102] + [42] * 25 + [103] + [42] * 38 + [104])
        self.assertEqual(
            [item["audio_signal_length"].data.item() for item in partial["mm_kwargs"]["audio"]], [32000, 48000]
        )


class _Tokenizer:
    def get_vocab(self):
        return {"<|audio|>": 42}

    def encode(self, text, add_special_tokens=False):
        if text != "<|audio|>":
            raise AssertionError("Renderer-supplied prompt IDs must not be retokenized")
        return [42]


class _PassthroughItems:
    def __init__(self, sentinel):
        self.sentinel = sentinel

    def get_processor_data(self):
        return {}

    def get_passthrough_data(self):
        return {"sentinel": self.sentinel}


if __name__ == "__main__":
    unittest.main()
