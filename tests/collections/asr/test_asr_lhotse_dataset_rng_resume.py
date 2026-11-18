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
"""
The ASR Lhotse datasets with random choices resume exactly through the configured stateful loader:
``use_stateful_dataloader=True`` enables their checkpointed generator, and a restored loader reproduces the
uninterrupted batches.
"""

import random
from unittest.mock import patch

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import DummyManifest
from omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_eou_label_lhotse import LhotseSpeechToTextBpeEOUDataset
from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex
from nemo.collections.asr.data.audio_to_text_lhotse_speaker import LhotseSpeechToTextSpkBpeDataset
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model

pytestmark = pytest.mark.unit

PROMPT_DICTIONARY = {'en-US': 0, 'de-DE': 3, 'auto': 6}


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> SentencePieceTokenizer:
    tmpdir = tmp_path_factory.mktemp("rng_resume_tokens")
    text_path = tmpdir / "text.txt"
    text_path.write_text("\n".join(map(chr, range(ord('a'), ord('z')))))
    model_path, _ = create_spt_model(
        text_path, vocab_size=32, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir)
    )
    return SentencePieceTokenizer(model_path)


def _cuts(kind: str) -> CutSet:
    """Eight one-second cuts with in-memory audio; speaker cuts have two distinguishable speakers."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=8, with_data=True)
    for cut in cuts:
        if kind == "speaker":
            cut.supervisions = [
                SupervisionSegment(
                    id=f"{cut.id}-0", recording_id=cut.recording_id, start=0, duration=0.5, text="first", speaker="0"
                ),
                SupervisionSegment(
                    id=f"{cut.id}-1",
                    recording_id=cut.recording_id,
                    start=0.5,
                    duration=0.5,
                    text="second",
                    speaker="1",
                ),
            ]
        else:
            cut.supervisions[0].language = "de-DE"
    return cuts


def _dataset(kind: str, tokenizer, pad_distribution: str = "uniform"):
    if kind == "eou":
        cfg = OmegaConf.create(
            {
                "check_tokenizer": False,
                "sample_rate": 16000,
                "random_padding": {
                    "prob": 0.9,
                    "max_total_duration": 6.0,
                    "min_post_pad_duration": 0.5,
                    "pad_distribution": pad_distribution,
                },
            }
        )
        return LhotseSpeechToTextBpeEOUDataset(cfg=cfg, tokenizer=tokenizer)
    if kind == "speaker":
        return LhotseSpeechToTextSpkBpeDataset(cfg={}, tokenizer=tokenizer)
    cfg = {"prompt_dictionary": PROMPT_DICTIONARY, "num_prompts": 8, "default_prompt_mode": "unified"}
    return LhotseSpeechToTextBpeDatasetWithPromptIndex(tokenizer=tokenizer, cfg=cfg)


def _loader(kind, tokenizer, mode, pad_distribution="uniform"):
    config = OmegaConf.create(
        {
            "batch_size": 2,
            "shuffle": True,
            "seed": 0,
            "shard_seed": 0,
            "num_workers": 0,
            "use_stateful_dataloader": True,
            "force_map_dataset": mode == "map",
            "force_iterable_dataset": mode == "iterable",
        }
    )
    dataset = _dataset(kind, tokenizer, pad_distribution)  # checkpointed randomness is not enabled by hand
    with patch(
        "nemo.collections.common.data.lhotse.dataloader.read_cutset_from_config", return_value=(_cuts(kind), False)
    ):
        return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=dataset)


def _summarize(kind, batch):
    if kind == "eou":
        tensors = [batch.audio_signal, batch.audio_lengths, batch.eou_targets, batch.eou_target_lengths]
    else:
        tensors = list(batch)
    return [t.tolist() for t in tensors]


def _random_part(kind, summary):
    """The part of a batch that the dataset draws at random."""
    # EOU: padded audio lengths; speaker: target-speaker masks; prompt: language prompt indices.
    return summary[1] if kind == "eou" else summary[4]


@pytest.mark.parametrize("mode", ["map", "iterable"])
@pytest.mark.parametrize(
    "kind,pad_distribution", [("eou", "uniform"), ("eou", "normal"), ("speaker", None), ("prompt", None)]
)
def test_stateful_loader_resumes_random_draws(tokenizer, kind, pad_distribution, mode):
    uninterrupted_loader = _loader(kind, tokenizer, mode, pad_distribution)
    uninterrupted = [_summarize(kind, batch) for batch in uninterrupted_loader]
    loader = _loader(kind, tokenizer, mode, pad_distribution)
    it = iter(loader)
    _ = next(it)
    _ = next(it)
    state = loader.state_dict()

    # A resumed process starts with different global RNG states; the fresh dataset is not reseeded.
    random.seed(1234)
    torch.manual_seed(1234)
    resumed_loader = _loader(kind, tokenizer, mode, pad_distribution)
    resumed_loader.load_state_dict(state)
    resumed = [_summarize(kind, batch) for batch in resumed_loader]

    assert len(uninterrupted) == 4
    assert resumed == uninterrupted[2:]
    # The loader enabled checkpointed randomness, and the draws vary between batches.
    random_parts = [_random_part(kind, summary) for summary in uninterrupted]
    assert len({str(part) for part in random_parts}) > 1
