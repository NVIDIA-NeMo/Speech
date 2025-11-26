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

from nemo.collections.speechlm2.parts.metrics import BLEU, WER, Intelligibility


def test_bleu():
    metric = BLEU(verbose=False)
    metric.update(
        name="dataset_1",
        refs=["a b c d e f g h i j k l", "m n o p r s t u v"],
        hyps=["a b c d e f g h i j k l", "m n o p r s t u v"],
    )
    metric.update(
        name="dataset_2",
        refs=["a b c"],
        hyps=["a b d"],
    )
    ans = metric.compute()
    assert ans["txt_bleu_dataset_1"] == 100.0
    assert ans["txt_bleu_dataset_2"] == 0.0
    assert ans["txt_bleu"] == 50.0  # average across datasets


def test_wer():
    metric = WER(verbose=False)
    metric.update(
        name="dataset_1",
        refs=["a b c d e f g h i j k l", "m n o p r s t u v"],
        hyps=["a b c d e f g h i j k l", "m n o p r s t u v"],
    )
    metric.update(
        name="dataset_2",
        refs=["a b c"],
        hyps=["a b d"],
    )
    ans = metric.compute()
    assert ans["wer_dataset_1"] == 0.0
    assert ans["wer_dataset_2"] == 1 / 3
    assert ans["wer"] == 1 / 6  # average across datasets


def test_intelligibility():
    metric = Intelligibility(pretrained_asr=None, verbose=False, reuse_asr_hyps=True)
    metric.update(
        name="dataset_1",
        refs=["a b c d e f g h i j k l", "m n o p r s t u v"],
        asr_hyps=["a b c d e f g h i j k l", "m n o p r s t u v"],
        pred_audio=None,
    )
    metric.update(
        name="dataset_2",
        refs=["a b c"],
        asr_hyps=["a b d"],
        pred_audio=None,
    )
    ans = metric.compute()
    # wer
    assert ans["wer_dataset_1"] == 0.0
    assert ans["wer_dataset_2"] == 1 / 3
    assert ans["wer"] == 1 / 6  # average across datasets
    # cer
    assert ans["cer_dataset_1"] == 0.0
    assert ans["cer_dataset_2"] == 1 / 5
