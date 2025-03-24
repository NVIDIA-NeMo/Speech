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

import random
from pathlib import Path

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from nemo.collections.asr.parts.submodules.ngram_lm import KenLMBatchedWrapper, NGramGPULanguageModel
from nemo.core.utils.optional_libs import KENLM_AVAILABLE, TRITON_AVAILABLE

DEVICES = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append('cuda')


@pytest.fixture(scope="module")
def n_gpu_lm(test_data_dir):
    kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
    vocab_size = 1024
    return NGramGPULanguageModel.from_arpa(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False)


@pytest.fixture(scope="module")
def kenlm_wrapper(test_data_dir):
    kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
    vocab_size = 1024
    return KenLMBatchedWrapper.from_file(lm_path=kenlm_model_path, vocab_size=vocab_size)


class TestNGramGPULanguageModel:
    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_load(self, test_data_dir):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        _ = NGramGPULanguageModel.from_arpa(kenlm_model_path, vocab_size=1024)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("bos", [True, False])
    def test_initial_states(
        self,
        n_gpu_lm: NGramGPULanguageModel,
        kenlm_wrapper: KenLMBatchedWrapper,
        bos: bool,
        batch_size: int,
        device: torch.device,
    ):
        n_gpu_lm = n_gpu_lm.to(device)
        init_states = n_gpu_lm.get_init_states(batch_size=batch_size, bos=bos)
        init_states_kenlm = kenlm_wrapper.get_init_states(batch_size=batch_size, bos=bos)
        scores_lm, _ = n_gpu_lm.advance(init_states)
        scores_ref, _ = kenlm_wrapper.advance(init_states_kenlm)
        assert torch.allclose(scores_lm, scores_ref.to(device))

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_triton_vs_pytorch_random_states(self, n_gpu_lm: NGramGPULanguageModel, batch_size=2, num_iterations=100):
        """Randomly initializes the states and compares the scores from Triton and PyTorch implementations."""
        torch.manual_seed(777)
        device = torch.device("cuda")
        n_gpu_lm = n_gpu_lm.to(device)
        for _ in tqdm(range(num_iterations)):
            start_state = random.randint(0, n_gpu_lm.num_states - 1)
            with torch.no_grad():
                scores1, states1 = n_gpu_lm._advance_pytorch(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
                scores2, states2 = n_gpu_lm._advance_triton(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
            assert (states1 == states2).all()
            assert torch.allclose(scores1, scores2)

    @pytest.mark.unit
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("bos", [True, False])
    def test_final(
        self, n_gpu_lm: NGramGPULanguageModel, kenlm_wrapper: KenLMBatchedWrapper, bos: bool, device: torch.device
    ):
        """Test final (eos) scores"""
        n_gpu_lm = n_gpu_lm.to(device)
        sentences = [
            [25, 70, 12],
            [58, 41, 186, 293, 306, 999, 163, 264, 689, 683, 999],
            [],  # empty sentence
        ]
        last_states = []
        for sentence in sentences:
            state = kenlm_wrapper.get_init_state(bos=bos)
            for label in sentence:
                _, state = kenlm_wrapper.advance_single(state=state, label=label)
            last_states.append(state)
        final_ref = kenlm_wrapper.get_final(states=last_states).to(device=device)

        last_states = []
        for sentence in sentences:
            states = n_gpu_lm.get_init_states(batch_size=1, bos=bos)
            for label in sentence:
                _, states = n_gpu_lm.advance(states=states)
                states = states[0, label].unsqueeze(0)
            last_states.append(states)
        final_lm = n_gpu_lm.get_final(states=torch.cat(last_states, dim=0))

        assert torch.allclose(final_lm, final_ref), "Final scores do not match"

    @pytest.mark.unit
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("bos", [True, False])
    @pytest.mark.parametrize("eos", [True, False])
    def test_sentences(
        self,
        n_gpu_lm: NGramGPULanguageModel,
        kenlm_wrapper: KenLMBatchedWrapper,
        bos: bool,
        eos: bool,
        device: torch.device,
    ):
        n_gpu_lm = n_gpu_lm.to(device)
        sentences = [
            [25, 70, 12],
            [58, 41, 186, 293, 306, 999, 163, 264, 689, 683, 999],
            [],  # empty sentence
        ]
        # non-batched
        for sentence in sentences:
            scores_ref = kenlm_wrapper.score_sentences([sentence], bos=bos, eos=eos).to(device)
            scores_lm = n_gpu_lm(
                labels=torch.LongTensor([sentence]).to(device),
                bos=bos,
                eos=eos,
            )
            assert torch.allclose(scores_ref, scores_lm), "Non-batched scores do not match"

        # batched
        scores_ref = kenlm_wrapper.score_sentences(sentences, bos=bos, eos=eos).to(device)
        scores_lm = n_gpu_lm(
            labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences], batch_first=True).to(device),
            labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences]).to(device),
            bos=bos,
            eos=eos,
        )
        assert torch.allclose(scores_lm, scores_ref), "Batched scores do not match"

    @pytest.mark.unit
    def test_save_load_nemo(self, tmp_path, test_data_dir):
        vocab_size = 1024
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        n_gpu_lm = NGramGPULanguageModel.from_arpa(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False)
        nemo_path = tmp_path / "ngram_lm.nemo"
        n_gpu_lm.save_to(f"{nemo_path}")
        n_gpu_lm_loaded = NGramGPULanguageModel.from_nemo(f"{nemo_path}", vocab_size=vocab_size)

        # arcs data
        assert torch.allclose(n_gpu_lm_loaded.arcs_weights, n_gpu_lm.arcs_weights)
        assert (n_gpu_lm_loaded.from_states == n_gpu_lm.from_states).all()
        assert (n_gpu_lm_loaded.to_states == n_gpu_lm.to_states).all()
        assert (n_gpu_lm_loaded.ilabels == n_gpu_lm.ilabels).all()

        # states data
        assert (n_gpu_lm_loaded.start_end_arcs == n_gpu_lm.start_end_arcs).all()
        assert (n_gpu_lm_loaded.state_order == n_gpu_lm.state_order).all()
        assert (n_gpu_lm_loaded.backoff_to_states == n_gpu_lm.backoff_to_states).all()
        assert torch.allclose(n_gpu_lm_loaded.backoff_weights, n_gpu_lm.backoff_weights)
        assert torch.allclose(n_gpu_lm_loaded.final_weights, n_gpu_lm.final_weights)

    @pytest.mark.unit
    def test_save_load_from_file(self, tmp_path, test_data_dir):
        vocab_size = 1024
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        n_gpu_lm = NGramGPULanguageModel.from_file(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False)
        nemo_path = tmp_path / "ngram_lm.nemo"
        n_gpu_lm.save_to(f"{nemo_path}")
        n_gpu_lm_loaded = NGramGPULanguageModel.from_file(f"{nemo_path}", vocab_size=vocab_size)

        # arcs data
        assert torch.allclose(n_gpu_lm_loaded.arcs_weights, n_gpu_lm.arcs_weights)
        assert (n_gpu_lm_loaded.from_states == n_gpu_lm.from_states).all()
        assert (n_gpu_lm_loaded.to_states == n_gpu_lm.to_states).all()
        assert (n_gpu_lm_loaded.ilabels == n_gpu_lm.ilabels).all()

        # states data
        assert (n_gpu_lm_loaded.start_end_arcs == n_gpu_lm.start_end_arcs).all()
        assert (n_gpu_lm_loaded.state_order == n_gpu_lm.state_order).all()
        assert (n_gpu_lm_loaded.backoff_to_states == n_gpu_lm.backoff_to_states).all()
        assert torch.allclose(n_gpu_lm_loaded.backoff_weights, n_gpu_lm.backoff_weights)
        assert torch.allclose(n_gpu_lm_loaded.final_weights, n_gpu_lm.final_weights)
