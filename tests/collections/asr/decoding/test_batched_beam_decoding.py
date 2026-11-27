# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import copy
import glob
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from kaldialign import edit_distance
from omegaconf import open_dict
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint
from nemo.collections.asr.parts.submodules.ctc_beam_decoding import BeamBatchedCTCInfer
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import BeamBatchedRNNTInfer
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding, RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.tdt_beam_decoding import BeamBatchedTDTInfer
from nemo.collections.asr.parts.submodules.tdt_frame_sync_batched_computer import FrameSyncBatchedTDTComputer
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.utils import numba_utils
from nemo.core.utils.cuda_python_utils import skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__
from tests.collections.asr.decoding.utils import load_audio

RNNT_MODEL = "stt_en_conformer_transducer_small"
CTC_MODEL = "nvidia/stt_en_conformer_ctc_small"
TDT_MODEL = "nvidia/stt_en_fastconformer_tdt_large"
MAX_SAMPLES = 10

DEVICES = [torch.device("cpu")]

if torch.cuda.is_available():
    # The `gpu` marker puts the CUDA cases in the GPU CI jobs (see tests/functional_tests/_gpu_unit_test_selection.sh).
    DEVICES.append(pytest.param(torch.device('cuda'), marks=pytest.mark.gpu))

# malsd_batch implementations: the PyTorch one on each device (None), and each CUDA graphs mode
DEVICES_AND_CUDA_GRAPHS_MODES = [pytest.param(torch.device("cpu"), None, id="cpu")]
if torch.cuda.is_available():
    DEVICES_AND_CUDA_GRAPHS_MODES += [
        pytest.param(torch.device("cuda"), mode, marks=pytest.mark.gpu, id=f"cuda-{mode or 'torch'}")
        for mode in [None, "no_graphs", "no_while_loops", "full_graph"]
    ]

# frame_sync_batch: without CUDA graphs on each device, and with CUDA graphs
DEVICES_AND_FRAME_SYNC_CUDA_GRAPHS = [pytest.param(torch.device("cpu"), False, id="cpu")]
if torch.cuda.is_available():
    DEVICES_AND_FRAME_SYNC_CUDA_GRAPHS += [
        pytest.param(torch.device("cuda"), allow_cuda_graphs, marks=pytest.mark.gpu, id=name)
        for allow_cuda_graphs, name in [(False, "cuda"), (True, "cuda-graphs")]
    ]

# random transducer for decoding unit tests: labels (without blank), encoder output, max symbols per frame
TINY_VOCAB_SIZE = 4
TINY_ENCODED_LENGTHS = [12, 7, 9, 3]
TINY_MAX_SYMBOLS = 3
TINY_DURATIONS = {"rnnt": None, "tdt": [0, 1, 2]}

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


# available audio filename fixtures
@pytest.fixture(scope="module")
def test_audio_filenames(test_data_dir):
    return tuple(glob.glob(os.path.join(test_data_dir, "asr", "test", "an4", "wav", "*.wav")))


# model fixtures
@pytest.fixture(scope="module")
def rnnt_model():
    model = ASRModel.from_pretrained(model_name=RNNT_MODEL, map_location="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def tdt_model():
    model = ASRModel.from_pretrained(model_name=TDT_MODEL, map_location="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def ctc_model():
    model = ASRModel.from_pretrained(model_name=CTC_MODEL, map_location="cpu")
    model.eval()
    return model


# encoder output fixtures
@pytest.fixture(scope="module")
def get_rnnt_encoder_output(rnnt_model, test_audio_filenames):
    encoder_output, encoded_lengths = get_transducer_model_encoder_output(
        test_audio_filenames, MAX_SAMPLES, rnnt_model
    )
    return encoder_output, encoded_lengths


@pytest.fixture(scope="module")
def get_tdt_encoder_output(tdt_model, test_audio_filenames):
    encoder_output, encoded_lengths = get_transducer_model_encoder_output(test_audio_filenames, MAX_SAMPLES, tdt_model)
    return encoder_output, encoded_lengths


@pytest.fixture(scope="module")
def get_ctc_output(ctc_model, test_audio_filenames):
    encoder_output, encoded_lengths = get_ctc_model_output(test_audio_filenames, MAX_SAMPLES, ctc_model)
    return encoder_output, encoded_lengths


@pytest.fixture(scope="module")
def kenlm_model_path(tmp_path_factory, test_data_dir):
    lm_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
    assert os.path.exists(lm_path), f"LM file not found: {lm_path}"
    lm_nemo_path = tmp_path_factory.mktemp("lm") / f"{lm_path.name}.nemo"
    NGramGPULanguageModel.from_file(lm_path, vocab_size=1024).save_to(f"{lm_nemo_path}")
    return f"{lm_nemo_path}"


def get_transducer_model_encoder_output(
    test_audio_filenames,
    num_samples: int,
    model: ASRModel,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    audio_filepaths = test_audio_filenames[:num_samples]

    with torch.no_grad():
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        model.eval()

        all_inputs, all_lengths = [], []
        for audio_file in tqdm(audio_filepaths, desc="Loading audio files"):
            audio_tensor, _ = load_audio(audio_file)
            all_inputs.append(audio_tensor)
            all_lengths.append(torch.tensor(audio_tensor.shape[0], dtype=torch.int64))

        input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(device=device, dtype=dtype)
        length_batch = torch.tensor(all_lengths, dtype=torch.int64).to(device)

        encoded_outputs, encoded_length = model(input_signal=input_batch, input_signal_length=length_batch)

    return encoded_outputs, encoded_length


def get_ctc_model_output(
    test_audio_filenames,
    num_samples: int,
    model: ASRModel,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    audio_filepaths = test_audio_filenames[:num_samples]

    with torch.no_grad():
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        model.eval()

        all_inputs, all_lengths = [], []
        for audio_file in tqdm(audio_filepaths, desc="Loading audio files"):
            audio_tensor, _ = load_audio(audio_file)
            all_inputs.append(audio_tensor)
            all_lengths.append(torch.tensor(audio_tensor.shape[0], dtype=torch.int64))

        input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(device=device, dtype=dtype)
        length_batch = torch.tensor(all_lengths, dtype=torch.int64).to(device)

        log_probs, encoded_length, _ = model(input_signal=input_batch, input_signal_length=length_batch)

    return log_probs, encoded_length


def print_unit_test_info(strategy, batch_size, beam_size, allow_cuda_graphs, device):
    print(
        f"""Beam search algorithm: {strategy},
                Batch size: {batch_size},
                Beam size: {beam_size},
                Cuda Graphs: {allow_cuda_graphs},
                Decoding device: {device}
            """
    )


def check_res_best_hyps(num_samples, hyps):
    assert type(hyps) == list
    assert type(hyps[0]) == rnnt_utils.Hypothesis

    assert len(hyps) == num_samples

    assert all(
        [
            hasattr(hyps[hyp_idx], "y_sequence")
            and hasattr(hyps[hyp_idx], "score")
            and hasattr(hyps[hyp_idx], "timestamp")
            for hyp_idx in range(num_samples)
        ]
    )


def print_res_best_hyps(hyps):
    for hyp_idx, hyp in enumerate(hyps):
        print("Sample: ", hyp_idx)
        print("Decoded text: ", hyp.text)
        print("Score: ", hyp.score)
        print("Transcript", hyp.y_sequence)
        print("Timesteps", hyp.timestamp)
        print()


def check_res_nbest_hyps(num_samples, batch_nbest_hyps):
    assert type(batch_nbest_hyps) == list
    assert type(batch_nbest_hyps[0]) == rnnt_utils.NBestHypotheses

    assert len(batch_nbest_hyps) == num_samples

    for idx in range(num_samples):
        assert all(
            [
                hasattr(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx], "y_sequence")
                and hasattr(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx], "score")
                and hasattr(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx], "timestamp")
                for hyp_idx in range(len(batch_nbest_hyps[idx].n_best_hypotheses))
            ]
        )

        # Empty transcript (blank-only beam) is valid; y_sequence and timestamp must stay aligned.
        assert all(
            len(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx].y_sequence)
            == len(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx].timestamp)
            for hyp_idx in range(len(batch_nbest_hyps[idx].n_best_hypotheses))
        )


def print_res_nbest_hyps(batch_nbest_hyps):
    for batch_idx, nbest_hyps in enumerate(batch_nbest_hyps):
        print(f"Batch idx: {batch_idx}")
        for idx, hyp in enumerate(nbest_hyps):
            print(f"Hyp index: {idx + 1}")
            print("Text: ", hyp.text)
            print("Score: ", hyp.score)
            print("Transcripts: ", hyp.y_sequence)
            print("Timesteps: ", hyp.timestamp)
            print()


def decode_text_from_hypotheses(hyps, model):
    if isinstance(model, EncDecCTCModel):
        return model.decoding.decode_hypothesis(hyps, fold_consecutive=False)
    else:
        return model.decoding.decode_hypothesis(hyps)


def decode_text_from_nbest_hypotheses(hyps, model):
    if isinstance(model, EncDecCTCModel):
        return [
            model.decoding.decode_hypothesis(nbest_hyp.n_best_hypotheses, fold_consecutive=False) for nbest_hyp in hyps
        ]
    else:
        return [model.decoding.decode_hypothesis(nbest_hyp.n_best_hypotheses) for nbest_hyp in hyps]


class LabelScoresFusionModel(torch.nn.Module):
    """Fusion model with a single state that gives label `k` the score `scores[k]` after any history."""

    def __init__(self, scores: torch.Tensor):
        super().__init__()
        self.register_buffer("scores", scores)

    def compatible_with_cuda_graphs(self) -> bool:
        return True

    def get_init_states(self, batch_size: int, bos: bool = True) -> torch.Tensor:
        return torch.zeros([batch_size], dtype=torch.long, device=self.scores.device)

    def advance(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scores.repeat(states.shape[0], 1)
        return scores, torch.zeros_like(scores, dtype=torch.long)


def get_tiny_transducer(durations: list[int] | None = None) -> tuple[RNNTDecoder, RNNTJoint]:
    """Random prediction and joint networks over TINY_VOCAB_SIZE labels; a TDT joint when `durations` are given."""
    torch.manual_seed(0)
    decoder = RNNTDecoder(prednet={'pred_hidden': 8, 'pred_rnn_layers': 1}, vocab_size=TINY_VOCAB_SIZE)
    joint = RNNTJoint(
        {'encoder_hidden': 8, 'pred_hidden': 8, 'joint_hidden': 8, 'activation': 'relu'},
        TINY_VOCAB_SIZE,
        num_extra_outputs=len(durations) if durations else 0,
    )
    decoder.freeze()
    joint.freeze()
    return decoder, joint


def decode_tiny_transducer(
    durations: list[int] | None, device: torch.device, cuda_graphs_mode: str | None, **beam_kwargs
) -> list[rnnt_utils.Hypothesis]:
    """Best malsd_batch hypotheses (scores not normalized) of the tiny transducer on fixed random encoder output."""
    if cuda_graphs_mode == "full_graph":
        skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()
    decoder, joint = (module.to(device) for module in get_tiny_transducer(durations))
    generator = torch.Generator().manual_seed(1)
    encoder_output = torch.randn(len(TINY_ENCODED_LENGTHS), 8, max(TINY_ENCODED_LENGTHS), generator=generator)
    beam_kwargs = dict(
        blank_index=TINY_VOCAB_SIZE,
        beam_size=4,
        search_type="malsd_batch",
        max_symbols_per_step=TINY_MAX_SYMBOLS,
        score_norm=False,
        allow_cuda_graphs=cuda_graphs_mode is not None,
        **beam_kwargs,
    )
    if durations:
        decoding = BeamBatchedTDTInfer(decoder, joint, durations=durations, **beam_kwargs)
    else:
        decoding = BeamBatchedRNNTInfer(decoder, joint, **beam_kwargs)
    if cuda_graphs_mode is not None:
        decoding.decoding_computer.force_cuda_graphs_mode(mode=cuda_graphs_mode)
    with torch.no_grad():
        return decoding(
            encoder_output=encoder_output.to(device),
            encoded_lengths=torch.tensor(TINY_ENCODED_LENGTHS, device=device),
        )[0]


def assert_same_hypotheses(actual: list[rnnt_utils.Hypothesis], expected: list[rnnt_utils.Hypothesis]):
    assert [torch.as_tensor(hyp.y_sequence).tolist() for hyp in actual] == [
        torch.as_tensor(hyp.y_sequence).tolist() for hyp in expected
    ]
    assert [hyp.score for hyp in actual] == pytest.approx([hyp.score for hyp in expected], abs=1e-4)


def get_tiny_encoder_output() -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed random encoder output [B, D, T] of the tiny transducer, and its lengths."""
    generator = torch.Generator().manual_seed(1)
    encoder_output = torch.randn(len(TINY_ENCODED_LENGTHS), 8, max(TINY_ENCODED_LENGTHS), generator=generator)
    return encoder_output, torch.tensor(TINY_ENCODED_LENGTHS)


def get_frame_sync_decoding(
    device: torch.device, allow_cuda_graphs: bool, durations: list[int] = TINY_DURATIONS["tdt"], **beam_kwargs
) -> BeamBatchedTDTInfer:
    """frame_sync_batch decoding of the tiny TDT transducer, returning n-best lists with scores not normalized."""
    decoder, joint = (module.to(device) for module in get_tiny_transducer(durations))
    beam_kwargs = dict(
        blank_index=TINY_VOCAB_SIZE,
        beam_size=4,
        search_type="frame_sync_batch",
        max_symbols_per_step=TINY_MAX_SYMBOLS,
        score_norm=False,
        return_best_hypothesis=False,
        allow_cuda_graphs=allow_cuda_graphs,
        **beam_kwargs,
    )
    return BeamBatchedTDTInfer(decoder, joint, durations=durations, **beam_kwargs)


def decode_tiny_encoder_output(decoding: BeamBatchedTDTInfer, device: torch.device) -> list:
    encoder_output, encoded_lengths = get_tiny_encoder_output()
    with torch.no_grad():
        return decoding(encoder_output=encoder_output.to(device), encoded_lengths=encoded_lengths.to(device))[0]


def frame_sync_reference(
    decoder: RNNTDecoder,
    joint: RNNTJoint,
    encoder_output: torch.Tensor,
    durations: list[int],
    beam_size: int,
    max_symbols: int,
    beam_beta: float = 0.0,
    lm_scores: torch.Tensor | None = None,
    lm_alpha: float = 0.0,
    pruning_mode: str = "late",
    blank_lm_score_mode: str = "no_score",
) -> list[tuple[list[int], float]]:
    """
    Frame-synchronous TDT beam search over one utterance [T, D], a hypothesis at a time: n-best (labels, score).

    Hypotheses wait in the bucket of their frame; arrivals with the same labels in a bucket are merged (scores
    log-added, the best one's state kept), and every bucket that received arrivals keeps its best `beam_size`.
    """
    blank = joint.num_classes_with_blank - joint.num_extra_outputs - 1
    encoder_output = joint.project_encoder(encoder_output)
    length = encoder_output.shape[0]  # bucket of the finished hypotheses
    num_durations = len(durations)
    # bucket -> {labels: (score, decoder state before the last label, last label, zero-duration labels in a row)}
    start = (0.0, decoder.initialize_state(encoder_output.new_zeros(1)), blank, 0)
    buckets = {0 if length > 0 else length: {(): start}}
    while frames := [frame for frame in buckets if frame < length]:
        t = min(frames)
        arrivals = []
        for labels, (score, state, last_label, num_zero) in buckets.pop(t).items():
            if num_zero >= max_symbols:
                arrivals.append((t + 1, labels, (score, state, last_label, 0)))
                continue
            label = torch.tensor([[last_label]], device=encoder_output.device)
            output, new_state = decoder.predict(label, state, add_sos=False, batch_size=1)[:2]
            logits = joint.joint_after_projection(encoder_output[t].view(1, 1, -1), joint.project_prednet(output))
            logits = logits.view(-1).float()
            log_probs = logits[:-num_durations].log_softmax(-1)
            duration_log_probs = logits[-num_durations:].log_softmax(-1)
            label_scores, blank_score = log_probs[:blank] + beam_beta, log_probs[blank]
            fusion_scores = torch.zeros_like(label_scores)
            if lm_scores is not None:
                fusion_scores = lm_alpha * lm_scores.to(label_scores.device)
                if blank_lm_score_mode == "lm_weighted_full":
                    fusion_scores = fusion_scores + lm_alpha * torch.log1p(-blank_score.exp().clamp(max=1 - 1e-2))
                    blank_score = blank_score * (1 + lm_alpha)
            num_labels = min(beam_size, blank)
            if pruning_mode == "late":
                top_labels = (label_scores + fusion_scores).topk(num_labels).indices
            else:
                top_labels = label_scores.topk(num_labels).indices
            label_scores = label_scores + fusion_scores

            nonzero = [i for i in duration_log_probs.argsort(descending=True).tolist() if durations[i] > 0]
            for i in nonzero[:beam_size]:
                blank_child = (score + float(blank_score + duration_log_probs[i]), state, last_label, 0)
                arrivals.append((t + durations[i], labels, blank_child))
            top_durations = duration_log_probs.topk(min(beam_size, num_durations)).indices
            pairs = [
                (float(label_scores[k] + duration_log_probs[i]), int(k), int(i))
                for k in top_labels
                for i in top_durations
            ]
            for pair_score, k, i in sorted(pairs, reverse=True)[:beam_size]:
                label_child = (score + pair_score, new_state, k, 0 if durations[i] > 0 else num_zero + 1)
                arrivals.append((t + durations[i], labels + (k,), label_child))

        for frame, labels, hyp in arrivals:
            bucket = buckets.setdefault(min(frame, length), {})
            if labels in bucket:
                best = max(bucket[labels], hyp, key=lambda h: h[0])
                hyp = (float(np.logaddexp(bucket[labels][0], hyp[0])),) + best[1:]
            bucket[labels] = hyp
        for frame in {min(frame, length) for frame, _, _ in arrivals}:
            best = sorted(buckets[frame].items(), key=lambda item: item[1][0], reverse=True)[:beam_size]
            buckets[frame] = dict(best)
    finished = sorted(buckets.get(length, {}).items(), key=lambda item: item[1][0], reverse=True)
    return [(list(labels), hyp[0]) for labels, hyp in finished]


class TestRNNTDecoding:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "maes_batch", "allow_cuda_graphs": False},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_beam_decoding_return_best_hypothesis(
        self, test_audio_filenames, rnnt_model, get_rnnt_encoder_output, beam_config, device, batch_size, beam_size
    ):
        num_samples = min(batch_size, len(test_audio_filenames))
        model = rnnt_model.to(device)
        encoder_output, encoded_lengths = get_rnnt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "maes_batch", "allow_cuda_graphs": False},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.gpu
    def test_rnnt_beam_decoding_return_nbest(
        self, test_audio_filenames, rnnt_model, get_rnnt_encoder_output, beam_config, device, beam_size, batch_size
    ):
        device = torch.device("cuda")
        num_samples = min(batch_size, len(test_audio_filenames))
        model = rnnt_model.to(device)
        encoder_output, encoded_lengths = get_rnnt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=False,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            batch_nbest_hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_nbest_hyps(num_samples, batch_nbest_hyps)
            batch_nbest_hyps = decode_text_from_nbest_hypotheses(batch_nbest_hyps, model)
            print_res_nbest_hyps(batch_nbest_hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "maes_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("pruning_mode", ["late", "early"])
    @pytest.mark.parametrize("blank_lm_score_mode", ["no_score", "lm_weighted_full"])
    @pytest.mark.gpu
    def test_rnnt_beam_decoding_kenlm(
        self,
        kenlm_model_path,
        test_audio_filenames,
        rnnt_model,
        get_rnnt_encoder_output,
        beam_config,
        device,
        batch_size,
        beam_size,
        pruning_mode,
        blank_lm_score_mode,
    ):
        device = torch.device("cuda")

        num_samples = min(batch_size, len(test_audio_filenames))
        model = rnnt_model.to(device)
        encoder_output, encoded_lengths = get_rnnt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        vocab_size = model.tokenizer.vocab_size

        fusion_models = [NGramGPULanguageModel.from_file(lm_path=kenlm_model_path, vocab_size=vocab_size)]
        fusion_models_alpha = [0.3]

        decoding = BeamBatchedRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            pruning_mode=pruning_mode,
            blank_lm_score_mode=blank_lm_score_mode,
            fusion_models=fusion_models,
            fusion_models_alpha=fusion_models_alpha,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)


class TestTDTDecoding:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "frame_sync_batch", "allow_cuda_graphs": False},
            {"search_type": "frame_sync_batch", "allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_beam_decoding_return_best_hypothesis(
        self, test_audio_filenames, tdt_model, get_tdt_encoder_output, beam_config, device, batch_size, beam_size
    ):
        num_samples = min(batch_size, len(test_audio_filenames))
        model = tdt_model.to(device)
        encoder_output, encoded_lengths = get_tdt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "frame_sync_batch", "allow_cuda_graphs": False},
            {"search_type": "frame_sync_batch", "allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.gpu
    def test_tdt_beam_decoding_return_nbest(
        self, test_audio_filenames, tdt_model, get_tdt_encoder_output, beam_config, device, beam_size, batch_size
    ):
        device = torch.device("cuda")
        num_samples = min(batch_size, len(test_audio_filenames))
        model = tdt_model.to(device)
        encoder_output, encoded_lengths = get_tdt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=False,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            batch_nbest_hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_nbest_hyps(num_samples, batch_nbest_hyps)
            batch_nbest_hyps = decode_text_from_nbest_hypotheses(batch_nbest_hyps, model)
            print_res_nbest_hyps(batch_nbest_hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "frame_sync_batch", "allow_cuda_graphs": False},
            {"search_type": "frame_sync_batch", "allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("pruning_mode", ["late", "early"])
    @pytest.mark.parametrize("blank_lm_score_mode", ["lm_weighted_full", "no_score"])
    @pytest.mark.gpu
    def test_tdt_beam_decoding_kenlm(
        self,
        kenlm_model_path,
        test_audio_filenames,
        tdt_model,
        get_tdt_encoder_output,
        beam_config,
        device,
        batch_size,
        beam_size,
        pruning_mode,
        blank_lm_score_mode,
    ):
        device = torch.device("cuda")

        num_samples = min(batch_size, len(test_audio_filenames))
        model = tdt_model.to(device)
        encoder_output, encoded_lengths = get_tdt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size

        fusion_models = [NGramGPULanguageModel.from_file(lm_path=kenlm_model_path, vocab_size=vocab_size)]
        fusion_models_alpha = [0.3]

        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            pruning_mode=pruning_mode,
            blank_lm_score_mode=blank_lm_score_mode,
            fusion_models=fusion_models,
            fusion_models_alpha=fusion_models_alpha,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
    @pytest.mark.gpu
    def test_tdt_frame_sync_batch_cuda_graphs(self, kenlm_model_path, tdt_model, get_tdt_encoder_output):
        """frame_sync_batch with an n-gram LM finds the same n-best lists with CUDA graphs as without them."""
        device = torch.device("cuda")
        model = tdt_model.to(device)
        encoder_output, encoded_lengths = get_tdt_encoder_output
        encoder_output, encoded_lengths = encoder_output.to(device), encoded_lengths.to(device)
        durations = list(model.to_config_dict()["model_defaults"]["tdt_durations"])
        vocab_size = model.tokenizer.vocab_size

        nbest_hyps = {}
        for allow_cuda_graphs in [False, True]:
            decoding = BeamBatchedTDTInfer(
                model.decoder,
                model.joint,
                blank_index=vocab_size,
                durations=durations,
                beam_size=4,
                search_type="frame_sync_batch",
                score_norm=True,
                return_best_hypothesis=False,
                fusion_models=[NGramGPULanguageModel.from_file(lm_path=kenlm_model_path, vocab_size=vocab_size)],
                fusion_models_alpha=[0.3],
                beam_beta=0.5,
                allow_cuda_graphs=allow_cuda_graphs,
            )
            with torch.no_grad():
                nbest_hyps[allow_cuda_graphs] = decoding(
                    encoder_output=encoder_output, encoded_lengths=encoded_lengths
                )[0]

        for with_graphs, without_graphs in zip(nbest_hyps[True], nbest_hyps[False]):
            assert_same_hypotheses(with_graphs.n_best_hypotheses, without_graphs.n_best_hypotheses)


@pytest.mark.gpu
class TestTransducerCudaGraphBeamDecoding:
    """
    Tests CudaGraphs implementations from Transducer models (RNN-T and TDT)
    """

    @pytest.mark.with_downloads
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
    @pytest.mark.parametrize("force_mode", ["no_graphs", "no_while_loops", "full_graph"])
    @pytest.mark.parametrize("model_type", ["rnnt", "tdt"])
    def test_stated_stateless(self, test_audio_filenames, rnnt_model, tdt_model, model_type, force_mode: str):
        """
        Compares pure Pytorch and with three modes of statefull implementations for double floating point precision.
            1. Pure pytorch, but statefull implementation: no_graphs
            2. With CudaGrpahs: no_while_loops and full_graph.
        """
        if force_mode == "full_graph":
            skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

        batch_size = 16
        device = torch.device("cuda")
        model = rnnt_model.to(device) if model_type == "rnnt" else tdt_model.to(device)
        decoding_config = copy.deepcopy(model.cfg.decoding)

        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypothesis"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False

        model.change_decoding_strategy(decoding_config)

        actual_hypotheses = model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        actual_transcripts = [[hyp.text for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_scores = [[hyp.score for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_timestamps = [[hyp.timestamp for hyp in actual_beam] for actual_beam in actual_hypotheses]

        # transcribe with use implementation with cuda graphs
        decoding_config["beam"]["allow_cuda_graphs"] = True
        model.change_decoding_strategy(decoding_config)
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode(mode=force_mode)

        cudagraph_hypotheses = model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        cudagraph_transcripts = [[hyp.text for hyp in cudagraphs_beam] for cudagraphs_beam in cudagraph_hypotheses]
        cudagraph_scores = [[hyp.score for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]
        cudagraph_timestamps = [[hyp.timestamp for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]

        for batch_idx in range(min(batch_size, len(test_audio_filenames))):
            assert len(actual_transcripts[batch_idx]) == len(cudagraph_transcripts[batch_idx])
            assert cudagraph_scores[batch_idx] == pytest.approx(
                actual_scores[batch_idx], abs=1e-2
            ), f"Scores mismatch for batch_idx {batch_idx}"
            assert (
                cudagraph_timestamps[batch_idx] == actual_timestamps[batch_idx]
            ), f"Timestamps mismatch for batch_idx {batch_idx}"

            for actual, fast in zip(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx]):
                ref_words = actual.split()
                hyp_words = fast.split()
                wer = edit_distance(ref_words, hyp_words)['total'] / max(len(ref_words), 1)

                assert wer <= 1e-3, "Cuda graph beam decoder should match original decoder implementation."

                if actual != fast:
                    print("Erroneous samples in batch:", batch_idx)
                    print("Original transcript:", actual)
                    print("New transcript:", fast)

    @pytest.mark.with_downloads
    @pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()), reason="CUDA decoder can run only on CUDA"
    )
    @pytest.mark.parametrize("model_type", ["rnnt", "tdt"])
    def test_stated_stateless_bf16(self, test_audio_filenames, rnnt_model, tdt_model, model_type):
        """
        Checks that we are able to run without errors all decodings in bfloat16.
        Computational errors accumulate, so just checking if algorithms run without errors
        """
        batch_size = 16
        device = torch.device("cuda")
        model = rnnt_model.to(device) if model_type == "rnnt" else tdt_model.to(device)
        decoding_config = copy.deepcopy(model.cfg.decoding)

        # checking pytorch implementation
        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypothesis"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False

        model.change_decoding_strategy(decoding_config)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)

        modes = ["no_graphs", "no_while_loops", "full_graph"]
        for force_mode in modes:
            if force_mode == "full_graph":
                skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

            # transcribe with use implementation with cuda graphs
            decoding_config["beam"]["allow_cuda_graphs"] = True
            model.change_decoding_strategy(decoding_config)
            model.decoding.decoding.decoding_computer.force_cuda_graphs_mode(mode=force_mode)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)


class TestTransducerBeamBeta:
    """`beam_beta` in malsd_batch: a fixed score for every non-blank label, in each implementation."""

    @pytest.mark.unit
    @pytest.mark.parametrize("model_type", ["rnnt", "tdt"])
    @pytest.mark.parametrize("device,cuda_graphs_mode", DEVICES_AND_CUDA_GRAPHS_MODES)
    def test_beam_beta_scores_every_label(self, model_type, device, cuda_graphs_mode):
        """Without fusion models, `beam_beta` searches as a late-pruned fusion model scoring every label `beam_beta`."""
        durations = TINY_DURATIONS[model_type]
        beam_beta = 0.7

        with_beam_beta = decode_tiny_transducer(durations, device, cuda_graphs_mode, beam_beta=beam_beta)
        constant_label_score = decode_tiny_transducer(
            durations,
            device,
            cuda_graphs_mode,
            fusion_models=[LabelScoresFusionModel(torch.full([TINY_VOCAB_SIZE], beam_beta))],
            fusion_models_alpha=[1.0],
            pruning_mode="late",
            blank_lm_score_mode="no_score",
        )
        without_beam_beta = decode_tiny_transducer(durations, device, cuda_graphs_mode)

        assert_same_hypotheses(with_beam_beta, constant_label_score)
        assert [hyp.score for hyp in with_beam_beta] != pytest.approx([hyp.score for hyp in without_beam_beta])

    @pytest.mark.unit
    @pytest.mark.parametrize("model_type", ["rnnt", "tdt"])
    @pytest.mark.parametrize("device,cuda_graphs_mode", DEVICES_AND_CUDA_GRAPHS_MODES)
    @pytest.mark.parametrize("pruning_mode", ["early", "late"])
    @pytest.mark.parametrize("blank_lm_score_mode", ["no_score", "lm_weighted_full"])
    def test_beam_beta_with_fusion_models(
        self, model_type, device, cuda_graphs_mode, pruning_mode, blank_lm_score_mode
    ):
        """With a fusion model of weight alpha, `beam_beta` is the same as raising its label scores by beam_beta / alpha."""
        durations = TINY_DURATIONS[model_type]
        alpha, beam_beta = 0.5, 0.7
        lm_scores = torch.log_softmax(torch.randn(TINY_VOCAB_SIZE, generator=torch.Generator().manual_seed(2)), -1)
        fusion = dict(fusion_models_alpha=[alpha], pruning_mode=pruning_mode, blank_lm_score_mode=blank_lm_score_mode)

        with_beam_beta = decode_tiny_transducer(
            durations,
            device,
            cuda_graphs_mode,
            fusion_models=[LabelScoresFusionModel(lm_scores)],
            beam_beta=beam_beta,
            **fusion,
        )
        raised_lm_scores = decode_tiny_transducer(
            durations,
            device,
            cuda_graphs_mode,
            fusion_models=[LabelScoresFusionModel(lm_scores + beam_beta / alpha)],
            **fusion,
        )

        assert_same_hypotheses(with_beam_beta, raised_lm_scores)

    @pytest.mark.unit
    @pytest.mark.parametrize("model_type", ["rnnt", "tdt"])
    @pytest.mark.parametrize("device,cuda_graphs_mode", DEVICES_AND_CUDA_GRAPHS_MODES)
    def test_large_beam_beta_sets_hypothesis_length(self, model_type, device, cuda_graphs_mode):
        """A large negative `beam_beta` leaves only blanks; a large positive one emits a label wherever one is allowed."""
        durations = [0, 1] if model_type == "tdt" else None

        no_labels = decode_tiny_transducer(durations, device, cuda_graphs_mode, beam_beta=-1e3)
        all_labels = decode_tiny_transducer(durations, device, cuda_graphs_mode, beam_beta=1e3)

        assert [len(hyp.y_sequence) for hyp in no_labels] == [0] * len(TINY_ENCODED_LENGTHS)
        for hyp, frames in zip(all_labels, TINY_ENCODED_LENGTHS):
            if model_type == "rnnt":
                # max_symbols labels on every frame, each followed by the forced blank
                assert len(hyp.y_sequence) == frames * TINY_MAX_SYMBOLS
            else:
                # a label on every frame (none skips one), and at most max_symbols per frame
                assert frames <= len(hyp.y_sequence) <= frames * TINY_MAX_SYMBOLS

    @pytest.mark.unit
    @pytest.mark.parametrize("model_type", ["rnnt", "tdt"])
    def test_beam_beta_from_decoding_config(self, model_type):
        durations = TINY_DURATIONS[model_type]
        decoder, joint = get_tiny_transducer(durations)
        vocabulary = list("abcd")[:TINY_VOCAB_SIZE]
        cfg = RNNTDecodingConfig(strategy="malsd_batch", model_type=model_type, durations=durations or [])
        cfg.beam.beam_beta = 0.7

        decoding = RNNTDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, vocabulary=vocabulary)
        assert decoding.decoding.decoding_computer.beam_beta == 0.7

        # greedy decoding does not use the beam settings
        cfg.strategy = "greedy_batch"
        RNNTDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, vocabulary=vocabulary)

    @pytest.mark.unit
    @pytest.mark.parametrize("strategy", ["beam", "maes", "maes_batch"])
    def test_beam_beta_unsupported_strategy(self, strategy):
        decoder, joint = get_tiny_transducer()
        cfg = RNNTDecodingConfig(strategy=strategy)
        cfg.beam.beam_beta = 0.7

        with pytest.raises(NotImplementedError, match="beam_beta"):
            RNNTDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, vocabulary=list("abcd")[:TINY_VOCAB_SIZE])


class TestTDTFrameSyncBeamDecoding:
    """`frame_sync_batch`: frame-synchronous beam search for TDT models."""

    @pytest.mark.unit
    @pytest.mark.parametrize("device,allow_cuda_graphs", DEVICES_AND_FRAME_SYNC_CUDA_GRAPHS)
    @pytest.mark.parametrize(
        "beam_beta,lm_alpha,pruning_mode,blank_lm_score_mode",
        [
            (0.0, 0.0, None, None),
            (0.7, 0.0, None, None),
            (0.0, 0.5, "late", "lm_weighted_full"),
            (0.7, 0.5, "early", "no_score"),
            (-0.3, 0.5, "early", "lm_weighted_full"),
        ],
    )
    def test_matches_reference_search(
        self, device, allow_cuda_graphs, beam_beta, lm_alpha, pruning_mode, blank_lm_score_mode
    ):
        """The batched search finds the n-best lists of a search over each utterance, a hypothesis at a time."""
        lm_scores = torch.log_softmax(torch.randn(TINY_VOCAB_SIZE, generator=torch.Generator().manual_seed(2)), -1)
        fusion = {}
        if lm_alpha:
            fusion = dict(
                fusion_models=[LabelScoresFusionModel(lm_scores)],
                fusion_models_alpha=[lm_alpha],
                pruning_mode=pruning_mode,
                blank_lm_score_mode=blank_lm_score_mode,
            )
        decoding = get_frame_sync_decoding(device, allow_cuda_graphs, beam_beta=beam_beta, **fusion)
        # twice: the second batch reuses the decoding state (and the CUDA graph)
        for _ in range(2):
            nbest_hyps = decode_tiny_encoder_output(decoding, device)
        if allow_cuda_graphs:
            assert decoding.decoding_computer.state.graph is not None

        decoder, joint = get_tiny_transducer(TINY_DURATIONS["tdt"])
        encoder_output, _ = get_tiny_encoder_output()
        for nbest, encoder_frames, length in zip(nbest_hyps, encoder_output, TINY_ENCODED_LENGTHS):
            with torch.no_grad():
                expected = frame_sync_reference(
                    decoder,
                    joint,
                    encoder_frames[:, :length].T,
                    durations=TINY_DURATIONS["tdt"],
                    beam_size=4,
                    max_symbols=TINY_MAX_SYMBOLS,
                    beam_beta=beam_beta,
                    lm_scores=lm_scores if lm_alpha else None,
                    lm_alpha=lm_alpha,
                    pruning_mode=pruning_mode or "late",
                    blank_lm_score_mode=blank_lm_score_mode or "no_score",
                )
            actual = nbest.n_best_hypotheses
            assert [hyp.y_sequence.tolist() for hyp in actual] == [labels for labels, _ in expected]
            # the reference runs on CPU
            assert [hyp.score for hyp in actual] == pytest.approx([score for _, score in expected], abs=1e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("device,allow_cuda_graphs", DEVICES_AND_FRAME_SYNC_CUDA_GRAPHS)
    def test_zero_duration_labels_capped_by_max_symbols(self, device, allow_cuda_graphs):
        """With every label preferred to blank and duration 0 preferred to 1, each frame gets max_symbols labels."""
        decoding = get_frame_sync_decoding(device, allow_cuda_graphs, durations=[0, 1], beam_beta=1e3)
        with torch.no_grad():
            decoding.joint.joint_net[-1].bias[-2] += 10.0  # duration 0

        hyps = [nbest.n_best_hypotheses[0] for nbest in decode_tiny_encoder_output(decoding, device)]

        for hyp, frames in zip(hyps, TINY_ENCODED_LENGTHS):
            assert np.bincount(hyp.timestamp, minlength=frames).tolist() == [TINY_MAX_SYMBOLS] * frames
            assert set(hyp.token_duration.tolist()) == {0}

    @pytest.mark.unit
    @pytest.mark.parametrize("device,allow_cuda_graphs", DEVICES_AND_FRAME_SYNC_CUDA_GRAPHS)
    def test_batch_with_empty_utterance(self, device, allow_cuda_graphs):
        """An utterance without frames gets an empty hypothesis; the others are decoded as without it."""
        decoding = get_frame_sync_decoding(device, allow_cuda_graphs)
        encoder_output, encoded_lengths = get_tiny_encoder_output()
        encoded_lengths[1] = 0

        with torch.no_grad():
            with_empty = decoding(encoder_output=encoder_output.to(device), encoded_lengths=encoded_lengths.to(device))
            without_empty = decoding(
                encoder_output=encoder_output[[0, 2, 3]].to(device),
                encoded_lengths=encoded_lengths[[0, 2, 3]].to(device),
            )

        empty = with_empty[0][1].n_best_hypotheses
        assert [(hyp.y_sequence.tolist(), hyp.score) for hyp in empty] == [([], 0.0)]
        for actual, expected in zip([with_empty[0][i] for i in [0, 2, 3]], without_empty[0]):
            assert_same_hypotheses(actual.n_best_hypotheses, expected.n_best_hypotheses)

    @pytest.mark.unit
    @pytest.mark.parametrize("device,allow_cuda_graphs", DEVICES_AND_FRAME_SYNC_CUDA_GRAPHS)
    def test_retries_batch_that_outgrows_label_nodes(self, device, allow_cuda_graphs):
        decoding = get_frame_sync_decoding(device, allow_cuda_graphs)
        expected = decode_tiny_encoder_output(decoding, device)
        computer = decoding.decoding_computer
        computer.nodes_per_frame = 1e-3
        computer.reset_cuda_graphs_state()

        actual = decode_tiny_encoder_output(decoding, device)

        assert computer.nodes_per_frame > 1e-3
        for actual_nbest, expected_nbest in zip(actual, expected):
            assert_same_hypotheses(actual_nbest.n_best_hypotheses, expected_nbest.n_best_hypotheses)

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Synchronization is checked on CUDA")
    @pytest.mark.gpu
    def test_step_does_not_synchronize(self):
        device = torch.device("cuda")
        fusion = dict(fusion_models=[LabelScoresFusionModel(torch.zeros(TINY_VOCAB_SIZE))], fusion_models_alpha=[0.5])
        decoding = get_frame_sync_decoding(device, allow_cuda_graphs=False, beam_beta=0.5, **fusion)
        decode_tiny_encoder_output(decoding, device)
        computer = decoding.decoding_computer

        with torch.inference_mode():
            computer._reset_hypotheses()
            torch.cuda.set_sync_debug_mode("error")
            try:
                for _ in range(3):
                    computer._step()
            finally:
                torch.cuda.set_sync_debug_mode("default")

    @pytest.mark.unit
    @pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()), reason="bfloat16 is checked on CUDA"
    )
    @pytest.mark.gpu
    def test_bfloat16(self):
        """Autocast does not change the search, and a bfloat16 model decodes the same with and without CUDA graphs."""
        device = torch.device("cuda")
        decoding = get_frame_sync_decoding(device, allow_cuda_graphs=False)
        expected = decode_tiny_encoder_output(decoding, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual = decode_tiny_encoder_output(decoding, device)
        for actual_nbest, expected_nbest in zip(actual, expected):
            assert_same_hypotheses(actual_nbest.n_best_hypotheses, expected_nbest.n_best_hypotheses)

        nbest_hyps = {}
        for allow_cuda_graphs in [False, True]:
            decoding = get_frame_sync_decoding(device, allow_cuda_graphs)
            decoding.decoder.to(torch.bfloat16)
            decoding.joint.to(torch.bfloat16)
            nbest_hyps[allow_cuda_graphs] = decode_tiny_encoder_output(decoding, device)
        for with_graphs, without_graphs in zip(nbest_hyps[True], nbest_hyps[False]):
            assert_same_hypotheses(with_graphs.n_best_hypotheses, without_graphs.n_best_hypotheses)
            assert len(with_graphs.n_best_hypotheses) > 0

    @pytest.mark.unit
    def test_from_decoding_config(self):
        durations = TINY_DURATIONS["tdt"]
        decoder, joint = get_tiny_transducer(durations)
        cfg = RNNTDecodingConfig(strategy="frame_sync_batch", model_type="tdt", durations=durations)
        cfg.beam.beam_size = 3
        cfg.beam.beam_beta = 0.7
        cfg.beam.max_symbols_per_step = 2

        decoding = RNNTDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, vocabulary=list("abcd"))

        computer = decoding.decoding.decoding_computer
        assert isinstance(computer, FrameSyncBatchedTDTComputer)
        assert (computer.beam_size, computer.max_symbols, computer.beam_beta) == (3, 2, 0.7)

    @pytest.mark.unit
    def test_not_supported_for_rnnt(self):
        decoder, joint = get_tiny_transducer()
        cfg = RNNTDecodingConfig(strategy="frame_sync_batch")

        with pytest.raises(NotImplementedError, match="does not support"):
            RNNTDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, vocabulary=list("abcd"))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "option", ["preserve_alignments", "enable_per_stream_biasing", "preserve_step_confidence"]
    )
    def test_unsupported_options(self, option):
        with pytest.raises(NotImplementedError, match="frame_sync_batch"):
            get_frame_sync_decoding(torch.device("cpu"), allow_cuda_graphs=False, **{option: True})


class TestCTCDecoding:
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"allow_cuda_graphs": False},
            {"allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("device", DEVICES)
    def test_ctc_beam_decoding_return_best_hypothesis(
        self, test_audio_filenames, ctc_model, get_ctc_output, beam_config, device, batch_size, beam_size
    ):
        num_samples = min(batch_size, len(test_audio_filenames))
        model = ctc_model.to(device)
        log_probs, encoded_lengths = get_ctc_output
        log_probs, encoded_lengths = log_probs[:num_samples].to(device), encoded_lengths[:num_samples].to(device)

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedCTCInfer(
            blank_index=vocab_size,
            beam_size=beam_size,
            return_best_hypothesis=True,
            **beam_config,
        )

        print_unit_test_info(
            strategy="beam_batch",
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(decoder_output=log_probs, decoder_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"allow_cuda_graphs": False},
            {"allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.gpu
    def test_ctc_beam_decoding_return_nbest(
        self, test_audio_filenames, ctc_model, get_ctc_output, beam_config, device, beam_size, batch_size
    ):
        device = torch.device("cuda")
        num_samples = min(batch_size, len(test_audio_filenames))
        model = ctc_model.to(device)
        log_probs, encoded_lengths = get_ctc_output
        log_probs, encoded_lengths = log_probs[:num_samples].to(device), encoded_lengths[:num_samples].to(device)

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedCTCInfer(
            blank_index=vocab_size,
            beam_size=beam_size,
            return_best_hypothesis=False,
            **beam_config,
        )

        print_unit_test_info(
            strategy="beam_batch",
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            batch_nbest_hyps = decoding(decoder_output=log_probs, decoder_lengths=encoded_lengths)[0]

            check_res_nbest_hyps(num_samples, batch_nbest_hyps)
            batch_nbest_hyps = decode_text_from_nbest_hypotheses(batch_nbest_hyps, model)
            print_res_nbest_hyps(batch_nbest_hyps)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"allow_cuda_graphs": False, "ngram_lm_alpha": 0.3, "beam_beta": 1.0},
            {"allow_cuda_graphs": False, "ngram_lm_alpha": 0.3, "beam_beta": 1.0},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.gpu
    def test_ctc_beam_decoding_kenlm(
        self,
        kenlm_model_path,
        test_audio_filenames,
        ctc_model,
        get_ctc_output,
        beam_config,
        device,
        batch_size,
        beam_size,
    ):
        device = torch.device("cuda")
        beam_config["ngram_lm_model"] = kenlm_model_path

        num_samples = min(batch_size, len(test_audio_filenames))
        model = ctc_model.to(device)
        decoder_output, decoder_lengths = get_ctc_output
        decoder_output, decoder_lengths = decoder_output[:num_samples].to(device), decoder_lengths[:num_samples].to(
            device
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedCTCInfer(
            blank_index=vocab_size,
            beam_size=beam_size,
            return_best_hypothesis=True,
            **beam_config,
        )

        print_unit_test_info(
            strategy="beam_batch",
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(decoder_output=decoder_output, decoder_lengths=decoder_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)
