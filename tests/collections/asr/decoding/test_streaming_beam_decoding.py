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

"""Streaming beam-search (MALSD + MAES) decoding tests.

Beam-search analogue of ``test_streaming_decoding.py``. Exercises the streaming
path of the batched beam-search computers by manually feeding the encoder
output to ``model.decoding.decoding.decoding_computer`` in chunks and threading
``prev_batched_state`` (a ``BatchedBeamLoopingState``):

  - :func:`test_malsd_streaming_batched_state` -- covers RNNT and TDT MALSD
    (:class:`ModifiedALSDBatchedRNNTComputer`, :class:`ModifiedALSDBatchedTDTComputer`)
    across the eager path and both captured-graph variants (``full_graph`` and
    ``no_while_loops``).
  - :func:`test_malsd_streaming_batched_state_with_word_boosting` -- same MALSD matrix
    but with a ``GPUBoostingTreeModel`` fusion model plugged in (``boosting_tree.
    key_phrases_list``); exercises cross-chunk restoration of per-beam fusion states
    in :meth:`_restore_state_from_prev`.
  - :func:`test_maes_streaming_batched_state` -- covers RNNT MAES
    (:class:`ModifiedAESBatchedRNNTComputer`); MAES is RNNT-only and pure-PyTorch,
    so there is no ``is_tdt`` / ``cuda_graphs_mode`` axis.

Per-chunk results are merged into a single ``BatchedBeamHyps`` via
``flatten_()`` + ``merge_(..., is_chunk_continuation=True,
boundary_prev_ptr=...)`` -- the same accumulation pattern used by the
cache-aware / chunked streaming inference scripts.

Streamed transcripts are asserted to be identical to the non-streaming
reference produced by ``model.transcribe`` with the same beam settings:
beam search with ``prev_batched_state`` is chunk-invariant because all
cross-chunk per-beam state (scores, ``last_label``, decoded lengths,
decoder + fusion states, ``last_timestamp_lasts``, ...) is preserved across
boundaries.
"""

import copy
from typing import Optional

import pytest
import torch
from omegaconf import open_dict
from tqdm.auto import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import BatchedBeamLoopingState
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import (
    BatchedBeamHyps,
    batched_beam_hyps_to_hypotheses,
)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from tests.collections.asr.decoding.utils import load_audio, make_preprocessor_deterministic


def get_devices_for_testing(use_cpu_always: bool = False) -> list[torch.device]:
    devices = [torch.device("cpu")] if use_cpu_always else []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    if torch.mps.is_available():
        devices.append(torch.device("mps"))

    if len(devices) == 0:
        # no fast device for testing, add CPU
        devices.append(torch.device("cpu"))
    return devices


DEVICES = get_devices_for_testing(use_cpu_always=True)


def _make_device_param_matrix() -> list:
    """
    Build the ``(device, cuda_graphs_mode)`` parametrize entries with explicit, readable
    pytest IDs (``cpu-no-graphs``, ``cuda-full-graph``, ``cuda-no-while-loops``, ...) so the
    test matrix shows which device + graph-mode pair is exercised instead of opaque
    ``device0`` / ``device1`` ids.

    ``cuda_graphs_mode`` is ``None`` for the eager path or one of the
    :class:`ModifiedALSDBatched{RNNT,TDT}Computer.CudaGraphsMode` string values
    (``"full_graph"`` / ``"no_while_loops"``) for the two captured-graph variants. The test
    uses ``force_cuda_graphs_mode`` to pin the variant explicitly so each captured path is
    actually exercised (otherwise ``maybe_enable_cuda_graphs`` would always pick
    ``full_graph`` whenever conditional nodes are supported).

    Coverage:
      - every device in ``DEVICES`` with ``cuda_graphs_mode=None`` (eager path).
      - every CUDA device additionally with both ``"full_graph"`` and ``"no_while_loops"``.
    """
    entries: list = []
    for device in DEVICES:
        entries.append(pytest.param(device, None, id=f"{device.type}-no-graphs"))
    for device in DEVICES:
        if device.type == "cuda":
            entries.append(pytest.param(device, "full_graph", id=f"{device.type}-full-graph"))
            entries.append(pytest.param(device, "no_while_loops", id=f"{device.type}-no-while-loops"))
    return entries


DEVICE_PARAM_MATRIX = _make_device_param_matrix()


def _make_maes_device_param_matrix() -> list:
    """Build readable ``device`` parametrize entries for MAES tests.

    MAES has no CUDA-graphs path (it's pure-PyTorch), so the matrix is just one entry per
    available device with explicit IDs (``cpu``, ``cuda``, ...) instead of opaque
    ``device0`` / ``device1``.
    """
    return [pytest.param(device, id=device.type) for device in DEVICES]


MAES_DEVICE_PARAM_MATRIX = _make_maes_device_param_matrix()


def get_model_encoder_output(
    test_audio_filenames,
    num_samples: int,
    model: ASRModel,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    audio_filepaths = test_audio_filenames[:num_samples]

    with torch.no_grad():
        make_preprocessor_deterministic(model)
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


def get_batch_encoder_outputs_from_records(records, model, device):
    """Helper function to get encoder outputs for a batch of manifest records"""
    filenames = [record["audio_filepath"] for record in records]
    local_batch_size = len(filenames)
    encoder_output, encoder_output_len = get_model_encoder_output(
        test_audio_filenames=filenames, model=model, num_samples=local_batch_size, device=device
    )
    return encoder_output, encoder_output_len


def _configure_malsd_decoding(
    model: ASRModel,
    cuda_graphs_mode: Optional[str],
    beam_size: int,
    max_symbols: int,
    key_phrases_list: Optional[list[str]] = None,
    boosting_tree_alpha: float = 1.0,
) -> None:
    """Switch ``model`` to the ``malsd_batch`` beam-search strategy used by the streaming tests.

    ``cuda_graphs_mode`` is the CudaGraphsMode string (``"full_graph"`` or
    ``"no_while_loops"``) when CUDA graphs should be used, or ``None`` for the eager path.
    When non-None we also call ``force_cuda_graphs_mode`` to pin the variant after the
    decoding strategy is swapped in -- ``maybe_enable_cuda_graphs`` would otherwise auto-pick
    ``full_graph`` whenever conditional nodes are supported, leaving the ``no_while_loops``
    branch effectively untested.

    When ``key_phrases_list`` is given, a ``boosting_tree`` fusion model is plugged in
    (``BoostingTreeModelConfig.key_phrases_list``) so the streaming path exercises
    cross-chunk fusion-state restoration in :meth:`_restore_state_from_prev`.
    """
    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    decoding_cfg.strategy = "malsd_batch"
    with open_dict(decoding_cfg):
        decoding_cfg.beam.beam_size = beam_size
        decoding_cfg.beam.max_symbols = max_symbols
        decoding_cfg.beam.allow_cuda_graphs = cuda_graphs_mode is not None
        decoding_cfg.beam.return_best_hypothesis = True
        decoding_cfg.beam.score_norm = True
        if key_phrases_list is not None:
            decoding_cfg.beam.boosting_tree = {"key_phrases_list": list(key_phrases_list)}
            decoding_cfg.beam.boosting_tree_alpha = boosting_tree_alpha
    model.change_decoding_strategy(decoding_cfg)
    if cuda_graphs_mode is not None:
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode(cuda_graphs_mode)


def _configure_maes_decoding(
    model: ASRModel,
    beam_size: int,
    maes_num_steps: int,
    maes_expansion_beta: int,
    maes_expansion_gamma: float,
) -> None:
    """Switch ``model`` to the ``maes_batch`` beam-search strategy used by the streaming tests.

    MAES is RNNT-only and currently pure-PyTorch (CUDA graphs are not implemented;
    ``allow_cuda_graphs`` is accepted only for API parity with MALSD and is ignored by
    :class:`ModifiedAESBatchedRNNTComputer`).
    """
    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    decoding_cfg.strategy = "maes_batch"
    with open_dict(decoding_cfg):
        decoding_cfg.beam.beam_size = beam_size
        decoding_cfg.beam.maes_num_steps = maes_num_steps
        decoding_cfg.beam.maes_expansion_beta = maes_expansion_beta
        decoding_cfg.beam.maes_expansion_gamma = maes_expansion_gamma
        decoding_cfg.beam.allow_cuda_graphs = False
        decoding_cfg.beam.return_best_hypothesis = True
        decoding_cfg.beam.score_norm = True
    model.change_decoding_strategy(decoding_cfg)


def _run_streaming_batched_state(
    model: ASRModel,
    manifest_path,
    device: torch.device,
    chunk_size: int,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    """Drive the model's beam-search ``decoding_computer`` chunk-by-chunk and return
    ``(ref_transcripts, streaming_transcripts)``.

    Shared between the MALSD and MAES streaming tests: both decoders return a
    ``(BatchedBeamHyps, None, BatchedBeamLoopingState)`` triple and accept
    ``prev_batched_state`` for cross-chunk state threading. The per-chunk results are
    flattened and merged into a single accumulator via ``flatten_()`` +
    ``merge_(..., is_chunk_continuation=True, boundary_prev_ptr=...)`` -- the same
    accumulation pattern used by the cache-aware / chunked streaming inference scripts.
    """
    manifest = read_manifest(manifest_path)
    transcriptions = model.transcribe(audio=str(manifest_path.absolute()), batch_size=batch_size)
    ref_transcripts = [hyp.text for hyp in transcriptions]

    all_hyps = []
    decoding_computer = model.decoding.decoding.decoding_computer
    print(f"decoding_computer: {type(decoding_computer)}")
    with torch.no_grad(), torch.inference_mode():
        for i in range(0, len(manifest), batch_size):
            encoder_output, encoder_output_len = get_batch_encoder_outputs_from_records(
                manifest[i : i + batch_size], model=model, device=device
            )
            local_batch_size = encoder_output_len.shape[0]
            state: Optional[BatchedBeamLoopingState] = None
            current_batched_hyps: BatchedBeamHyps | None = None
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            for t in range(0, encoder_output.shape[1], chunk_size):
                rest_len = encoder_output_len - t
                current_len = torch.full_like(encoder_output_len, fill_value=chunk_size)
                current_len = torch.minimum(current_len, rest_len)
                current_len = torch.maximum(current_len, torch.zeros_like(current_len))
                chunk_batched_hyps, _, state = decoding_computer(
                    x=encoder_output[:, t : t + chunk_size],
                    out_len=current_len,
                    prev_batched_state=state,
                )
                # Flatten this chunk's prefix tree and thread the cross-chunk beam
                # permutation (``root_ptrs``) into the accumulator so the final
                # ``flatten_sort_`` walks back through the right beam history.
                chunk_root_ptrs = chunk_batched_hyps.flatten_()
                if current_batched_hyps is None:
                    current_batched_hyps = chunk_batched_hyps
                else:
                    current_batched_hyps.merge_(
                        chunk_batched_hyps,
                        is_chunk_continuation=True,
                        boundary_prev_ptr=chunk_root_ptrs,
                    )
            assert current_batched_hyps is not None
            all_hyps.extend(batched_beam_hyps_to_hypotheses(current_batched_hyps, batch_size=local_batch_size))

    streaming_transcripts = [model.tokenizer.ids_to_text(hyp.y_sequence.tolist()) for hyp in all_hyps]
    return ref_transcripts, streaming_transcripts


@pytest.mark.with_downloads
@pytest.mark.parametrize("device,cuda_graphs_mode", DEVICE_PARAM_MATRIX)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("beam_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_malsd_streaming_batched_state(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    device: torch.device,
    cuda_graphs_mode: Optional[str],
    is_tdt: bool,
    chunk_size: int,
    batch_size: int,
    beam_size: int,
    max_symbols: int,
):
    """Streaming MALSD decoding with batched beam state passed across chunks."""
    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large
    model.eval()
    model.to(device=device)

    _configure_malsd_decoding(model, cuda_graphs_mode, beam_size=beam_size, max_symbols=max_symbols)

    ref_transcripts, streaming_transcripts = _run_streaming_batched_state(
        model=model,
        manifest_path=an4_val_manifest_corrected,
        device=device,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )
    assert ref_transcripts == streaming_transcripts


@pytest.mark.with_downloads
@pytest.mark.parametrize("device", MAES_DEVICE_PARAM_MATRIX)
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("beam_size", [4])
@pytest.mark.parametrize("maes_num_steps", [2])
@pytest.mark.parametrize("maes_expansion_beta", [2])
@pytest.mark.parametrize("maes_expansion_gamma", [2.3])
def test_maes_streaming_batched_state(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    device: torch.device,
    chunk_size: int,
    batch_size: int,
    beam_size: int,
    maes_num_steps: int,
    maes_expansion_beta: int,
    maes_expansion_gamma: float,
):
    """Streaming MAES decoding with batched beam state passed across chunks.

    MAES is RNNT-only and pure-PyTorch (no CUDA graphs path), so the device matrix is
    just the set of available devices and there is no ``cuda_graphs_mode`` / ``is_tdt``
    parameter.
    """
    model = stt_en_fastconformer_transducer_large
    model.eval()
    model.to(device=device)

    _configure_maes_decoding(
        model,
        beam_size=beam_size,
        maes_num_steps=maes_num_steps,
        maes_expansion_beta=maes_expansion_beta,
        maes_expansion_gamma=maes_expansion_gamma,
    )

    ref_transcripts, streaming_transcripts = _run_streaming_batched_state(
        model=model,
        manifest_path=an4_val_manifest_corrected,
        device=device,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )
    assert ref_transcripts == streaming_transcripts


# Phrases chosen from the AN4 val transcripts so the boosting tree is actually exercised
# (an empty / unseen list collapses to the no-boosting path and would not test fusion-state
# restoration across chunks).
_WB_KEY_PHRASES: list[str] = ["nineteen", "forty", "fifty", "repeat", "stop", "yes"]


@pytest.mark.with_downloads
@pytest.mark.parametrize("device,cuda_graphs_mode", DEVICE_PARAM_MATRIX)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("beam_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_malsd_streaming_batched_state_with_word_boosting(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    device: torch.device,
    cuda_graphs_mode: Optional[str],
    is_tdt: bool,
    chunk_size: int,
    batch_size: int,
    beam_size: int,
    max_symbols: int,
):
    """Streaming MALSD with word-boosting (``boosting_tree``) is chunk-invariant.

    Adds a ``GPUBoostingTreeModel`` fusion model on top of the standard streaming MALSD
    test. The reference (``model.transcribe``) and the streaming path are configured
    identically, so the boosting tree's per-beam fusion states must be restored across
    chunks via ``_restore_state_from_prev`` for the two transcripts to match.
    """
    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large
    model.eval()
    model.to(device=device)

    _configure_malsd_decoding(
        model,
        cuda_graphs_mode,
        beam_size=beam_size,
        max_symbols=max_symbols,
        key_phrases_list=_WB_KEY_PHRASES,
    )

    ref_transcripts, streaming_transcripts = _run_streaming_batched_state(
        model=model,
        manifest_path=an4_val_manifest_corrected,
        device=device,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )
    assert ref_transcripts == streaming_transcripts
