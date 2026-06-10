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

Beam-search analogue of ``test_streaming_decoding.py``. See module docstrings on
individual tests for coverage details.
"""

import copy
from typing import Optional

import pytest
import torch
from omegaconf import open_dict
from tqdm.auto import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import BatchedBeamState
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import BatchedBeamHyps
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from tests.collections.asr.decoding.utils import load_audio, make_preprocessor_deterministic


def get_devices_for_testing(use_cpu_always: bool = False) -> list[torch.device]:
    devices = [torch.device("cpu")] if use_cpu_always else []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    if torch.mps.is_available():
        devices.append(torch.device("mps"))
    if not devices:
        devices.append(torch.device("cpu"))
    return devices


DEVICES = get_devices_for_testing(use_cpu_always=True)


def _make_device_param_matrix() -> list:
    entries: list = []
    for device in DEVICES:
        entries.append(pytest.param(device, None, id=f"{device.type}-no-graphs"))
    for device in DEVICES:
        if device.type == "cuda":
            entries.append(pytest.param(device, "full_graph", id=f"{device.type}-full-graph"))
            entries.append(pytest.param(device, "no_while_loops", id=f"{device.type}-no-while-loops"))
    return entries


DEVICE_PARAM_MATRIX = _make_device_param_matrix()
MAES_DEVICE_PARAM_MATRIX = [pytest.param(device, id=device.type) for device in DEVICES]

_WB_KEY_PHRASES: list[str] = ["nineteen", "forty", "fifty", "repeat", "stop", "yes"]


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
    filenames = [record["audio_filepath"] for record in records]
    encoder_output, encoder_output_len = get_model_encoder_output(
        test_audio_filenames=filenames, model=model, num_samples=len(filenames), device=device
    )
    return encoder_output, encoder_output_len


def _select_transducer_model(is_tdt: bool, rnnt_model: ASRModel, tdt_model: ASRModel) -> ASRModel:
    return tdt_model if is_tdt else rnnt_model


def _prepare_model(model: ASRModel, device: torch.device) -> ASRModel:
    model.eval()
    model.to(device=device)
    return model


def _reset_decoding_computer_state(model: ASRModel) -> None:
    decoding_computer = model.decoding.decoding.decoding_computer
    if hasattr(decoding_computer, "reset_cuda_graphs_state"):
        decoding_computer.reset_cuda_graphs_state()


def _configure_malsd_decoding(
    model: ASRModel,
    cuda_graphs_mode: Optional[str],
    beam_size: int,
    max_symbols: int,
    key_phrases_list: Optional[list[str]] = None,
    boosting_tree_alpha: float = 1.0,
    enable_per_stream_biasing: bool = False,
) -> None:
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
        if enable_per_stream_biasing:
            decoding_cfg.beam.enable_per_stream_biasing = True
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


def _chunk_lengths(encoder_output_len: torch.Tensor, t: int, chunk_size: int) -> torch.Tensor:
    rest_len = encoder_output_len - t
    current_len = torch.full_like(encoder_output_len, fill_value=chunk_size)
    current_len = torch.minimum(current_len, rest_len)
    return torch.maximum(current_len, torch.zeros_like(current_len))


def _decode_malsd_encoder_in_chunks(
    decoding_computer,
    encoder_output: torch.Tensor,
    encoder_output_len: torch.Tensor,
    chunk_size: int,
    multi_biasing_ids: Optional[torch.Tensor] = None,
) -> BatchedBeamHyps:
    encoder_output = encoder_output.transpose(1, 2)
    state: Optional[BatchedBeamState] = None
    current_batched_hyps: BatchedBeamHyps | None = None

    decode_kwargs = {}
    if multi_biasing_ids is not None:
        decode_kwargs["multi_biasing_ids"] = multi_biasing_ids

    for t in range(0, encoder_output.shape[1], chunk_size):
        chunk_batched_hyps, state = decoding_computer(
            x=encoder_output[:, t : t + chunk_size],
            out_len=_chunk_lengths(encoder_output_len, t, chunk_size),
            prev_batched_state=state,
            **decode_kwargs,
        )
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
    return current_batched_hyps


def _register_per_stream_biasing(
    decoding_computer,
    tokenizer,
    boost_texts: list[str],
    device: torch.device,
    boosting_model_alpha: float = 10.0,
) -> tuple[torch.Tensor, list[BiasingRequestItemConfig | None]]:
    batch_size = len(boost_texts)
    multi_biasing_ids = torch.full([batch_size], fill_value=-1, dtype=torch.long, device=device)
    biasing_requests: list[BiasingRequestItemConfig | None] = []

    for batch_idx, boost_text in enumerate(boost_texts):
        if not boost_text:
            biasing_requests.append(None)
            continue
        request = BiasingRequestItemConfig(
            boosting_model_cfg=BoostingTreeModelConfig(key_phrases_list=[boost_text], unk_score=-100),
            boosting_model_alpha=boosting_model_alpha,
        )
        request.add_to_multi_model(
            tokenizer=tokenizer,
            biasing_multi_model=decoding_computer.biasing_multi_model,
        )
        if request.multi_model_id is not None:
            multi_biasing_ids[batch_idx] = request.multi_model_id
        biasing_requests.append(request)

    return multi_biasing_ids, biasing_requests


def _unregister_per_stream_biasing(decoding_computer, biasing_requests: list[BiasingRequestItemConfig | None]) -> None:
    for request in biasing_requests:
        if request is not None and request.multi_model_id is not None:
            decoding_computer.biasing_multi_model.remove_model(request.multi_model_id)
            request.multi_model_id = None


def _transcripts_from_batched_hyps(model: ASRModel, batched_hyps: BatchedBeamHyps) -> list[str]:
    return [model.tokenizer.ids_to_text(hyp.y_sequence.tolist()) for hyp in batched_hyps.to_hyps_list(score_norm=True)]


def _run_malsd_streaming_manifest(
    model: ASRModel,
    manifest_path,
    device: torch.device,
    chunk_size: int,
    batch_size: int,
    boost_texts: Optional[list[str]] = None,
    boosting_model_alpha: float = 10.0,
) -> list[str]:
    manifest = read_manifest(manifest_path)
    decoding_computer = model.decoding.decoding.decoding_computer
    all_transcripts: list[str] = []

    with torch.no_grad(), torch.inference_mode():
        for i in range(0, len(manifest), batch_size):
            batch_records = manifest[i : i + batch_size]
            encoder_output, encoder_output_len = get_batch_encoder_outputs_from_records(
                batch_records, model=model, device=device
            )

            multi_biasing_ids = None
            biasing_requests: list[BiasingRequestItemConfig | None] = []
            if boost_texts is not None:
                assert decoding_computer.biasing_multi_model is not None
                batch_boost_texts = boost_texts[i : i + batch_size]
                multi_biasing_ids, biasing_requests = _register_per_stream_biasing(
                    decoding_computer,
                    model.tokenizer,
                    batch_boost_texts,
                    device,
                    boosting_model_alpha=boosting_model_alpha,
                )

            batched_hyps = _decode_malsd_encoder_in_chunks(
                decoding_computer,
                encoder_output,
                encoder_output_len,
                chunk_size,
                multi_biasing_ids=multi_biasing_ids,
            )
            if boost_texts is not None:
                _unregister_per_stream_biasing(decoding_computer, biasing_requests)

            all_transcripts.extend(_transcripts_from_batched_hyps(model, batched_hyps))

    return all_transcripts


def _run_streaming_batched_state(
    model: ASRModel,
    manifest_path,
    device: torch.device,
    chunk_size: int,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    transcriptions = model.transcribe(audio=str(manifest_path.absolute()), batch_size=batch_size)
    ref_transcripts = [hyp.text for hyp in transcriptions]
    streaming_transcripts = _run_malsd_streaming_manifest(
        model, manifest_path, device, chunk_size, batch_size
    )
    return ref_transcripts, streaming_transcripts


def _assert_transcripts_equal(ref_transcripts: list[str], streaming_transcripts: list[str], context: str) -> None:
    assert ref_transcripts == streaming_transcripts, (
        f"{context}\n"
        + "\n".join(
            f"  [{i}] ref: {ref!r} != streaming: {stream!r}"
            for i, (ref, stream) in enumerate(zip(ref_transcripts, streaming_transcripts))
            if ref != stream
        )
    )


@pytest.mark.with_downloads
@pytest.mark.parametrize("device,cuda_graphs_mode", DEVICE_PARAM_MATRIX)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("beam_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_malsd_streaming_batched_state(
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
    model = _prepare_model(
        _select_transducer_model(is_tdt, stt_en_fastconformer_transducer_large, stt_en_fastconformer_tdt_large),
        device,
    )
    _configure_malsd_decoding(model, cuda_graphs_mode, beam_size=beam_size, max_symbols=max_symbols)
    ref_transcripts, streaming_transcripts = _run_streaming_batched_state(
        model, an4_val_manifest_corrected, device, chunk_size, batch_size
    )
    _assert_transcripts_equal(ref_transcripts, streaming_transcripts, "MALSD chunked streaming must match transcribe")


@pytest.mark.with_downloads
@pytest.mark.parametrize("device", MAES_DEVICE_PARAM_MATRIX)
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("beam_size", [4])
@pytest.mark.parametrize("maes_num_steps", [2])
@pytest.mark.parametrize("maes_expansion_beta", [2])
@pytest.mark.parametrize("maes_expansion_gamma", [2.3])
def test_maes_streaming_batched_state(
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
    model = _prepare_model(stt_en_fastconformer_transducer_large, device)
    _configure_maes_decoding(
        model, beam_size, maes_num_steps, maes_expansion_beta, maes_expansion_gamma
    )
    ref_transcripts, streaming_transcripts = _run_streaming_batched_state(
        model, an4_val_manifest_corrected, device, chunk_size, batch_size
    )
    _assert_transcripts_equal(ref_transcripts, streaming_transcripts, "MAES chunked streaming must match transcribe")


@pytest.mark.with_downloads
@pytest.mark.parametrize("device,cuda_graphs_mode", DEVICE_PARAM_MATRIX)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("beam_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_malsd_streaming_batched_state_with_word_boosting(
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
    model = _prepare_model(
        _select_transducer_model(is_tdt, stt_en_fastconformer_transducer_large, stt_en_fastconformer_tdt_large),
        device,
    )
    _configure_malsd_decoding(
        model, cuda_graphs_mode, beam_size, max_symbols, key_phrases_list=_WB_KEY_PHRASES
    )
    ref_transcripts, streaming_transcripts = _run_streaming_batched_state(
        model, an4_val_manifest_corrected, device, chunk_size, batch_size
    )
    _assert_transcripts_equal(
        ref_transcripts, streaming_transcripts, "MALSD with global boosting_tree must be chunk-invariant"
    )


@pytest.mark.with_downloads
@pytest.mark.parametrize("device,cuda_graphs_mode", DEVICE_PARAM_MATRIX)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("beam_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_malsd_streaming_boosting_with_ref_transcripts(
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
    """Metamorphic test analogous to ``test_label_looping_streaming_boosting_with_ref_transcripts``."""
    model = _prepare_model(
        _select_transducer_model(is_tdt, stt_en_fastconformer_transducer_large, stt_en_fastconformer_tdt_large),
        device,
    )

    _configure_malsd_decoding(model, cuda_graphs_mode, beam_size, max_symbols)
    ref_transcripts = [
        hyp.text
        for hyp in model.transcribe(audio=str(an4_val_manifest_corrected.absolute()), batch_size=batch_size)
    ]

    _configure_malsd_decoding(
        model, cuda_graphs_mode, beam_size, max_symbols, enable_per_stream_biasing=True
    )
    _reset_decoding_computer_state(model)

    streaming_transcripts = _run_malsd_streaming_manifest(
        model,
        an4_val_manifest_corrected,
        device,
        chunk_size,
        batch_size,
        boost_texts=ref_transcripts,
    )
    _assert_transcripts_equal(
        ref_transcripts,
        streaming_transcripts,
        "Per-stream biasing with reference transcripts must not change MALSD beam output",
    )
