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

import copy
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from omegaconf import open_dict
from tqdm.auto import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    BatchedLabelLoopingState,
    GreedyBatchedLabelLoopingComputerBase,
)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import BatchedHyps, Hypothesis, batched_hyps_to_hypotheses
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


DEVICES = get_devices_for_testing(use_cpu_always=False)


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


@pytest.mark.with_downloads
@pytest.mark.parametrize(
    "device,use_cuda_graph_decoder",
    [(device, False) for device in DEVICES] + [(device, True) for device in DEVICES if device.type == "cuda"],
)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_label_looping_streaming_batched_state(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    device: torch.device,
    use_cuda_graph_decoder: bool,
    is_tdt: bool,
    chunk_size: int,
    batch_size: int,
    max_symbols: int,
):
    """Test streaming decoding with batched state"""
    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large
    model.eval()
    model.to(device=device)

    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    decoding_cfg.strategy = "greedy_batch"
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
        decoding_cfg.greedy.max_symbols = max_symbols
    model.change_decoding_strategy(decoding_cfg)

    manifest = read_manifest(an4_val_manifest_corrected)
    transcriptions = model.transcribe(audio=str(an4_val_manifest_corrected.absolute()), batch_size=batch_size)
    ref_transcripts = [hyp.text for hyp in transcriptions]

    all_hyps = []
    decoding_computer: GreedyBatchedLabelLoopingComputerBase = model.decoding.decoding.decoding_computer
    with torch.no_grad(), torch.inference_mode():
        for i in range(0, len(manifest), batch_size):
            encoder_output, encoder_output_len = get_batch_encoder_outputs_from_records(
                manifest[i : i + batch_size], model=model, device=device
            )
            local_batch_size = encoder_output_len.shape[0]
            # decode encoder output by chunks, passing state between decoder invocations
            state: Optional[BatchedLabelLoopingState] = None
            batched_hyps: BatchedHyps | None = None
            encoder_output = encoder_output.transpose(1, 2)
            for t in range(0, encoder_output.shape[1], chunk_size):
                rest_len = encoder_output_len - t
                current_len = torch.full_like(encoder_output_len, fill_value=chunk_size)
                current_len = torch.minimum(current_len, rest_len)
                current_len = torch.maximum(current_len, torch.zeros_like(current_len))
                batched_hyps_chunk, _, state = decoding_computer(
                    x=encoder_output[:, t : t + chunk_size],
                    out_len=current_len,
                    prev_batched_state=state,
                )
                if batched_hyps is None:
                    batched_hyps = batched_hyps_chunk
                else:
                    batched_hyps.merge_(batched_hyps_chunk)
            assert batched_hyps is not None
            all_hyps.extend(batched_hyps_to_hypotheses(batched_hyps, None, batch_size=local_batch_size))

    streaming_transcripts = []
    for hyp in all_hyps:
        streaming_transcripts.append(model.tokenizer.ids_to_text(hyp.y_sequence.tolist()))
    assert ref_transcripts == streaming_transcripts


@pytest.mark.with_downloads
@pytest.mark.parametrize(
    "device,use_cuda_graph_decoder",
    [(device, False) for device in DEVICES] + [(device, True) for device in DEVICES if device.type == "cuda"],
)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_label_looping_streaming_partial_hypotheses(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    device: torch.device,
    use_cuda_graph_decoder: bool,
    is_tdt: bool,
    chunk_size: int,
    batch_size: int,
    max_symbols: int,
):
    """Test streaming decoding with partial hypotheses"""
    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large
    model.eval()
    model.to(device=device)

    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    decoding_cfg.strategy = "greedy_batch"
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
        decoding_cfg.greedy.max_symbols = max_symbols
    model.change_decoding_strategy(decoding_cfg)

    manifest = read_manifest(an4_val_manifest_corrected)
    transcriptions = model.transcribe(audio=str(an4_val_manifest_corrected.absolute()), batch_size=batch_size)
    ref_transcripts = [hyp.text for hyp in transcriptions]

    all_hyps = []
    rnnt_infer = model.decoding.decoding
    with torch.no_grad(), torch.inference_mode():
        for i in range(0, len(manifest), batch_size):
            encoder_output, encoder_output_len = get_batch_encoder_outputs_from_records(
                manifest[i : i + batch_size], model=model, device=device
            )
            # decode encoder output by chunks, passing state between decoder invocations
            hyps: list[Hypothesis] | None = None
            for t in range(0, encoder_output.shape[2], chunk_size):
                rest_len = encoder_output_len - t
                current_len = torch.full_like(encoder_output_len, fill_value=chunk_size)
                current_len = torch.minimum(current_len, rest_len)
                current_len = torch.maximum(current_len, torch.zeros_like(current_len))
                (hyps,) = rnnt_infer(
                    encoder_output=encoder_output[:, :, t : t + chunk_size],
                    encoded_lengths=current_len,
                    partial_hypotheses=hyps,
                )
            # free up memory by resetting decoding state
            for hyp in hyps:
                hyp.clean_decoding_state_()
            all_hyps.extend(hyps)
    streaming_transcripts = []
    for hyp in all_hyps:
        streaming_transcripts.append(model.tokenizer.ids_to_text(hyp.y_sequence.tolist()))
    assert ref_transcripts == streaming_transcripts


@pytest.mark.with_downloads
@pytest.mark.parametrize(
    "device,use_cuda_graph_decoder",
    [(device, False) for device in DEVICES] + [(device, True) for device in DEVICES if device.type == "cuda"],
)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_label_looping_continuous_streaming_batched_state(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    device: torch.device,
    use_cuda_graph_decoder: bool,
    is_tdt: bool,
    chunk_size: int,
    batch_size: int,
    max_symbols: int,
):
    """Test streaming continuos decoding with partial hypotheses"""
    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large
    model.eval()
    model.to(device=device)

    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    decoding_cfg.strategy = "greedy_batch"
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
        decoding_cfg.greedy.max_symbols = max_symbols
    model.change_decoding_strategy(decoding_cfg)

    manifest = read_manifest(an4_val_manifest_corrected)
    transcriptions = model.transcribe(audio=str(an4_val_manifest_corrected.absolute()), batch_size=batch_size)
    ref_transcripts = [hyp.text for hyp in transcriptions]

    all_hyps = [None for _ in range(len(manifest))]
    decoding_computer: GreedyBatchedLabelLoopingComputerBase = model.decoding.decoding.decoding_computer
    assert batch_size < len(
        manifest
    ), "Batch size should be less than the number of records in the manifest for continuous streaming test."
    with torch.no_grad(), torch.inference_mode():
        # get first 2 batches
        encoder_output, encoder_output_len = get_batch_encoder_outputs_from_records(
            manifest[:batch_size], model=model, device=device
        )
        encoder_output_next, encoder_output_len_next = get_batch_encoder_outputs_from_records(
            manifest[batch_size : batch_size + batch_size], model=model, device=device
        )
        # we always work with encoder_output, getting next utterances from encoder_output_next
        # so we need to pad encoder_output if it is shorter than encoder_output_next
        if encoder_output.shape[2] < encoder_output_next.shape[2]:
            encoder_output = F.pad(encoder_output, (0, encoder_output_next.shape[2] - encoder_output.shape[2]))
        expanded_batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, chunk_size)
        next_batch_i = 0
        next_batch_global_i = batch_size
        next_query_utterance_i = batch_size + batch_size
        has_next = True  # if we have anything in next batch to decode
        hyps: list[Hypothesis | None] = [None for _ in range(batch_size)]
        hyps_global_indices = list(range(batch_size))
        encoder_output_t = torch.zeros_like(encoder_output_len)
        state = None  # decoding state
        # while there is something to decode
        while ((rest_len := encoder_output_len - encoder_output_t) > 0).any():
            frame_indices = encoder_output_t[:, None] + torch.arange(chunk_size, device=device)[None, :]
            frame_indices = torch.minimum(frame_indices, encoder_output_len[:, None] - 1)
            current_len = torch.full_like(encoder_output_len, fill_value=chunk_size)
            current_len = torch.minimum(current_len, rest_len)
            encoder_frames = encoder_output[expanded_batch_indices, :, frame_indices]
            batched_hyps, _, state = decoding_computer(
                x=encoder_frames,
                out_len=current_len,
                prev_batched_state=state,
            )
            hyps_continuations = batched_hyps_to_hypotheses(batched_hyps, None, batch_size=batch_size)
            for i, (hyp, hyp_continuation) in enumerate(zip(hyps, hyps_continuations)):
                if hyp is None:
                    hyps[i] = hyp_continuation
                else:
                    hyp.merge_(hyp_continuation)
            encoder_output_t += current_len
            rest_len -= current_len

            decoding_computer.reset_state_by_mask(state, rest_len == 0)
            finished_decoding_indices = torch.nonzero(rest_len == 0, as_tuple=True)[0].cpu().tolist()
            for idx in finished_decoding_indices:
                hyp = hyps[idx]
                if all_hyps[hyps_global_indices[idx]] is None:
                    all_hyps[hyps_global_indices[idx]] = hyp
                hyps[idx] = None  # reset to None
                if has_next:
                    # get next utterance to decode for finished hypothesis
                    encoder_output[idx] = encoder_output_next[next_batch_i]
                    encoder_output_len[idx] = encoder_output_len_next[next_batch_i]
                    hyps_global_indices[idx] = next_batch_global_i
                    encoder_output_t[idx] = 0
                    next_batch_i += 1
                    next_batch_global_i += 1
                    # if next_batch_i is out of bounds, get next batch of encoder outputs
                    if next_batch_i >= encoder_output_len_next.shape[0]:
                        if next_query_utterance_i < len(manifest):
                            encoder_output_next, encoder_output_len_next = get_batch_encoder_outputs_from_records(
                                manifest[next_query_utterance_i : next_query_utterance_i + batch_size],
                                model=model,
                                device=device,
                            )
                            # pad if needed to allow futher assignment of encoder_output_next to encoder_output
                            if encoder_output.shape[2] < encoder_output_next.shape[2]:
                                encoder_output = F.pad(
                                    encoder_output, (0, encoder_output_next.shape[2] - encoder_output.shape[2])
                                )
                            next_batch_i = 0
                            next_query_utterance_i += batch_size
                        else:
                            has_next = False

    streaming_transcripts = []
    for hyp in all_hyps:
        streaming_transcripts.append(model.tokenizer.ids_to_text(hyp.y_sequence.tolist()))
    assert ref_transcripts == streaming_transcripts


@pytest.mark.with_downloads
@pytest.mark.parametrize(
    "device,use_cuda_graph_decoder",
    [(device, False) for device in DEVICES] + [(device, True) for device in DEVICES if device.type == "cuda"],
)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_label_looping_continuous_streaming_partial_hypotheses(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    device: torch.device,
    use_cuda_graph_decoder: bool,
    is_tdt: bool,
    chunk_size: int,
    batch_size: int,
    max_symbols: int,
):
    """Test streaming continuos decoding with partial hypotheses"""
    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large
    model.to(device=device)

    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    decoding_cfg.strategy = "greedy_batch"
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
        decoding_cfg.greedy.max_symbols = max_symbols
    model.change_decoding_strategy(decoding_cfg)

    manifest = read_manifest(an4_val_manifest_corrected)
    transcriptions = model.transcribe(audio=str(an4_val_manifest_corrected.absolute()), batch_size=batch_size)
    ref_transcripts = [hyp.text for hyp in transcriptions]

    all_hyps = [None for _ in range(len(manifest))]
    rnnt_infer = model.decoding.decoding
    assert batch_size < len(
        manifest
    ), "Batch size should be less than the number of records in the manifest for continuous streaming test."
    with torch.no_grad(), torch.inference_mode():
        # get first 2 batches
        encoder_output, encoder_output_len = get_batch_encoder_outputs_from_records(
            manifest[:batch_size], model=model, device=device
        )
        encoder_output_next, encoder_output_len_next = get_batch_encoder_outputs_from_records(
            manifest[batch_size : batch_size + batch_size], model=model, device=device
        )
        # we always work with encoder_output, getting next utterances from encoder_output_next
        # so we need to pad encoder_output if it is shorter than encoder_output_next
        if encoder_output.shape[2] < encoder_output_next.shape[2]:
            encoder_output = F.pad(encoder_output, (0, encoder_output_next.shape[2] - encoder_output.shape[2]))
        expanded_batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, chunk_size)
        # NB: we assume that encoder_output_len and encoder_output_len_next
        # have no zero elements (no empty utterances), and we do not check this condition further
        next_batch_i = 0
        next_batch_global_i = batch_size
        next_query_utterance_i = batch_size + batch_size
        has_next = True  # if we have anything in next batch to decode
        hyps: list[Hypothesis | None] = [None for _ in range(batch_size)]
        hyps_global_indices = list(range(batch_size))
        encoder_output_t = torch.zeros_like(encoder_output_len)
        # while there is something to decode
        while ((rest_len := encoder_output_len - encoder_output_t) > 0).any():
            frame_indices = encoder_output_t[:, None] + torch.arange(chunk_size, device=device)[None, :]
            frame_indices = torch.minimum(frame_indices, encoder_output_len[:, None] - 1)
            current_len = torch.full_like(encoder_output_len, fill_value=chunk_size)
            current_len = torch.minimum(current_len, rest_len)
            encoder_frames = encoder_output[expanded_batch_indices, :, frame_indices].transpose(1, 2)
            (hyps,) = rnnt_infer(
                encoder_output=encoder_frames,
                encoded_lengths=current_len,
                partial_hypotheses=hyps,
            )
            encoder_output_t += current_len
            rest_len -= current_len
            finished_decoding_indices = torch.nonzero(rest_len == 0, as_tuple=True)[0].cpu().tolist()
            for idx in finished_decoding_indices:
                hyp = hyps[idx]
                all_hyps[hyps_global_indices[idx]] = hyp
                # NB: we clean decoding state and set hyp to None only if we have next utterances to decode
                # otherwise for each decoder invocation with 0 length it will recreate the hypothesis object,
                # which is computationally expensive
                # decoding current hyp with 0 length will not change the hypothesis
                if has_next:
                    hyp.clean_decoding_state_()
                    hyps[idx] = None  # reset to None
                    # get next utterance to decode for finished hypothesis
                    encoder_output[idx] = encoder_output_next[next_batch_i]
                    encoder_output_len[idx] = encoder_output_len_next[next_batch_i]
                    hyps_global_indices[idx] = next_batch_global_i
                    encoder_output_t[idx] = 0
                    next_batch_i += 1
                    next_batch_global_i += 1
                    # if next_batch_i is out of bounds, get next batch of encoder outputs
                    if next_batch_i >= encoder_output_len_next.shape[0]:
                        if next_query_utterance_i < len(manifest):
                            encoder_output_next, encoder_output_len_next = get_batch_encoder_outputs_from_records(
                                manifest[next_query_utterance_i : next_query_utterance_i + batch_size],
                                model=model,
                                device=device,
                            )
                            # pad if needed to allow futher assignment of encoder_output_next to encoder_output
                            if encoder_output.shape[2] < encoder_output_next.shape[2]:
                                encoder_output = F.pad(
                                    encoder_output, (0, encoder_output_next.shape[2] - encoder_output.shape[2])
                                )
                            next_batch_i = 0
                            next_query_utterance_i += batch_size
                        else:
                            has_next = False
    for hyp in hyps:
        if hyp is not None:
            hyp.clean_decoding_state_()

    streaming_transcripts = []
    for hyp in all_hyps:
        streaming_transcripts.append(model.tokenizer.ids_to_text(hyp.y_sequence.tolist()))
    assert ref_transcripts == streaming_transcripts


@pytest.mark.with_downloads
@pytest.mark.parametrize(
    "device,use_cuda_graph_decoder",
    [(device, False) for device in DEVICES] + [(device, True) for device in DEVICES if device.type == "cuda"],
)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1])  # Small chunk size to trigger more state updates
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_label_looping_streaming_boosting_with_ref_transcripts(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    device: torch.device,
    use_cuda_graph_decoder: bool,
    is_tdt: bool,
    chunk_size: int,
    batch_size: int,
    max_symbols: int,
):
    """
    Metamorphic test: boosting with reference transcripts should yield identical results.

    This test validates that when we boost with the exact transcripts that the model
    would produce without boosting, the results remain the same. This is a metamorphic
    property that should hold for correct implementations.

    This test specifically validates the fix for TDT streaming boosting where
    fusion states were incorrectly updated using `active_mask` instead of `found_labels_mask`.
    """
    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large
    model.eval()
    model.to(device=device)

    # First, get reference transcripts without boosting
    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    decoding_cfg.strategy = "greedy_batch"
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
        decoding_cfg.greedy.max_symbols = max_symbols
    model.change_decoding_strategy(decoding_cfg)

    manifest = read_manifest(an4_val_manifest_corrected)
    transcriptions = model.transcribe(audio=str(an4_val_manifest_corrected.absolute()), batch_size=batch_size)
    ref_transcripts = [hyp.text for hyp in transcriptions]

    # Now set up per-stream boosting with reference transcripts
    decoding_cfg_boosted = copy.deepcopy(model.cfg.decoding)
    decoding_cfg_boosted.strategy = "greedy_batch"
    with open_dict(decoding_cfg_boosted):
        decoding_cfg_boosted.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
        decoding_cfg_boosted.greedy.max_symbols = max_symbols
        decoding_cfg_boosted.greedy.enable_per_stream_biasing = True
    model.change_decoding_strategy(decoding_cfg_boosted)

    all_hyps = []
    decoding_computer: GreedyBatchedLabelLoopingComputerBase = model.decoding.decoding.decoding_computer

    with torch.no_grad(), torch.inference_mode():
        for i in range(0, len(manifest), batch_size):
            batch_records = manifest[i : i + batch_size]
            batch_ref_transcripts = ref_transcripts[i : i + batch_size]

            encoder_output, encoder_output_len = get_batch_encoder_outputs_from_records(
                batch_records, model=model, device=device
            )
            local_batch_size = encoder_output_len.shape[0]

            # Create biasing requests for each sample in the batch
            biasing_requests = []
            multi_biasing_ids = torch.full([local_batch_size], fill_value=-1, dtype=torch.long, device=device)

            for batch_idx, ref_text in enumerate(batch_ref_transcripts):
                if ref_text:  # Only boost non-empty transcripts
                    request = BiasingRequestItemConfig(
                        boosting_model_cfg=BoostingTreeModelConfig(key_phrases_list=[ref_text], unk_score=-100),
                        boosting_model_alpha=10.0,
                    )
                    request.add_to_multi_model(
                        tokenizer=model.tokenizer,
                        biasing_multi_model=decoding_computer.biasing_multi_model,
                    )
                    if request.multi_model_id is not None:
                        multi_biasing_ids[batch_idx] = request.multi_model_id
                    biasing_requests.append(request)
                else:
                    biasing_requests.append(None)

            # Decode encoder output by chunks, passing state between decoder invocations
            state: Optional[BatchedLabelLoopingState] = None
            batched_hyps: BatchedHyps | None = None
            encoder_output = encoder_output.transpose(1, 2)

            for t in range(0, encoder_output.shape[1], chunk_size):
                rest_len = encoder_output_len - t
                current_len = torch.full_like(encoder_output_len, fill_value=chunk_size)
                current_len = torch.minimum(current_len, rest_len)
                current_len = torch.maximum(current_len, torch.zeros_like(current_len))

                batched_hyps_chunk, _, state = decoding_computer(
                    x=encoder_output[:, t : t + chunk_size],
                    out_len=current_len,
                    prev_batched_state=state,
                    multi_biasing_ids=multi_biasing_ids,
                )

                if batched_hyps is None:
                    batched_hyps = batched_hyps_chunk
                else:
                    batched_hyps.merge_(batched_hyps_chunk)

            # Clean up biasing models
            for request in biasing_requests:
                if request is not None and request.multi_model_id is not None:
                    decoding_computer.biasing_multi_model.remove_model(request.multi_model_id)
                    request.multi_model_id = None

            assert batched_hyps is not None
            all_hyps.extend(batched_hyps_to_hypotheses(batched_hyps, None, batch_size=local_batch_size))

    streaming_transcripts = []
    for hyp in all_hyps:
        streaming_transcripts.append(model.tokenizer.ids_to_text(hyp.y_sequence.tolist()))

    # The key assertion: boosting with ref transcripts should yield same results
    assert ref_transcripts == streaming_transcripts, (
        f"Boosting with reference transcripts should yield identical results.\n"
        f"This failure indicates a bug in fusion state handling during streaming decoding.\n"
        f"Differences found:\n"
        + "\n".join(
            f"  [{i}] ref: {ref!r} != boosted: {boosted!r}"
            for i, (ref, boosted) in enumerate(zip(ref_transcripts, streaming_transcripts))
            if ref != boosted
        )
    )
