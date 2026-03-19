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

import editdistance
import pytest
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from nemo.collections.asr.models.aed_multitask_models import lens_to_mask
from nemo.collections.asr.parts.submodules.aed_decoding import (
    GreedyBatchedStreamingAEDComputer,
    return_decoder_input_ids,
)
from nemo.collections.asr.parts.submodules.multitask_decoding import (
    AEDStreamingDecodingConfig,
    MultiTaskDecodingConfig,
)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize
from tests.collections.asr.decoding.utils import load_audio, make_preprocessor_deterministic

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))

if torch.mps.is_available():
    DEVICES.append(torch.device("mps"))


def get_batch_encoder_outputs_from_records(records, model, device):
    """Helper function to get encoder outputs for a batch of manifest records"""
    local_batch_size = len(records)
    filenames = [record["audio_filepath"] for record in records]
    audio_filepaths = filenames[:local_batch_size]

    with torch.no_grad():
        make_preprocessor_deterministic(model)
        model.eval()

        all_inputs, all_lengths = [], []
        for audio_file in tqdm(audio_filepaths, desc="Loading audio files"):
            audio_tensor, _ = load_audio(audio_file)
            all_inputs.append(audio_tensor)
            all_lengths.append(torch.tensor(audio_tensor.shape[0], dtype=torch.int64))

        input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(
            device=device, dtype=torch.float32
        )
        length_batch = torch.tensor(all_lengths, dtype=torch.int64).to(device)
        # get encoder output using full audio signal
        _, encoded_length, encoded_output, _ = model(input_signal=input_batch, input_signal_length=length_batch)

    return encoded_output, encoded_length


@pytest.mark.with_downloads
@pytest.mark.parametrize(
    "device,use_cuda_graph_decoder",
    [(device, False) for device in DEVICES] + [(device, True) for device in DEVICES if device.type == "cuda"],
)
@pytest.mark.parametrize("decoding_policy", ["waitk", "alignatt"])
@pytest.mark.parametrize("chunk_size", [3, 4])
@pytest.mark.parametrize("batch_size", [4])
def test_multi_task_streaming_decoding(
    tmp_path_factory,
    an4_val_manifest_corrected,
    canary_180m_flash,
    device: torch.device,
    use_cuda_graph_decoder: bool,
    decoding_policy: str,
    chunk_size: int,
    batch_size: int,
):
    """Test streaming decoding with multi-task model for different decoding policies"""
    model = canary_180m_flash
    model.eval()
    model.to(device=device)

    # setup streaming decoding config
    streaming_decoding_cfg = AEDStreamingDecodingConfig()
    streaming_decoding_cfg.streaming_policy = decoding_policy
    streaming_decoding_cfg.chunk_secs = 1
    streaming_decoding_cfg.right_context_secs = 0.0
    streaming_decoding_cfg.batch_size = batch_size
    streaming_decoding_cfg.prompt = OmegaConf.create({'pnc': 'no', 'task': 'asr'})

    context_encoder_frames = ContextSize(
        left=100,
        chunk=chunk_size,
        right=0.0,
    )

    # setup decoding strategy
    if hasattr(model, 'change_decoding_strategy'):
        multitask_decoding = MultiTaskDecodingConfig()
        multitask_decoding.strategy = "greedy"
        model.change_decoding_strategy(multitask_decoding)

    manifest = read_manifest(an4_val_manifest_corrected)

    all_hyps = []
    tokens_frame_alignment = []
    predicted_token_ids = []
    decoding_computer = GreedyBatchedStreamingAEDComputer(
        model,
        frame_chunk_size=chunk_size,
        decoding_cfg=streaming_decoding_cfg,
    )

    with torch.no_grad(), torch.inference_mode():
        for i in range(0, len(manifest), batch_size):
            encoder_output, encoder_output_len = get_batch_encoder_outputs_from_records(
                manifest[i : i + batch_size], model=model, device=device
            )
            local_batch_size = encoder_output_len.shape[0]
            decoder_input_ids = return_decoder_input_ids(streaming_decoding_cfg, model)
            model_state = GreedyBatchedStreamingAEDComputer.initialize_aed_model_state(
                asr_model=model,
                decoder_input_ids=decoder_input_ids,
                batch_size=local_batch_size,
                context_encoder_frames=context_encoder_frames,
                chunk_secs=streaming_decoding_cfg.chunk_secs,
                right_context_secs=streaming_decoding_cfg.right_context_secs,
            )

            # decode encoder output by chunks, passing state between decoder invocations
            for t in range(0, encoder_output.shape[1], chunk_size):
                current_len = torch.full_like(encoder_output_len, fill_value=t + chunk_size)
                current_len = torch.minimum(current_len, encoder_output_len)
                model_state.is_last_chunk_batch = current_len >= encoder_output_len
                encoder_input_mask = lens_to_mask(current_len, encoder_output[:, : t + chunk_size].shape[1]).to(
                    encoder_output.dtype
                )

                model_state = decoding_computer(
                    encoder_output=encoder_output[:, : t + chunk_size],
                    encoder_output_len=current_len,
                    encoder_input_mask=encoder_input_mask,
                    prev_batched_state=model_state,
                )
            # get final results for each sample in the batch
            for j in range(local_batch_size):
                transcription_idx = model_state.pred_tokens_ids[
                    j, model_state.decoder_input_ids.size(-1) : model_state.current_context_lengths[j]
                ]
                transcription = model.tokenizer.ids_to_text(transcription_idx.tolist()).strip()
                all_hyps.append(transcription)
                tokens_frame_alignment.append(model_state.tokens_frame_alignment[j])
                predicted_token_ids.append(
                    model_state.pred_tokens_ids[
                        j, model_state.decoder_input_ids.size(-1) : model_state.current_context_lengths[j]
                    ]
                )

    # compare decoding results with reference transcripts
    ref_transcripts = [item['text'] for item in manifest]
    assert editdistance.eval(ref_transcripts, all_hyps) <= len(ref_transcripts) * 0.1  # Expected WER is less than 10%

    # compute latency
    audio_encoder_fs = 80  # in ms
    laal_list = None
    if decoding_policy == "waitk":
        laal_list = decoding_computer.compute_waitk_lagging(
            manifest, predicted_token_ids, context_encoder_frames, audio_encoder_fs, BOW_PREFIX="\u2581"
        )
    elif decoding_policy == "alignatt":
        laal_list = decoding_computer.compute_alignatt_lagging(
            manifest,
            predicted_token_ids,
            tokens_frame_alignment,
            context_encoder_frames,
            audio_encoder_fs,
            BOW_PREFIX="\u2581",
        )
    else:
        raise ValueError(f"Decoding policy {decoding_policy} is not supported")
    laal = sum(laal_list) / len(laal_list)
    assert 300 <= laal <= 900  # Expected LAAL is between 300ms and 900ms depending on the decoding policy
