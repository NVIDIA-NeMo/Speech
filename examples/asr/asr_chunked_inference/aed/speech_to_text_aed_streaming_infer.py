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

"""
Script to perform streaming inference using AED (Canary) models.
The current solution supports Wait-k and AlignAtt streaming decoding policies. 

Wait-k policy predicts only one token per each new speech chunk that increases the overall latency.
In this case, it is unclear at what point you can forget part of the left context when recognizing with a limited buffer size.
It is recommended to set the left context to the maximum possible value (infinite left context) for the Wait-k policy.

AlignAtt policy predicts tokens according to the cross-attention condition per each next token prediction.
If the condition is met, then the audio size does not need to be increased, and the prediction of the next token continues.
Otherwise, the audio buffer size needs to be increased. This policy shows lower latency in comparison with Wait-k.
This policy is also suitable for window recognition (left context is fixed and not infinite), but you can lose some accuracy in this case.

Recommended settings for audio buffer:
- streaming with 1.5s latency: 10-1-0.5 (10s left, 1s chunk, 0.5s right)

Example usage:

```shell
python speech_to_text_aed_streaming_infer.py \
    pretrained_name=nvidia/canary-1b-v2 \
    model_path=null \
    audio_dir="<optional path to folder of audio files>" \
    dataset_manifest="<optional path to manifest>" \
    output_filename="<optional output filename>" \
    right_context_secs=0.5 \
    chunk_secs=1.0 \
    left_context_secs=10.0 \
    batch_size=32 \
    clean_groundtruth_text=False \
    langid='en' \
    decoding.streaming_policy=alignatt \
    
```
"""
import copy
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
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
from nemo.collections.asr.parts.utils.eval_utils import cal_write_text_metric, cal_write_wer
from nemo.collections.asr.parts.utils.manifest_utils import filepath_to_absolute, read_manifest
from nemo.collections.asr.parts.utils.streaming_utils import (
    AudioBatch,
    ContextSize,
    SimpleAudioDataset,
    StreamingBatchedAudioBuffer,
)
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    get_inference_device,
    get_inference_dtype,
    setup_model,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging


def make_divisible_by(num, factor: int) -> int:
    """Make num divisible by factor"""
    return (num // factor) * factor


@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for buffered inference.
    """

    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Chunked configs
    chunk_secs: float = 2  # Chunk length in seconds
    left_context_secs: float = (
        10.0  # left context: larger value improves quality without affecting theoretical latency
    )
    right_context_secs: float = 2  # right context

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = True  # allow to select MPS device (Apple Silicon M-series GPU)
    compute_dtype: Optional[str] = (
        None  # "float32", "bfloat16" or "float16"; if None (default): bfloat16 if available, else float32
    )
    matmul_precision: str = "high"  # Literal["highest", "high", "medium"]
    audio_type: str = "wav"
    sort_input_manifest: bool = True

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for RNNT models
    decoding: AEDStreamingDecodingConfig = field(default_factory=AEDStreamingDecodingConfig)

    # Config for word / character error rate calculation
    calculate_wer: bool = True  # for ASR task
    calculate_bleu: bool = False  # for AST task
    calculate_latency: bool = True
    clean_groundtruth_text: bool = False
    ignore_capitalization: bool = True
    ignore_punctuation: bool = True
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False

    # extra arguments for Canary prompt generation
    presort_manifest: bool = True
    return_hypotheses: bool = False
    channel_selector: Optional[int] = None
    gt_text_attr_name: str = "text"
    gt_lang_attr_name: str = "source_lang"
    timestamps: bool = False
    prompt: dict = field(default_factory=dict)

    # debug mode
    debug_mode: bool = False


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    """
    Transcribes the input audio and can be used to infer long audio files by chunking
    them into smaller segments.
    """
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    cfg = OmegaConf.structured(cfg)

    if cfg.decoding.streaming_policy not in {"alignatt", "waitk"}:
        raise ValueError("This script currently supports only `alignatt` or `waitk` streaming policy")

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    filepaths = None
    manifest = cfg.dataset_manifest
    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
        manifest = None  # ignore dataset_manifest if audio_dir and dataset_manifest both presents

    # setup device
    map_location = get_inference_device(cuda=cfg.cuda, allow_mps=cfg.allow_mps)
    compute_dtype = get_inference_dtype(cfg.compute_dtype, device=map_location)

    logging.info(f"Inference will be done on device : {map_location} with compute_dtype: {compute_dtype}")

    # setup ASR model
    asr_model, model_name = setup_model(cfg, map_location)

    model_cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    # some changes for streaming scenario
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0

    if model_cfg.preprocessor.normalize != "per_feature":
        logging.error("Only MultitaskAED models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(model_cfg.preprocessor, True)

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    asr_model.freeze()
    asr_model = asr_model.to(asr_model.device)
    asr_model.to(compute_dtype)

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        multitask_decoding = MultiTaskDecodingConfig()
        multitask_decoding.strategy = "greedy"
        asr_model.change_decoding_strategy(multitask_decoding)

    logging.setLevel(logging.INFO)

    if manifest is not None:
        records = read_manifest(manifest)
        manifest_dir = Path(manifest).parent.absolute()
        # fix relative paths
        for record in records:
            record["audio_filepath"] = str(filepath_to_absolute(record["audio_filepath"], manifest_dir))
        # sort the samples by duration to reduce batched decoding time (could be 2 times faster than default random order)
        if cfg.sort_input_manifest:
            records = sorted(records, key=lambda x: x['duration'], reverse=True)
    else:
        assert filepaths is not None
        records = [{"audio_filepath": audio_file} for audio_file in filepaths]
    assert len(records) > 0, "No audio files found"

    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.eval()

    audio_sample_rate = model_cfg.preprocessor['sample_rate']

    feature_stride_sec = model_cfg.preprocessor['window_stride']
    features_per_sec = 1.0 / feature_stride_sec
    encoder_subsampling_factor = asr_model.encoder.subsampling_factor

    features_frame2audio_samples = make_divisible_by(
        int(audio_sample_rate * feature_stride_sec), factor=encoder_subsampling_factor
    )
    encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor

    context_encoder_frames = ContextSize(
        left=int(cfg.left_context_secs * features_per_sec / encoder_subsampling_factor),
        chunk=int(cfg.chunk_secs * features_per_sec / encoder_subsampling_factor),
        right=int(cfg.right_context_secs * features_per_sec / encoder_subsampling_factor),
    )
    context_samples = ContextSize(
        left=context_encoder_frames.left * encoder_subsampling_factor * features_frame2audio_samples,
        chunk=context_encoder_frames.chunk * encoder_subsampling_factor * features_frame2audio_samples,
        right=context_encoder_frames.right * encoder_subsampling_factor * features_frame2audio_samples,
    )

    logging.info(
        "Corrected contexts (sec): "
        f"Left {context_samples.left / audio_sample_rate:.2f}, "
        f"Chunk {context_samples.chunk / audio_sample_rate:.2f}, "
        f"Right {context_samples.right / audio_sample_rate:.2f}"
    )
    logging.info(f"Corrected contexts (subsampled encoder frames): {context_encoder_frames}")
    logging.info(f"Corrected contexts (in audio samples): {context_samples}")
    latency_secs = (context_samples.chunk + context_samples.right) / audio_sample_rate
    logging.info(f"Theoretical latency: {latency_secs:.2f} seconds")

    audio_dataset = SimpleAudioDataset(
        audio_filenames=[record["audio_filepath"] for record in records], sample_rate=audio_sample_rate
    )
    audio_dataloader = DataLoader(
        dataset=audio_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=AudioBatch.collate_fn,
        drop_last=False,
        in_order=True,
    )

    decoding_computer = GreedyBatchedStreamingAEDComputer(
        asr_model,
        frame_chunk_size=context_encoder_frames.chunk,
        decoding_cfg=cfg.decoding,
    )

    decoder_input_ids = return_decoder_input_ids(cfg, asr_model)

    with torch.no_grad(), torch.inference_mode():
        all_hyps = []
        tokens_frame_alignment = []
        predicted_token_ids = []
        audio_data: AudioBatch
        for audio_data in tqdm(audio_dataloader):
            # get audio
            # NB: preprocessor runs on torch.float32, no need to cast dtype here
            audio_batch = audio_data.audio_signals.to(device=map_location)
            audio_batch_lengths = audio_data.audio_signal_lengths.to(device=map_location)
            batch_size = audio_batch.shape[0]
            device = audio_batch.device

            # initialize aed model state
            model_state = GreedyBatchedStreamingAEDComputer.initialize_aed_model_state(
                asr_model=asr_model,
                decoder_input_ids=decoder_input_ids,
                batch_size=batch_size,
                context_encoder_frames=context_encoder_frames,
                chunk_secs=cfg.chunk_secs,
                right_context_secs=cfg.right_context_secs,
            )

            # decode audio by chunks
            left_sample = 0
            end_of_window_sample = min(context_samples.chunk, audio_batch.shape[1])
            # right_sample = initial latency in audio samples
            right_sample = min(context_samples.chunk + context_samples.right, audio_batch.shape[1])
            # start with empty buffer
            buffer = StreamingBatchedAudioBuffer(
                batch_size=batch_size,
                context_samples=context_samples,
                dtype=audio_batch.dtype,
                device=device,
            )
            rest_audio_lengths = audio_batch_lengths.clone()
            is_last_window_batch = torch.zeros_like(rest_audio_lengths)

            step_idx = 0
            # iterate over audio samples
            while left_sample < audio_batch.shape[1]:
                # add samples to buffer
                chunk_length = min(right_sample, audio_batch.shape[1]) - left_sample  # M + R context
                is_last_chunk_batch = chunk_length >= rest_audio_lengths
                is_last_chunk = right_sample >= audio_batch.shape[1]
                chunk_lengths_batch = torch.where(
                    is_last_chunk_batch,
                    rest_audio_lengths,
                    torch.full_like(rest_audio_lengths, fill_value=chunk_length),
                )
                is_last_window_batch = end_of_window_sample >= audio_batch_lengths
                is_last_window_batch += is_last_chunk

                buffer.add_audio_batch_(
                    audio_batch[:, left_sample:right_sample],
                    audio_lengths=chunk_lengths_batch,
                    is_last_chunk=is_last_chunk,
                    is_last_chunk_batch=torch.zeros_like(is_last_chunk_batch),
                )

                model_state.is_last_chunk_batch = is_last_chunk_batch

                # get encoder output using full buffer [left-chunk-right]
                _, encoded_len, enc_states, _ = asr_model(
                    input_signal=buffer.samples, input_signal_length=buffer.context_size_batch.total()
                )
                # remove right context from encoder output length (only for non-last chunks)
                encoder_context_batch = buffer.context_size_batch.subsample(factor=encoder_frame2audio_samples)
                encoded_len_no_rc = encoder_context_batch.left + encoder_context_batch.chunk
                encoded_length_corrected = torch.where(is_last_chunk_batch, encoded_len, encoded_len_no_rc)
                encoder_input_mask = lens_to_mask(encoded_length_corrected, enc_states.shape[1]).to(enc_states.dtype)

                # keep track of the real frame position in the audio signal
                model_state.prev_encoder_shift = max(
                    end_of_window_sample // encoder_frame2audio_samples
                    - context_encoder_frames.left
                    - context_encoder_frames.chunk,
                    0,
                )
                # decode only chunk frames (controlled by encoder_context_batch.chunk)
                model_state = decoding_computer(
                    encoder_output=enc_states,
                    encoder_output_len=encoded_length_corrected,
                    encoder_input_mask=encoder_input_mask,
                    prev_batched_state=model_state,
                )
                # move to next sample
                rest_audio_lengths -= chunk_lengths_batch
                left_sample = right_sample
                right_sample = min(right_sample + context_samples.chunk, audio_batch.shape[1])  # add next chunk
                end_of_window_sample += context_samples.chunk  # add next chunk
                step_idx += 1

            # get final results for each sample in the batch
            for i in range(batch_size):
                transcription_idx = model_state.pred_tokens_ids[
                    i, model_state.decoder_input_ids.size(-1) : model_state.current_context_lengths[i]
                ]
                transcription = asr_model.tokenizer.ids_to_text(transcription_idx.tolist()).strip()
                logging.info(f"i: {transcription}")
                all_hyps.append(transcription)
                tokens_frame_alignment.append(model_state.tokens_frame_alignment[i])
                predicted_token_ids.append(
                    model_state.pred_tokens_ids[
                        i, model_state.decoder_input_ids.size(-1) : model_state.current_context_lengths[i]
                    ]
                )

    # write predictions to outputfile
    output_filename = cfg.output_filename
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    with open(output_filename, "w") as out_f:
        for i, record in enumerate(records):
            record["pred_text"] = all_hyps[i]
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logging.info(f"Finished writing predictions to {output_filename}!")

    # calculate WER for ASR task
    if cfg.calculate_wer:
        output_manifest_w_wer, total_res, _ = cal_write_wer(
            pred_manifest=output_filename,
            pred_text_attr_name="pred_text",
            clean_groundtruth_text=cfg.clean_groundtruth_text,
            langid=cfg.langid,
            use_cer=cfg.use_cer,
            ignore_capitalization=cfg.ignore_capitalization,
            ignore_punctuation=cfg.ignore_punctuation,
            output_filename=None,
        )
        if output_manifest_w_wer:
            logging.info(f"Writing prediction and error rate of each sample to {output_manifest_w_wer}!")
            logging.info(f"{total_res}")

    # calculate BLEU for AST task
    if cfg.calculate_bleu:
        output_manifest_w_bleu, total_res, _ = cal_write_text_metric(
            pred_manifest=output_filename,
            pred_text_attr_name="pred_text",
            gt_text_attr_name="answer",
            metric="bleu",
        )
        if output_manifest_w_bleu:
            logging.info(f"Writing prediction and error rate of each sample to {output_manifest_w_bleu}!")
            logging.info(f"{total_res}")

    # compute decoding latency (LAAL)
    if cfg.calculate_latency:
        laal_list = None
        audio_encoder_fs = encoder_subsampling_factor * feature_stride_sec * 1000  # to ms
        if cfg.decoding.streaming_policy == "waitk":
            laal_list = decoding_computer.compute_waitk_lagging(
                records, predicted_token_ids, context_encoder_frames, audio_encoder_fs, BOW_PREFIX="\u2581"
            )
        elif cfg.decoding.streaming_policy == "alignatt":
            laal_list = decoding_computer.compute_alignatt_lagging(
                records,
                predicted_token_ids,
                tokens_frame_alignment,
                context_encoder_frames,
                audio_encoder_fs,
                BOW_PREFIX="\u2581",
            )
        laal = sum(laal_list) / len(laal_list)
        logging.info(f"Decoding latency (LAAL): {laal:.2f} ms")

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
