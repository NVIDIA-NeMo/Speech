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
Script to perform buffered and streaming inference using RNNT models.

Buffered inference is the primary form of audio transcription when the audio segment is longer than 20-30 seconds.
Also, this is a demonstration of the algorithm that can be used for streaming inference with low latency.
This is especially useful for models such as Conformers, which have quadratic time and memory scaling with
audio duration.

The difference between streaming and buffered inference is the chunk size (or the latency of inference).
Buffered inference will use large chunk sizes (5-10 seconds) + some additional right for context.
Streaming inference will use small chunk sizes (0.1 to 0.25 seconds) + some additional right buffer for context.
Theoretical latency (latency without model inference time) is the sum of the chunk size and the right context.
Keeping large left context (~10s) is not required, but can improve the quality of transcriptions.

Recommended settings:
- long file transcription: in most cases 10-10-5 (10s left, 10s chunk, 5s right) will give results similar to offline
- streaming with 4s latency: 10-2-2 is usually similar or better than 10-0.16-3.84 and significantly faster

Example usage:

```shell
python speech_to_text_streaming_infer_rnnt.py \
    pretrained_name=nvidia/parakeet-rnnt-1.1b \
    model_path=null \
    audio_dir="<optional path to folder of audio files>" \
    dataset_manifest="<optional path to manifest>" \
    output_filename="<optional output filename>" \
    right_context_secs=2.0 \
    chunk_secs=2 \
    left_context_secs=10.0 \
    batch_size=32 \
    clean_groundtruth_text=False \
    langid='en'
```
"""
import copy
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# NB: PYTORCH_CUDA_ALLOC_CONF should be set before importing pytorch / nemo
# using expandable_segments can save more than 10x GPU memory when using small chunks
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in alloc_conf:
    if len(alloc_conf) > 0:
        alloc_conf += ",expandable_segments:True"
    else:
        alloc_conf = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf


import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nemo.collections.asr.models import EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    GreedyBatchedLabelLoopingComputerBase,
)
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.manifest_utils import filepath_to_absolute, read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import BatchedHyps, batched_hyps_to_hypotheses
from nemo.collections.asr.parts.utils.streaming_utils import (
    AudioBatch,
    ContextSize,
    SimpleAudioDataset,
    StreamingBatchedAudioBuffer,
)
from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    get_inference_device,
    get_inference_dtype,
    setup_model,
    write_transcription,
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
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
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

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for RNNT models
    decoding: RNNTDecodingConfig = field(default_factory=RNNTDecodingConfig)

    timestamps: bool = False  # output timestamps

    # Config for word / character error rate calculation
    calculate_wer: bool = True
    clean_groundtruth_text: bool = False
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False


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

    asr_model, model_name = setup_model(cfg, map_location)

    model_cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    # some changes for streaming scenario
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0

    if model_cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecRNNTBPEModel models trained with per_feature normalization are supported currently")

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

    # Change Decoding Config
    with open_dict(cfg.decoding):
        if cfg.decoding.strategy != "greedy_batch" or cfg.decoding.greedy.loop_labels is not True:
            raise NotImplementedError(
                "This script currently supports only `greedy_batch` strategy with Label-Looping algorithm"
            )
        cfg.decoding.tdt_include_token_duration = cfg.timestamps
        cfg.decoding.greedy.preserve_alignments = False
        cfg.decoding.fused_batch_size = -1  # temporarily stop fused batch during inference.
        cfg.decoding.beam.return_best_hypothesis = True  # return and write the best hypothsis only

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        if not isinstance(asr_model, EncDecRNNTModel) and not isinstance(asr_model, EncDecHybridRNNTCTCModel):
            raise ValueError("The script supports rnnt model and hybrid model with rnnt decodng!")
        else:
            # rnnt model
            if isinstance(asr_model, EncDecRNNTModel):
                asr_model.change_decoding_strategy(cfg.decoding)

            # hybrid ctc rnnt model with decoder_type = rnnt
            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(cfg.decoding, decoder_type='rnnt')

    if manifest is not None:
        records = read_manifest(manifest)
        manifest_dir = Path(manifest).parent.absolute()
        # fix relative paths
        for record in records:
            record["audio_filepath"] = str(filepath_to_absolute(record["audio_filepath"], manifest_dir))
    else:
        assert filepaths is not None
        records = [{"audio_filepath": audio_file} for audio_file in filepaths]

    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.eval()

    decoding_computer: GreedyBatchedLabelLoopingComputerBase = asr_model.decoding.decoding.decoding_computer

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

    with torch.no_grad(), torch.inference_mode():
        all_hyps = []
        audio_data: AudioBatch
        for audio_data in tqdm(audio_dataloader):
            # get audio
            # NB: preprocessor runs on torch.float32, no need to cast dtype here
            audio_batch = audio_data.audio_signals.to(device=map_location)
            audio_batch_lengths = audio_data.audio_signal_lengths.to(device=map_location)
            batch_size = audio_batch.shape[0]
            device = audio_batch.device

            # decode audio by chunks

            current_batched_hyps: BatchedHyps | None = None
            state = None
            left_sample = 0
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

            # iterate over audio samples
            while left_sample < audio_batch.shape[1]:
                # add samples to buffer
                chunk_length = min(right_sample, audio_batch.shape[1]) - left_sample
                is_last_chunk_batch = chunk_length >= rest_audio_lengths
                is_last_chunk = right_sample >= audio_batch.shape[1]
                chunk_lengths_batch = torch.where(
                    is_last_chunk_batch,
                    rest_audio_lengths,
                    torch.full_like(rest_audio_lengths, fill_value=chunk_length),
                )
                buffer.add_audio_batch_(
                    audio_batch[:, left_sample:right_sample],
                    audio_lengths=chunk_lengths_batch,
                    is_last_chunk=is_last_chunk,
                    is_last_chunk_batch=is_last_chunk_batch,
                )

                # get encoder output using full buffer [left-chunk-right]
                encoder_output, encoder_output_len = asr_model(
                    input_signal=buffer.samples,
                    input_signal_length=buffer.context_size_batch.total(),
                )
                encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
                # remove extra context from encoder_output (leave only frames corresponding to the chunk)
                encoder_context = buffer.context_size.subsample(factor=encoder_frame2audio_samples)
                encoder_context_batch = buffer.context_size_batch.subsample(factor=encoder_frame2audio_samples)
                # remove left context
                encoder_output = encoder_output[:, encoder_context.left :]

                # decode only chunk frames
                chunk_batched_hyps, _, state = decoding_computer(
                    x=encoder_output,
                    out_len=torch.where(
                        is_last_chunk_batch,
                        encoder_output_len - encoder_context_batch.left,
                        encoder_context_batch.chunk,
                    ),
                    prev_batched_state=state,
                )
                # merge hyps with previous hyps
                if current_batched_hyps is None:
                    current_batched_hyps = chunk_batched_hyps
                else:
                    current_batched_hyps.merge_(chunk_batched_hyps)

                # move to next sample
                rest_audio_lengths -= chunk_lengths_batch
                left_sample = right_sample
                right_sample = min(right_sample + context_samples.chunk, audio_batch.shape[1])  # add next chunk

            all_hyps.extend(batched_hyps_to_hypotheses(current_batched_hyps, None, batch_size=batch_size))

    # convert text
    for i, hyp in enumerate(all_hyps):
        hyp.text = asr_model.tokenizer.ids_to_text(hyp.y_sequence.tolist())
        if cfg.timestamps:
            hyp = asr_model.decoding.compute_rnnt_timestamps(hyp)
            hyp = process_timestamp_outputs(
                hyp,
                subsampling_factor=asr_model.encoder.subsampling_factor,
                window_stride=asr_model.cfg['preprocessor']['window_stride'],
            )
            all_hyps[i] = hyp

    output_filename, pred_text_attr_name = write_transcription(
        all_hyps, cfg, model_name, filepaths=filepaths, compute_langs=False, timestamps=cfg.timestamps
    )
    logging.info(f"Finished writing predictions to {output_filename}!")

    if cfg.calculate_wer:
        output_manifest_w_wer, total_res, _ = cal_write_wer(
            pred_manifest=output_filename,
            pred_text_attr_name=pred_text_attr_name,
            clean_groundtruth_text=cfg.clean_groundtruth_text,
            langid=cfg.langid,
            use_cer=cfg.use_cer,
            output_filename=None,
        )
        if output_manifest_w_wer:
            logging.info(f"Writing prediction and error rate of each sample to {output_manifest_w_wer}!")
            logging.info(f"{total_res}")

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
