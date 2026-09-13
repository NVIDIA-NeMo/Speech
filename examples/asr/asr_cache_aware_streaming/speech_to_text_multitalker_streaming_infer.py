# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import time
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
from nemo.collections.asr.parts.utils.diarization_utils import (
    collect_diar_predictions,
    write_and_score_diar_predictions,
)
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
    SpeakerTaggedASR,
    add_delay_for_real_time,
    configure_diar_streaming,
    get_multi_talker_samples_from_manifest,
    set_batch_rttm_masks,
    validate_feature_frame_strides,
    write_seglst_file,
)
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class MultitalkerTranscriptionConfig:
    """
    Configuration for Multi-talker transcription with an ASR model and a diarization model.
    """

    # Required configs
    diar_model: Optional[str] = None  # Path to a .nemo file
    diar_pretrained_name: Optional[str] = None  # Name of a pretrained model
    max_num_of_spks: Optional[int] = 4  # maximum number of speakers
    parallel_speaker_strategy: bool = True  # whether to use parallel speaker strategy
    masked_asr: bool = False  # whether to use masked ASR
    mask_preencode: bool = False  # whether to mask preencode or mask features
    cache_gating: bool = True  # whether to use cache gating
    cache_gating_buffer_size: int = 2  # buffer size for cache gating
    single_speaker_mode: bool = False  # whether to use single speaker mode
    feat_len_sec: float = 0.01

    # General configs
    session_len_sec: float = -1  # End-to-end diarization session length in seconds
    num_workers: int = 8
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    log: bool = True  # If True,log will be printed
    precision: str = "bf16"  # "bf16" or "32"

    # Streaming diarization configs
    streaming_mode: bool = True  # If True, streaming diarization will be used.
    spkcache_len: Optional[int] = None
    spkcache_update_period: int = 144
    fifo_len: int = 188
    diar_right_context: int = 0  # extra right context size for diarization (will increase total latency)

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # ASR Configs
    asr_model: Optional[str] = None
    device: str = 'cuda'
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    att_context_size: Optional[List[int]] = field(default_factory=lambda: [70, 13])
    use_amp: bool = True
    debug_mode: bool = False
    deploy_mode: bool = False
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 5
    online_normalization: bool = False
    output_path: Optional[str] = None
    diar_output_rttm_dir: Optional[str] = None
    diar_collar: float = 0.0
    diar_ignore_overlap: bool = False
    pad_and_drop_preencoded: bool = False
    generate_realtime_scripts: bool = False
    spk_supervision: str = "diar"  # ["diar", "rttm"]
    binary_diar_preds: bool = True

    # Multitalker transcription configs
    verbose: bool = False
    word_window: int = 50
    sent_break_sec: float = 1.0  # minimum time gap between sentences
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    print_time: bool = True
    print_sample_indices: List[int] = field(default_factory=lambda: [0])
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: Optional[str] = None
    ignored_initial_frame_steps: int = 5
    finetune_realtime_ratio: float = 0.01


def launch_serial_streaming(
    cfg,
    asr_model,
    diar_model,
    streaming_buffer,
    pad_and_drop_preencoded=False,
):
    """
    Launch the serial streaming inference with ASR model and diarization model.

    Args:
        cfg (Any): The configuration object containing the parameters for the streaming inference.
        asr_model (Any): The ASR model loaded from the path provided in MultitalkerTranscriptionConfig.
        diar_model (Any): The diarization model loadded from the path provided in MultitalkerTranscriptionConfig.
        streaming_buffer: An iterator that yields chunks of audio data and their lengths.
        pad_and_drop_preencoded: A boolean flag indicating whether to pad and drop the extra pre-encoded tokens.
    """
    diar_right_offset = (
        0 if cfg.spk_supervision == "rttm" else cfg.diar_right_context * diar_model.encoder.subsampling_factor
    )
    streaming_buffer_iter = streaming_buffer.iter_with_right_context(diar_right_offset)

    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    feat_frame_count = 0
    session_start_time = time.time()
    for step_num, (chunk_audio, chunk_lengths, diar_chunk_audio, diar_chunk_lengths) in enumerate(
        streaming_buffer_iter
    ):
        drop_extra_pre_encoded = (
            0
            if step_num == 0 and not pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        loop_start_time = time.time()
        with torch.inference_mode():
            with autocast:
                with torch.no_grad():
                    multispk_asr_streamer.perform_serial_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        diar_chunk_audio=diar_chunk_audio,
                        diar_chunk_lengths=diar_chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                    )
        if cfg.get("real_time_mode", False):
            add_delay_for_real_time(
                cfg=cfg,
                chunk_audio=chunk_audio,
                session_start_time=session_start_time,
                feat_frame_count=feat_frame_count,
                loop_end_time=time.time(),
                loop_start_time=loop_start_time,
            )
        feat_frame_count += chunk_audio.shape[-1] - cfg.discarded_frames
    return multispk_asr_streamer


def launch_parallel_streaming(
    cfg,
    asr_model,
    diar_model,
    streaming_buffer,
    pad_and_drop_preencoded=False,
):
    diar_right_offset = (
        0 if cfg.spk_supervision == "rttm" else cfg.diar_right_context * diar_model.encoder.subsampling_factor
    )
    streaming_buffer_iter = streaming_buffer.iter_with_right_context(diar_right_offset)
    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    feat_frame_count = 0
    session_start_time = time.time()
    for step_num, (chunk_audio, chunk_lengths, diar_chunk_audio, diar_chunk_lengths) in enumerate(
        streaming_buffer_iter
    ):
        drop_extra_pre_encoded = (
            0
            if step_num == 0 and not pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        loop_start_time = time.time()
        with torch.inference_mode():
            with autocast:
                with torch.no_grad():
                    multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        diar_chunk_audio=diar_chunk_audio,
                        diar_chunk_lengths=diar_chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                    )
        if cfg.get("real_time_mode", False):
            add_delay_for_real_time(
                cfg=cfg,
                chunk_audio=chunk_audio,
                session_start_time=session_start_time,
                feat_frame_count=feat_frame_count,
                loop_end_time=time.time(),
                loop_start_time=loop_start_time,
            )
        feat_frame_count += chunk_audio.shape[-1] - cfg.discarded_frames
    return multispk_asr_streamer


@hydra_runner(config_name="MultitalkerTranscriptionConfig", schema=MultitalkerTranscriptionConfig)
def main(cfg: MultitalkerTranscriptionConfig) -> Union[MultitalkerTranscriptionConfig]:
    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.diar_model is None and cfg.diar_pretrained_name is None:
        raise ValueError("Both cfg.diar_model and cfg.pretrained_name cannot be None!")
    if cfg.audio_file is None and cfg.manifest_file is None:
        raise ValueError("Both cfg.audio_file and cfg.manifest_file cannot be None!")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    if cfg.diar_model.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(
            checkpoint_path=cfg.diar_model, map_location=map_location, strict=False
        )
    elif cfg.diar_model.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.diar_model, map_location=map_location)
    else:
        raise ValueError("cfg.diar_model must end with.ckpt or.nemo!")

    bf16_requested = str(cfg.precision).lower().startswith("bf16")
    use_bf16 = bf16_requested and cfg.use_amp and accelerator == "gpu" and torch.cuda.is_bf16_supported()
    if bf16_requested and not use_bf16:
        logging.warning("BF16 precision was requested but is unavailable. Falling back to FP32.")
    if use_bf16:
        diar_model = diar_model.to(dtype=torch.bfloat16).eval()
    else:
        diar_model = diar_model.to(dtype=torch.float32).eval()

    if cfg.audio_file is not None and cfg.manifest_file is not None:
        logging.warning("Both audio_file and manifest_file are specified. Audio_file will be used with top priority.")
    elif cfg.audio_file is not None:
        logging.info("audio_file is specified. Using audio_file as input.")
    elif cfg.manifest_file is not None:
        logging.info("manifest_file is specified. Using manifest_file as input.")
    else:
        raise ValueError("One of audio_file or manifest_file must be specified!")

    if cfg.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {cfg.asr_model}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=cfg.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {cfg.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=cfg.asr_model)

    if cfg.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    # Initialize to avoid "possibly used before assignment" error
    multispk_asr_streamer = None

    asr_model = asr_model.to(cfg.device)
    asr_model = asr_model.eval()

    global autocast
    autocast = torch.amp.autocast(
        device_type=asr_model.device.type,
        dtype=torch.bfloat16,
        enabled=use_bf16,
    )

    # chunk_size is set automatically for models trained for streaming.
    # For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
    if cfg.chunk_size > 0:
        if cfg.shift_size < 0:
            shift_size = cfg.chunk_size
        else:
            shift_size = cfg.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=cfg.chunk_size, left_chunks=cfg.left_chunks, shift_size=shift_size
        )
    logging.info(asr_model.encoder.streaming_cfg)
    validate_feature_frame_strides(asr_model=asr_model, diar_model=diar_model)
    diar_chunk_len = asr_model.encoder.streaming_cfg.valid_out_len + asr_model.encoder.streaming_cfg.cache_drop_size
    asr_output_subsampling_factor = asr_model.encoder.subsampling_factor
    diar_output_subsampling_factor = configure_diar_streaming(
        diar_model=diar_model,
        cfg=cfg,
        output_subsampling_factor=asr_output_subsampling_factor,
        diar_chunk_len=diar_chunk_len,
    )

    if isinstance(diar_model.encoder.pre_encode, FeatureStacking) and not cfg.pad_and_drop_preencoded:
        logging.info(
            "FeatureStacking diarization detected: enabling padded ASR pre-encode cache so diarization can consume "
            "aligned cache-free chunks."
        )
        cfg.pad_and_drop_preencoded = True

    seglst_dict_list = []
    diar_predictions, diar_samples = [], []
    if cfg.audio_file is not None:
        # Stream a single audio file
        samples = [
            {
                'audio_filepath': cfg.audio_file,
            }
        ]
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=asr_model,
            online_normalization=cfg.online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
        cfg.batch_size = len(samples)
        streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)
        if cfg.parallel_speaker_strategy:
            multispk_asr_streamer = launch_parallel_streaming(
                cfg=cfg,
                asr_model=asr_model,
                diar_model=diar_model,
                streaming_buffer=streaming_buffer,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
            )
            batch_seglst_list = multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
        else:
            multispk_asr_streamer = launch_serial_streaming(
                cfg=cfg,
                asr_model=asr_model,
                diar_model=diar_model,
                streaming_buffer=streaming_buffer,
            )
            batch_seglst_list = multispk_asr_streamer.generate_seglst_dicts_from_serial_streaming(samples=samples)
        if cfg.diar_output_rttm_dir is not None:
            feature_frame_length_sec = asr_model.cfg.preprocessor.window_stride
            batch_predictions, batch_metadata = collect_diar_predictions(
                diar_preds=multispk_asr_streamer.instance_manager.diar_states.diar_pred_out_stream,
                samples=samples,
                feature_lengths=streaming_buffer.streams_length,
                feature_frame_length_sec=feature_frame_length_sec,
                diar_frame_length_sec=feature_frame_length_sec * diar_output_subsampling_factor,
            )
            diar_predictions.extend(batch_predictions)
            diar_samples.extend(batch_metadata)
        seglst_dict_list.extend(batch_seglst_list)

    else:
        # Stream audio files in a manifest file in batched mode
        feat_per_sec = round(asr_model.cfg.preprocessor.window_stride * asr_model.cfg.encoder.subsampling_factor, 2)
        samples, rttms_mask_mats = get_multi_talker_samples_from_manifest(
            cfg, manifest_file=cfg.manifest_file, feat_per_sec=feat_per_sec, max_spks=cfg.max_num_of_spks
        )
        logging.info(f"Loaded {len(samples)} from the manifest at {cfg.manifest_file}.")

        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=asr_model,
            online_normalization=cfg.online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )

        batch_samples = []
        for sample_idx, sample in enumerate(samples):
            batch_samples.append(sample)
            streaming_buffer.append_audio_file(sample['audio_filepath'], stream_id=-1)
            logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')

            if (sample_idx + 1) % cfg.batch_size == 0 or sample_idx == len(samples) - 1:
                logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                if cfg.spk_supervision == "rttm":
                    set_batch_rttm_masks(
                        diar_model=diar_model,
                        rttms_mask_mats=rttms_mask_mats,
                        batch_start=sample_idx + 1 - len(batch_samples),
                        batch_size=len(batch_samples),
                        device=asr_model.device,
                    )
                if cfg.parallel_speaker_strategy:
                    multispk_asr_streamer = launch_parallel_streaming(
                        cfg=cfg,
                        asr_model=asr_model,
                        diar_model=diar_model,
                        streaming_buffer=streaming_buffer,
                        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
                    )
                    batch_seglst_list = multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(
                        samples=batch_samples
                    )
                else:
                    multispk_asr_streamer = launch_serial_streaming(
                        cfg=cfg,
                        asr_model=asr_model,
                        diar_model=diar_model,
                        streaming_buffer=streaming_buffer,
                    )
                    batch_seglst_list = multispk_asr_streamer.generate_seglst_dicts_from_serial_streaming(
                        samples=batch_samples
                    )
                should_collect_diar = cfg.diar_output_rttm_dir is not None or any(
                    sample.get("rttm_filepath") for sample in batch_samples
                )
                if should_collect_diar:
                    feature_frame_length_sec = asr_model.cfg.preprocessor.window_stride
                    batch_predictions, batch_metadata = collect_diar_predictions(
                        diar_preds=multispk_asr_streamer.instance_manager.diar_states.diar_pred_out_stream,
                        samples=batch_samples,
                        feature_lengths=streaming_buffer.streams_length,
                        feature_frame_length_sec=feature_frame_length_sec,
                        diar_frame_length_sec=feature_frame_length_sec * diar_output_subsampling_factor,
                    )
                    diar_predictions.extend(batch_predictions)
                    diar_samples.extend(batch_metadata)
                seglst_dict_list.extend(batch_seglst_list)
                streaming_buffer.reset_buffer()
                batch_samples = []

    if diar_predictions:
        write_and_score_diar_predictions(
            predictions=diar_predictions,
            samples=diar_samples,
            output_subsampling_factor=diar_output_subsampling_factor,
            diar_output_rttm_dir=cfg.diar_output_rttm_dir,
            diar_collar=cfg.diar_collar,
            diar_ignore_overlap=cfg.diar_ignore_overlap,
        )

    if len(seglst_dict_list) == 0:
        logging.warning("No segmentation list dictionary found.")
        return

    if cfg.output_path is not None and multispk_asr_streamer is not None:
        if cfg.parallel_speaker_strategy:
            write_seglst_file(seglst_dict_list=seglst_dict_list, output_path=cfg.output_path)
        else:
            write_seglst_file(seglst_dict_list=seglst_dict_list, output_path=cfg.output_path)


if __name__ == '__main__':
    main()
