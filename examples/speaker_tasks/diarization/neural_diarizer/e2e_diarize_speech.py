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


"""
This script provides an inference and evaluation script for end-to-end speaker diarization models.
The performance of the diarization model is measured using the Diarization Error Rate (DER).
If you want to evaluate its performance, the manifest JSON file should contain the corresponding RTTM
(Rich Transcription Time Marked) file.
Please refer to the NeMo Library Documentation for more details on data preparation for diarization inference:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit
/asr/speaker_diarization/datasets.html#data-preparation-for-inference

Usage for diarization inference:

The end-to-end speaker diarization model can be specified by "model_path".
Data for diarization is fed through the "dataset_manifest".
By default, post-processing is bypassed, and only binarization is performed.
If you want to reproduce DER scores reported on NeMo model cards, you need to apply post-processing steps.
Use batch_size = 1 to have the longest inference window and the highest possible accuracy.

python $BASEPATH/neural_diarizer/e2e_diarize_speech.py \
    model_path=/path/to/diar_sortformer_4spk_v1.nemo \
    batch_size=1 \
    dataset_manifest=/path/to/diarization_manifest.json

"""
import json
import logging
import os
import tempfile
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.diarization_utils import convert_pred_mat_to_segments
from nemo.collections.asr.parts.utils.sortformer_utils import (
    InferenceProfiler,
    configure_output_subsampling_factor,
    get_prediction_cache_metadata,
    load_prediction_tensors,
    save_prediction_tensors,
)
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.collections.asr.parts.utils.transcribe_utils import read_and_maybe_sort_manifest
from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams, load_postprocessing_from_yaml
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.config import hydra_runner
from nemo.utils.dependency import import_optional_dependency

seed_everything(42)
torch.backends.cudnn.deterministic = True


@dataclass
class DiarizationConfig:
    """Diarization configuration parameters for inference."""

    model_path: Optional[str] = None  # Path to a .nemo file
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    presort_manifest: Optional[bool] = True

    postprocessing_yaml: Optional[str] = None  # Path to a yaml file for postprocessing configurations
    no_der: bool = False
    out_rttm_dir: Optional[str] = None
    out_preds_tensors: Optional[str] = None  # Explicit cache path; enables prediction loading and saving
    overwrite_preds_tensors: bool = False  # Ignore and replace an existing prediction cache
    precision: str = "bf16"  # 32, bf16, bf16-mixed

    # General configs
    session_len_sec: float = -1  # End-to-end diarization session length in seconds
    batch_size: int = 1
    num_workers: int = 0
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    bypass_postprocessing: bool = True  # If True, postprocessing will be bypassed
    log: bool = False  # If True, log will be printed
    output_subsampling_factor: Optional[int] = None  # Override prediction step in 10 ms feature frames
    profile_inference: bool = True  # Report wall time and detailed streaming-step timings

    use_lhotse: bool = True
    batch_duration: int = 100000
    # Total padded-audio duration threshold, in seconds, for feature extraction.
    # Above it, batches are split along the batch dimension; <= 0 disables splitting.
    # Lower this value for feature-extraction OOMs.
    max_batch_dur: float = 100000

    # Eval Settings: (0.25, False) should be default setting for sortformer eval.
    collar: float = 0.25  # Collar in seconds for DER calculation
    ignore_overlap: bool = False  # If True, DER will be calculated only for non-overlapping segments

    # Streaming diarization configs
    async_streaming: bool = False
    # Use fixed-size encoder inputs, trading extra padded computation for stable shapes.
    async_pad_to_max: bool = False
    # Compile the frontend encoder and optional transformer encoder; dynamic shapes support variable input lengths.
    compile_encoder: bool = False
    # Emulate production streams arriving independently; offline batches otherwise update in lockstep.
    async_desync_updates: bool = False
    spkcache_len: Optional[int] = None
    spkcache_update_period: int = 144
    fifo_len: int = 188
    chunk_len: int = 6
    chunk_left_context: Optional[int] = None
    chunk_right_context: int = 7

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # Optuna Config
    launch_pp_optim: bool = False  # If True, launch optimization process for postprocessing parameters
    optuna_study_name: str = "optim_postprocessing"
    optuna_temp_dir: str = "/tmp/optuna"
    optuna_storage: str = f"sqlite:///{optuna_study_name}.db"
    optuna_log_file: str = f"{optuna_study_name}.log"
    optuna_n_trials: int = 100000


def optuna_suggest_params(postprocessing_cfg: PostProcessingParams, trial) -> PostProcessingParams:
    """
    Suggests hyperparameters for postprocessing using Optuna.
    See the following link for `trial` instance in Optuna framework.
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial

    Args:
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        trial (optuna.Trial): The Optuna trial object used to suggest hyperparameters.

    Returns:
        PostProcessingParams: The updated postprocessing configuration with suggested hyperparameters.
    """
    postprocessing_cfg.onset = trial.suggest_float("onset", 0.4, 0.8, step=0.01)
    postprocessing_cfg.offset = trial.suggest_float("offset", 0.4, 0.9, step=0.01)
    postprocessing_cfg.pad_onset = trial.suggest_float("pad_onset", 0.1, 0.5, step=0.01)
    postprocessing_cfg.pad_offset = trial.suggest_float("pad_offset", 0.0, 0.2, step=0.01)
    postprocessing_cfg.min_duration_on = trial.suggest_float("min_duration_on", 0.0, 0.75, step=0.01)
    postprocessing_cfg.min_duration_off = trial.suggest_float("min_duration_off", 0.0, 0.75, step=0.01)
    return postprocessing_cfg


def get_tensor_path(cfg: DiarizationConfig) -> tuple[Optional[str], str, str]:
    """
    Resolve the explicit prediction-cache path and derive model/manifest identifiers.

    Args:
        cfg (DiarizationConfig): The configuration object containing model and dataset details.

    Returns:
        tensor_path (Optional[str]): Absolute prediction-cache path, or ``None`` when caching is disabled.
        model_id (str): Model identifier including the configured output subsampling factor.
        tensor_filename (str): Manifest-derived identifier used in the prediction tensor filename.
    """
    tensor_filename = os.path.basename(cfg.dataset_manifest).replace("manifest.", "").replace(".json", "")
    model_path = Path(cfg.model_path).expanduser().absolute()
    model_id = model_path.name.replace(".ckpt", "").replace(".nemo", "")
    model_id = f"{model_id}_sf{cfg.output_subsampling_factor}"
    if cfg.out_preds_tensors:
        tensor_path = Path(cfg.out_preds_tensors).expanduser().absolute()
    else:
        tensor_path = None
    return None if tensor_path is None else str(tensor_path), model_id, tensor_filename


def diarization_objective(
    trial,
    postprocessing_cfg: PostProcessingParams,
    temp_out_dir: str,
    infer_audio_rttm_dict: Dict[str, Dict[str, str]],
    diar_model_preds_total_list: List[torch.Tensor],
    unit_10ms_frame_count: int,
    collar: float = 0.25,
    ignore_overlap: bool = False,
) -> float:
    """
    Objective function for Optuna hyperparameter optimization in speaker diarization.

    This function evaluates the diarization performance using a set of postprocessing parameters
    suggested by Optuna. It converts prediction matrices to time-stamp segments, scores the
    diarization results, and returns the Diarization Error Rate (DER) as the optimization metric.

    Args:
        trial (optuna.Trial): The Optuna trial object used to suggest hyperparameters.
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        temp_out_dir (str): Temporary directory for storing intermediate outputs.
        infer_audio_rttm_dict (Dict[str, Dict[str, str]]): Dictionary containing audio file paths,
            offsets, durations, and RTTM file paths.
        diar_model_preds_total_list (List[torch.Tensor]): List of prediction matrices containing
            sigmoid values for each speaker.
            Dimension: [(1, num_frames, num_speakers), ..., (1, num_frames, num_speakers)]
        unit_10ms_frame_count (int): Number of 10 ms feature frames represented by each prediction frame.
        collar (float, optional): Collar in seconds for DER calculation. Defaults to 0.25.
        ignore_overlap (bool, optional): If True, DER will be calculated only for non-overlapping segments.
            Defaults to False.

    Returns:
        der (float): Diarization Error Rate for the given set of postprocessing parameters.
    """
    with tempfile.TemporaryDirectory(dir=temp_out_dir, prefix="Diar_PostProcessing_") as _:
        if trial is not None:
            postprocessing_cfg = optuna_suggest_params(postprocessing_cfg, trial)
        all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(
            audio_rttm_map_dict=infer_audio_rttm_dict,
            postprocessing_cfg=postprocessing_cfg,
            batch_preds_list=diar_model_preds_total_list,
            unit_10ms_frame_count=unit_10ms_frame_count,
            bypass_postprocessing=False,
        )
        metric, _, _ = score_labels(
            AUDIO_RTTM_MAP=infer_audio_rttm_dict,
            all_reference=all_refs,
            all_hypothesis=all_hyps,
            all_uem=all_uems,
            collar=collar,
            ignore_overlap=ignore_overlap,
        )
        der = abs(metric)
    return der


def run_optuna_hyperparam_search(
    cfg: DiarizationConfig,  # type: DiarizationConfig
    postprocessing_cfg: PostProcessingParams,
    infer_audio_rttm_dict: Dict[str, Dict[str, str]],
    preds_list: List[torch.Tensor],
    temp_out_dir: str,
    unit_10ms_frame_count: int,
):
    """
    Run Optuna hyperparameter optimization for speaker diarization.

    Args:
        cfg (DiarizationConfig): The configuration object containing model and dataset details.
        postprocessing_cfg (PostProcessingParams): The current postprocessing configuration.
        infer_audio_rttm_dict (dict): dictionary of audio file path, offset, duration and RTTM filepath.
        preds_list (List[torch.Tensor]): list of prediction matrices containing sigmoid values for each speaker.
            Dimension: [(1, num_frames, num_speakers), ..., (1, num_frames, num_speakers)]
        temp_out_dir (str): temporary directory for storing intermediate outputs.
        unit_10ms_frame_count (int): Number of 10 ms feature frames represented by each prediction frame.
    """
    optuna = import_optional_dependency("optuna")

    worker_function = lambda trial: diarization_objective(
        trial=trial,
        postprocessing_cfg=postprocessing_cfg,
        temp_out_dir=temp_out_dir,
        infer_audio_rttm_dict=infer_audio_rttm_dict,
        diar_model_preds_total_list=preds_list,
        unit_10ms_frame_count=unit_10ms_frame_count,
        collar=cfg.collar,
    )
    study = optuna.create_study(
        direction="minimize", study_name=cfg.optuna_study_name, storage=cfg.optuna_storage, load_if_exists=True
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    if cfg.optuna_log_file is not None:
        logger.addHandler(logging.FileHandler(cfg.optuna_log_file, mode="a"))
    logger.addHandler(logging.StreamHandler())
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    study.optimize(worker_function, n_trials=cfg.optuna_n_trials)


@hydra_runner(config_name="DiarizationConfig", schema=DiarizationConfig)
def main(cfg: DiarizationConfig) -> Union[DiarizationConfig]:
    """Main function for end-to-end speaker diarization inference."""
    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None:
        raise ValueError("cfg.model_path cannot be None. Please specify the path to the model.")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    if cfg.model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(
            checkpoint_path=cfg.model_path, map_location=map_location, strict=False
        )
    elif cfg.model_path.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.model_path, map_location=map_location)
    else:
        raise ValueError("cfg.model_path must end with.ckpt or.nemo!")

    diar_model.max_batch_dur = cfg.max_batch_dur

    cfg.output_subsampling_factor = configure_output_subsampling_factor(diar_model, cfg.output_subsampling_factor)
    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    trainer = pl.Trainer(devices=device, accelerator=accelerator, precision=cfg.precision)
    diar_model.set_trainer(trainer)

    if torch.cuda.is_bf16_supported() and cfg.precision.startswith("bf16"):
        diar_model = diar_model.to(dtype=torch.bfloat16).eval()
    else:
        diar_model = diar_model.eval()

    if cfg.presort_manifest:
        audio_key = cfg.get('audio_key', 'audio_filepath')
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            for item in read_and_maybe_sort_manifest(cfg.dataset_manifest, try_sort=cfg.presort_manifest):
                audio_file = get_full_path(audio_file=item[audio_key], manifest_file=cfg.dataset_manifest)
                item[audio_key] = audio_file
                f.write(json.dumps(item) + "\n")
            sorted_manifest_path = f.name
        diar_model._cfg.test_ds.manifest_filepath = sorted_manifest_path
        infer_audio_rttm_dict = audio_rttm_map(sorted_manifest_path)
    else:
        diar_model._cfg.test_ds.manifest_filepath = cfg.dataset_manifest
        infer_audio_rttm_dict = audio_rttm_map(cfg.dataset_manifest)
    remove_path_after_done = sorted_manifest_path if sorted_manifest_path is not None else None

    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    diar_model._cfg.test_ds.pin_memory = False
    diar_model._cfg.test_ds.num_spks = -1

    OmegaConf.set_struct(diar_model._cfg, False)
    diar_model._cfg.test_ds.use_lhotse = cfg.use_lhotse
    diar_model._cfg.test_ds.use_bucketing = False
    diar_model._cfg.test_ds.drop_last = False
    diar_model._cfg.test_ds.batch_duration = cfg.batch_duration
    OmegaConf.set_struct(diar_model._cfg, True)

    # Model setup for inference
    diar_model._cfg.test_ds.num_workers = cfg.num_workers
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)

    # Streaming mode setup (only if enabled)
    if diar_model.streaming_mode:
        diar_model.async_streaming = cfg.async_streaming
        diar_model.async_pad_to_max = cfg.async_pad_to_max
        diar_model.sortformer_modules.async_desync_updates = cfg.async_desync_updates
        diar_model.sortformer_modules.chunk_len = cfg.chunk_len
        if cfg.spkcache_len is not None:
            diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
        if cfg.chunk_left_context is not None:
            diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
        diar_model.sortformer_modules.chunk_right_context = cfg.chunk_right_context
        diar_model.sortformer_modules.fifo_len = cfg.fifo_len
        diar_model.sortformer_modules.log = cfg.log
        diar_model.sortformer_modules.spkcache_update_period = cfg.spkcache_update_period
        diar_model._check_streaming_parameters()

    if cfg.compile_encoder:
        logging.info("Compiling the frontend encoder")
        diar_model.encoder = torch.compile(diar_model.encoder, dynamic=True)
        if diar_model.transformer_encoder is not None and len(diar_model.transformer_encoder.layers) > 0:
            logging.info("Compiling the optional transformer encoder")
            diar_model.transformer_encoder = torch.compile(diar_model.transformer_encoder, dynamic=True)

    postprocessing_cfg = load_postprocessing_from_yaml(cfg.postprocessing_yaml)
    tensor_path, model_id, tensor_filename = get_tensor_path(cfg)
    cfg.optuna_study_name = f"__{model_id}_{tensor_filename}"
    cfg.optuna_storage: str = f"sqlite:///{cfg.optuna_temp_dir}/{cfg.optuna_study_name}.db"
    cfg.optuna_log_file: str = f"{cfg.optuna_temp_dir}/{cfg.optuna_study_name}.log"
    inference_profiler = InferenceProfiler(diar_model) if cfg.profile_inference else None
    if inference_profiler is not None:
        inference_profiler.install()

    prediction_cache_metadata = (
        get_prediction_cache_metadata(cfg, diar_model, infer_audio_rttm_dict) if tensor_path is not None else None
    )

    if tensor_path is not None and os.path.exists(tensor_path) and not cfg.overwrite_preds_tensors:
        logging.info(
            f"A saved prediction tensor has been found. Loading the saved prediction tensors from {tensor_path}..."
        )
        diar_model_preds_total_list = load_prediction_tensors(tensor_path, prediction_cache_metadata)
    else:
        logging.info("No saved prediction tensors found. Running inference on the dataset...")
        with torch.inference_mode(), torch.autocast(device_type=diar_model.device.type, dtype=diar_model.dtype):
            diar_model.test_batch()

        diar_model_preds_total_list = diar_model.preds_total_list
        if inference_profiler is not None:
            audio_duration = sum(float(item['duration']) for item in infer_audio_rttm_dict.values())
            inference_profiler.log_summary(audio_duration)
        if tensor_path is not None:
            save_prediction_tensors(tensor_path, diar_model.preds_total_list, prediction_cache_metadata)
            logging.info(f"Prediction tensors saved to {tensor_path}")

    if cfg.launch_pp_optim:
        # Launch a hyperparameter optimization process if launch_pp_optim is True
        run_optuna_hyperparam_search(
            cfg=cfg,
            postprocessing_cfg=postprocessing_cfg,
            infer_audio_rttm_dict=infer_audio_rttm_dict,
            preds_list=diar_model_preds_total_list,
            temp_out_dir=cfg.optuna_temp_dir,
            unit_10ms_frame_count=cfg.output_subsampling_factor,
        )

    # Evaluation
    if not cfg.no_der:
        if cfg.out_rttm_dir is not None and not os.path.exists(cfg.out_rttm_dir):
            os.mkdir(cfg.out_rttm_dir)

        logging.info("Running offline diarization evaluation...")
        all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(
            infer_audio_rttm_dict,
            postprocessing_cfg=postprocessing_cfg,
            batch_preds_list=diar_model_preds_total_list,
            unit_10ms_frame_count=cfg.output_subsampling_factor,
            bypass_postprocessing=cfg.bypass_postprocessing,
            out_rttm_dir=cfg.out_rttm_dir,
        )
        logging.info(f"Evaluating the model on the {len(diar_model_preds_total_list)} audio segments...")
        score_labels(
            AUDIO_RTTM_MAP=infer_audio_rttm_dict,
            all_reference=all_refs,
            all_hypothesis=all_hyps,
            all_uem=all_uems,
            collar=cfg.collar,
            ignore_overlap=cfg.ignore_overlap,
        )
        logging.info(f"PostProcessingParams: {postprocessing_cfg}")

    # clean-up
    if cfg.presort_manifest is not None:
        if remove_path_after_done is not None:
            os.unlink(remove_path_after_done)


if __name__ == '__main__':
    main()
