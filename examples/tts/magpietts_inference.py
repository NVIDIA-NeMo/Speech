# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
TTS Inference and Evaluation Script.

Supports both encoder-decoder MagpieTTS and decoder-only EasyMagpieTTS models
with:
- Automatic MoE detection and FLOPs calculation
- Comprehensive evaluation metrics (RTF, FLOPs, CER, SSIM, etc.)

This script provides a clean CLI for running TTS inference with optional
evaluation. Model-specific behaviour (dataset creation, inference loop, CLI
arguments) is handled by separate runner classes so there is no scattered
if/else branching.

Example usage:
    # MagpieTTS inference (encoder-decoder, default)
    python examples/tts/magpietts_inference.py \\
        --model_type magpie \\
        --nemo_files /path/to/model.nemo \\
        --datasets_json_path /path/to/evalset_config.json \\
        --out_dir /path/to/output \\
        --codecmodel_path /path/to/codec.nemo

    # EasyMagpieTTS inference (decoder-only)
    python examples/tts/magpietts_inference.py \\
        --model_type easy_magpie \\
        --nemo_files /path/to/model.nemo \\
        --datasets_json_path /path/to/evalset_config.json \\
        --out_dir /path/to/output \\
        --codecmodel_path /path/to/codec.nemo

    # With evaluation
    python examples/tts/magpietts_inference.py \\
        --model_type magpie \\
        --hparams_files /path/to/hparams.yaml \\
        --checkpoint_files /path/to/model.ckpt \\
        --datasets_json_path /path/to/evalset_config.json \\
        --out_dir /path/to/output \\
        --codecmodel_path /path/to/codec.nemo \\
        --run_evaluation \\
        --num_repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import load_evalset_config
from nemo.collections.tts.modules.magpietts_inference.evaluation import (
    DEFAULT_VIOLIN_METRICS,
    EvaluationConfig,
    compute_mean_with_confidence_interval,
    evaluate_generated_audio_dir,
)
from nemo.collections.tts.modules.magpietts_inference.inference import BaseInferenceConfig, BaseInferenceRunner
from nemo.collections.tts.modules.magpietts_inference.utils import (
    ModelLoadConfig,
    _add_common_args,
    _add_easy_magpie_args,
    _add_magpie_args,
    _build_easy_magpie_config,
    _build_magpie_config,
    _configure_cuda_for_rank,
    _get_torchrun_rank_info,
    _merge_multiturn_rank_outputs,
    _runner_eval_manifest_and_audio_dir,
    _save_grouped_multiturn_filewise_metrics,
    _select_runner_cls,
    _wait_for_multiturn_rank_manifests,
    append_metrics_to_csv,
    create_formatted_metrics_mean_ci,
    filter_datasets,
    get_experiment_name_from_checkpoint_path,
    load_easy_magpie_model,
    load_magpie_model,
    log_model_architecture_summary,
    seed_all,
    write_csv_header_if_needed,
)
from nemo.collections.tts.modules.magpietts_inference.visualization import create_combined_box_plot, create_violin_plot
from nemo.utils import logging


def run_inference_and_evaluation(
    runner: BaseInferenceRunner,
    checkpoint_name: str,
    inference_config: BaseInferenceConfig,
    eval_config: EvaluationConfig,
    dataset_meta_info: dict,
    datasets: List[str],
    out_dir: str,
    flops_per_component: dict,
    moe_info: str,
    num_repeats: int = 1,
    confidence_level: float = 0.95,
    violin_plot_metrics: Optional[List[str]] = None,
    clean_up_disk: bool = False,
    skip_evaluation: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Run inference and optional evaluation on specified datasets.

    This function is model-type agnostic -- it delegates dataset creation
    and batch inference to the provided ``runner``.

    Args:
        runner: Concrete inference runner (MagpieInferenceRunner or EasyMagpieInferenceRunner).
        checkpoint_name: Human-readable checkpoint identifier for output naming.
        inference_config: Configuration for inference.
        eval_config: Configuration for evaluation.
        dataset_meta_info: Dictionary containing dataset metadata.
        datasets: List of dataset names to process.
        out_dir: Output directory for results.
        flops_per_component: FLOPs info dict from log_model_architecture_summary.
        moe_info: MoE identifier string from log_model_architecture_summary.
        num_repeats: Number of times to repeat inference (for CI estimation).
        confidence_level: Confidence level for CI calculation.
        violin_plot_metrics: Metrics to include in violin plots.
        clean_up_disk: Whether to clean up output directory after completion.
        skip_evaluation: Whether to skip evaluation (inference only mode).

    Returns:
        Tuple of (mean CER across datasets, mean SSIM across datasets).
    """
    if violin_plot_metrics is None:
        violin_plot_metrics = list(DEFAULT_VIOLIN_METRICS)

    # Remove UTMOSv2 from plots if disabled
    if not eval_config.with_utmosv2 and 'utmosv2' in violin_plot_metrics:
        violin_plot_metrics.remove('utmosv2')

    rank, world_size, _ = _get_torchrun_rank_info()
    is_distributed = world_size > 1
    is_multiturn_user_audio = getattr(runner, "produces_turn_level_evaluation", False)

    if hasattr(runner, "set_distributed_context"):
        runner.set_distributed_context(rank=rank, world_size=world_size)

    # Build full checkpoint identifier (include MoE info if present)
    full_checkpoint_name = (
        f"{checkpoint_name}_{moe_info}{inference_config.build_identifier()}_SV_{eval_config.sv_model}"
    )

    # Tracking metrics across datasets
    ssim_per_dataset = []
    cer_per_dataset = []
    all_datasets_filewise_metrics = {}

    # CSV headers
    csv_header = (
        "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,"
        "wer_cumulative,cer_pred_gt_audio_filewise_avg,cer_pred_gt_audio_cumulative,"
        "wer_pred_gt_audio_filewise_avg,wer_pred_gt_audio_cumulative,"
        "ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,"
        "ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,"
        "ssim_gt_context_avg_alternate,esim_pred_gt_avg,ems_pred_gt_avg,"
        "pitch_distance_avg,intensity_distance_avg,speech_rate_distance_avg,"
        "cer_gt_audio_cumulative,wer_gt_audio_cumulative,"
        "utmosv2_avg,total_gen_audio_seconds,frechet_codec_distance,"
        "eou_cutoff_rate,eou_silence_rate,eou_noise_rate,eou_error_rate,"
        "katakana_cer_filewise_avg,katakana_cer_cumulative"
    )

    for dataset in datasets:
        logging.info(f"Processing dataset: {dataset}")

        meta = dataset_meta_info[dataset]
        manifest_records = read_manifest(meta['manifest_path'])

        if 'asr_model' in meta:
            asr_model_name = meta['asr_model']['name']
            asr_model_type = meta['asr_model']['type']
        else:
            asr_model_name = eval_config.asr_model_name
            asr_model_type = eval_config.asr_model_type

        if 'language' in meta:
            language = meta.get('language')
        else:
            language = eval_config.language

        tokenizer_names = meta.get('tokenizer_names', None)

        dataset_meta_for_dl = {
            "manifest_path": meta["manifest_path"],
            "audio_dir": meta["audio_dir"],
            "language": language,
            "tokenizer_names": tokenizer_names,
        }

        # Setup output directories
        eval_dir = os.path.join(out_dir, f"{full_checkpoint_name}_{language}_{dataset}")
        audio_dir = os.path.join(eval_dir, "audio")
        os.makedirs(eval_dir, exist_ok=True)

        # Setup CSV files
        per_run_csv = os.path.join(eval_dir, "all_experiment_metrics.csv")
        if rank == 0:
            write_csv_header_if_needed(per_run_csv, csv_header)

        metrics_all_repeats = []
        filewise_metrics_all_repeats = []

        for repeat_idx in range(num_repeats):
            repeat_log_msg = f"Repeat {repeat_idx + 1}/{num_repeats} for dataset {dataset}"
            if is_distributed:
                repeat_log_msg += f", rank {rank}/{world_size}"
            logging.info(repeat_log_msg)

            repeat_audio_dir = os.path.join(audio_dir, f"repeat_{repeat_idx}")
            os.makedirs(repeat_audio_dir, exist_ok=True)

            # Create dataset and run inference
            test_dataset = runner.create_dataset({dataset: dataset_meta_for_dl})

            if len(test_dataset) != len(manifest_records):
                raise ValueError(
                    f"Dataset length mismatch: {len(test_dataset)} vs {len(manifest_records)} manifest records"
                )

            if is_distributed and not is_multiturn_user_audio:
                raise RuntimeError(
                    "torchrun multi-GPU sharding is currently implemented for "
                    "--easy_magpie_inference_mode multiturn_user_audio only. "
                    "Use the existing single-process path for single_turn/magpie, or add a "
                    "rank-safe merge path for those runners."
                )

            inference_output_dir = repeat_audio_dir
            if is_distributed and is_multiturn_user_audio:
                inference_output_dir = os.path.join(repeat_audio_dir, f"rank_{rank:04d}")

            rtf_metrics_list, _, codec_file_paths = runner.run_inference_on_dataset(
                dataset=test_dataset,
                output_dir=inference_output_dir,
                manifest_records=manifest_records,
                audio_base_dir=meta['audio_dir'],
                save_cross_attention_maps=True,
                save_context_audio=(repeat_idx == 0),  # Only save context audio once
                save_predicted_codes=eval_config.with_fcd,  # Code files are only needed for FCD computation
            )

            # Compute mean RTF metrics
            mean_rtf = runner.compute_mean_rtf_metrics(rtf_metrics_list)

            # Add FLOPs metrics per component
            for component_name, component_flops in flops_per_component.items():
                for key, value in component_flops.items():
                    mean_rtf[f"{component_name}_{key}"] = value
                logging.info(f"{component_name} FLOPs per token: {component_flops['total_flops_per_token']:,}")

            rtf_metrics_filename = f"{dataset}_rtf_metrics_{repeat_idx}.json"
            if is_distributed:
                rtf_metrics_filename = f"{dataset}_rtf_metrics_{repeat_idx}_rank{rank:04d}.json"
            with open(os.path.join(eval_dir, rtf_metrics_filename), "w") as f:
                json.dump(mean_rtf, f, indent=4)

            if skip_evaluation:
                logging.info("Skipping evaluation as requested.")
                continue

            # Run evaluation
            if is_distributed and is_multiturn_user_audio:
                if rank != 0:
                    # Non-zero ranks only generate. Rank 0 waits and evaluates merged outputs.
                    continue

                _wait_for_multiturn_rank_manifests(repeat_audio_dir, world_size)
                merged_manifest_path = _merge_multiturn_rank_outputs(
                    repeat_audio_dir=repeat_audio_dir,
                    world_size=world_size,
                    save_predicted_codes=eval_config.with_fcd,
                )
                eval_manifest_path = merged_manifest_path
                eval_audio_dir = repeat_audio_dir
            else:
                eval_manifest_path, eval_audio_dir = _runner_eval_manifest_and_audio_dir(
                    runner,
                    default_manifest=meta['manifest_path'],
                    default_audio_dir=meta['audio_dir'],
                )

            eval_config_for_dataset = EvaluationConfig(
                sv_model=eval_config.sv_model,
                asr_model_name=asr_model_name,
                asr_model_type=asr_model_type,
                eou_model_name=eval_config.eou_model_name,
                language=language,
                with_utmosv2=eval_config.with_utmosv2,
                with_fcd=eval_config.with_fcd,
                codec_model_path=eval_config.codec_model_path,
                with_prosody_metrics=eval_config.with_prosody_metrics,
                prosody_model_size=eval_config.prosody_model_size,
                strip_text_annotations_for_metrics=eval_config.strip_text_annotations_for_metrics,
                device=eval_config.device,
                asr_batch_size=eval_config.asr_batch_size,
                eou_batch_size=eval_config.eou_batch_size,
            )

            metrics, filewise_metrics = evaluate_generated_audio_dir(
                manifest_path=eval_manifest_path,
                audio_dir=eval_audio_dir,
                generated_audio_dir=repeat_audio_dir,
                config=eval_config_for_dataset,
            )

            metrics_all_repeats.append(metrics)
            filewise_metrics_all_repeats.extend(filewise_metrics)

            # Save metrics
            metrics_path = os.path.join(eval_dir, f"{dataset}_metrics_{repeat_idx}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            sorted_filewise = sorted(filewise_metrics, key=lambda x: x.get('cer', 0), reverse=True)
            filewise_metrics_path = os.path.join(eval_dir, f"{dataset}_filewise_metrics_{repeat_idx}.json")
            with open(filewise_metrics_path, "w", encoding="utf-8") as f:
                json.dump(sorted_filewise, f, indent=4, ensure_ascii=False)

            if is_multiturn_user_audio:
                _save_grouped_multiturn_filewise_metrics(
                    eval_dir=eval_dir,
                    dataset=dataset,
                    repeat_idx=repeat_idx,
                    filewise_metrics=filewise_metrics,
                    manifest_path=eval_manifest_path,
                )

            # Append to per-run CSV
            append_metrics_to_csv(per_run_csv, full_checkpoint_name, dataset, metrics)

            # Create violin plot for this repeat
            violin_path = Path(eval_dir) / f"{dataset}_violin_{repeat_idx}.png"
            create_violin_plot(filewise_metrics, violin_plot_metrics, violin_path)

            # Delete temporary predicted codes files
            if is_distributed and is_multiturn_user_audio:
                for codec_file_path in Path(repeat_audio_dir).glob("predicted_codes_*.pt"):
                    if os.path.exists(codec_file_path):
                        os.remove(codec_file_path)
            else:
                for codec_file_path in codec_file_paths:
                    os.remove(codec_file_path)

        if rank != 0:
            continue

        if skip_evaluation or not metrics_all_repeats:
            continue

        # Store for combined plot
        all_datasets_filewise_metrics[dataset] = filewise_metrics_all_repeats

        # Compute mean with confidence interval across repeats
        metrics_mean_ci = compute_mean_with_confidence_interval(
            metrics_all_repeats,
            confidence=confidence_level,
        )

        formatted_metrics_mean_ci = create_formatted_metrics_mean_ci(metrics_mean_ci)

        # Write to aggregated CSV
        ci_csv = os.path.join(out_dir, "all_experiment_metrics_with_ci.csv")
        write_csv_header_if_needed(ci_csv, csv_header)
        append_metrics_to_csv(ci_csv, full_checkpoint_name, dataset, formatted_metrics_mean_ci)

        # Track per-dataset means
        ssim_values = [m['ssim_pred_context_avg'] for m in metrics_all_repeats]
        cer_values = [m['cer_cumulative'] for m in metrics_all_repeats]
        ssim_per_dataset.append(np.mean(ssim_values))
        cer_per_dataset.append(np.mean(cer_values))

    # Create combined plot if we have multiple datasets
    if rank == 0 and len(all_datasets_filewise_metrics) > 1:
        combined_plot_path = os.path.join(out_dir, f"{full_checkpoint_name}_combined_violin_plot.png")
        create_combined_box_plot(all_datasets_filewise_metrics, violin_plot_metrics, combined_plot_path)

    # Clean up if requested
    if rank == 0 and clean_up_disk:
        logging.info(f"Cleaning up output directory: {out_dir}")
        shutil.rmtree(out_dir)

    # Return averaged metrics
    if rank == 0 and ssim_per_dataset and cer_per_dataset:
        return np.mean(cer_per_dataset), np.mean(ssim_per_dataset)
    return None, None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser with all argument groups."""
    parser = argparse.ArgumentParser(
        description='TTS Inference and Evaluation (MagpieTTS & EasyMagpieTTS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _add_common_args(parser)
    _add_magpie_args(parser)
    _add_easy_magpie_args(parser)
    return parser


def main(argv=None):
    """Entry point for TTS inference and evaluation."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    if args.model_type == 'easy_magpie' and args.easy_magpie_inference_mode == 'multiturn_user_audio':
        _configure_cuda_for_rank()
        if args.batch_size > 1:
            parser.error("--easy_magpie_inference_mode multiturn_user_audio requires --batch_size 1.")

    if args.deterministic:
        seed_all(seed=9)

    dataset_meta_info = load_evalset_config(
        config_path=args.datasets_json_path, dataset_base_path=args.datasets_base_path
    )
    datasets = filter_datasets(dataset_meta_info, args.datasets)
    logging.info(f"Loaded {len(datasets)} datasets: {', '.join(datasets)}")

    # Validate model loading args
    has_checkpoint_mode = (
        args.hparams_files is not None
        and args.checkpoint_files is not None
        and args.hparams_files != "null"
        and args.checkpoint_files != "null"
    )
    has_nemo_mode = args.nemo_files is not None and args.nemo_files != "null"

    if not has_checkpoint_mode and not has_nemo_mode:
        parser.error("You must provide either:\n 1. --hparams_files and --checkpoint_files\n 2. --nemo_files")

    # Select model loader and config builder based on --model_type
    is_easy_magpie = args.model_type == 'easy_magpie'
    load_fn = load_easy_magpie_model if is_easy_magpie else load_magpie_model
    inference_config = _build_easy_magpie_config(args) if is_easy_magpie else _build_magpie_config(args)
    runner_cls = _select_runner_cls(args)

    eval_config = EvaluationConfig(
        sv_model=args.sv_model,
        asr_model_name=args.asr_model_name,
        asr_model_type=args.asr_model_type,
        eou_model_name=args.eou_model_name,
        language=args.language,
        with_utmosv2=not args.disable_utmosv2,
        with_fcd=not args.disable_fcd,
        codec_model_path=args.codecmodel_path if not args.disable_fcd else None,
        with_prosody_metrics=args.with_prosody_metrics,
        prosody_model_size=args.prosody_model_size,
        strip_text_annotations_for_metrics=args.strip_text_annotations_for_metrics,
        asr_batch_size=args.asr_batch_size,
        eou_batch_size=args.eou_batch_size,
    )

    cer, ssim = None, None

    # Iterate over model files (checkpoint or nemo)
    if has_checkpoint_mode:
        hparam_files = args.hparams_files.split(",")
        checkpoint_files = args.checkpoint_files.split(",")

        if len(hparam_files) != len(checkpoint_files):
            parser.error("Number of hparams_files must match number of checkpoint_files")

        for hparams_file, checkpoint_file in zip(hparam_files, checkpoint_files):
            logging.info(f"Processing checkpoint: {checkpoint_file}")

            model_config = ModelLoadConfig(
                hparams_file=hparams_file,
                checkpoint_file=checkpoint_file,
                codecmodel_path=args.codecmodel_path,
                legacy_codebooks=args.legacy_codebooks,
                legacy_text_conditioning=args.legacy_text_conditioning,
                hparams_from_wandb=args.hparams_file_from_wandb,
                phoneme_tokenizer_path=getattr(args, 'phoneme_tokenizer_path', None),
                disable_cas_for_context_text=args.disable_cas_for_context_text,
            )

            # Load model
            model, checkpoint_name = load_fn(model_config)
            # Log architecture summary and get MoE info + FLOPs metrics
            moe_info, flops_per_component = log_model_architecture_summary(model)

            # Add experiment name prefix if requested
            if args.log_exp_name and model_config.checkpoint_file:
                exp_name = get_experiment_name_from_checkpoint_path(model_config.checkpoint_file)
                checkpoint_name = f"{exp_name}__{checkpoint_name}"

            # Create inference runner
            runner = runner_cls(model, inference_config)

            cer, ssim = run_inference_and_evaluation(
                runner=runner,
                checkpoint_name=checkpoint_name,
                inference_config=inference_config,
                eval_config=eval_config,
                dataset_meta_info=dataset_meta_info,
                datasets=datasets,
                out_dir=args.out_dir,
                flops_per_component=flops_per_component,
                moe_info=moe_info,
                num_repeats=args.num_repeats,
                confidence_level=args.confidence_level,
                violin_plot_metrics=args.violin_plot_metrics,
                clean_up_disk=args.clean_up_disk,
                skip_evaluation=not args.run_evaluation,
            )

    else:  # nemo mode
        for nemo_file in args.nemo_files.split(","):
            logging.info(f"Processing NeMo file: {nemo_file}")

            model_config = ModelLoadConfig(
                nemo_file=nemo_file,
                codecmodel_path=args.codecmodel_path,
                legacy_codebooks=args.legacy_codebooks,
                legacy_text_conditioning=args.legacy_text_conditioning,
                phoneme_tokenizer_path=getattr(args, 'phoneme_tokenizer_path', None),
                disable_cas_for_context_text=args.disable_cas_for_context_text,
            )

            # Load model
            model, checkpoint_name = load_fn(model_config)
            # Log architecture summary and get MoE info + FLOPs metrics
            moe_info, flops_per_component = log_model_architecture_summary(model)

            # Create inference runner
            runner = runner_cls(model, inference_config)

            cer, ssim = run_inference_and_evaluation(
                runner=runner,
                checkpoint_name=checkpoint_name,
                inference_config=inference_config,
                eval_config=eval_config,
                dataset_meta_info=dataset_meta_info,
                datasets=datasets,
                out_dir=args.out_dir,
                flops_per_component=flops_per_component,
                moe_info=moe_info,
                num_repeats=args.num_repeats,
                confidence_level=args.confidence_level,
                violin_plot_metrics=args.violin_plot_metrics,
                clean_up_disk=args.clean_up_disk,
                skip_evaluation=not args.run_evaluation,
            )

    # Check quality targets
    if cer is not None and args.cer_target is not None:
        if cer > args.cer_target:
            raise ValueError(f"CER {cer:.4f} exceeds target {args.cer_target:.4f}")
        logging.info(f"CER {cer:.4f} meets target {args.cer_target:.4f}")

    if ssim is not None and args.ssim_target is not None:
        if ssim < args.ssim_target:
            raise ValueError(f"SSIM {ssim:.4f} below target {args.ssim_target:.4f}")
        logging.info(f"SSIM {ssim:.4f} meets target {args.ssim_target:.4f}")

    logging.info("Inference and evaluation completed successfully.")


if __name__ == '__main__':
    main()
