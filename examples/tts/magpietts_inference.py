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
MagpieTTS Inference and Evaluation Script.

Supports both standard and Mixture of Experts (MoE) models with:
- Automatic MoE detection and FLOPs calculation
- Comprehensive evaluation metrics (RTF, FLOPs, CER, SSIM, etc.)

This script provides a clean CLI for running MagpieTTS inference with optional evaluation.
It decouples inference and evaluation into separate modules for better maintainability.

Example usage:
    # Inference only (from .nemo file) - default behavior
    python examples/tts/magpietts_inference.py \\
        --nemo_files /path/to/model.nemo \\
        --datasets_json_path /path/to/evalset_config.json \\
        --out_dir /path/to/output \\
        --codecmodel_path /path/to/codec.nemo

    # Inference with evaluation (from checkpoint)
    python examples/tts/magpietts_inference.py \\
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
import copy
import json
import os
import shutil
from dataclasses import fields
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.models.magpietts import ModelInferenceParameters
from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import load_evalset_config

# Import the modular components
from nemo.collections.tts.modules.magpietts_inference.evaluation import (
    DEFAULT_VIOLIN_METRICS,
    STANDARD_METRIC_KEYS,
    EvaluationConfig,
    compute_mean_with_confidence_interval,
    evaluate_generated_audio_dir,
)
from nemo.collections.tts.modules.magpietts_inference.inference import InferenceConfig, MagpieInferenceRunner
from nemo.collections.tts.modules.magpietts_inference.utils import (
    ModelLoadConfig,
    get_experiment_name_from_checkpoint_path,
    load_magpie_model,
    log_model_architecture_summary,
)
from nemo.collections.tts.modules.magpietts_inference.visualization import create_combined_box_plot, create_violin_plot
from nemo.collections.tts.modules.magpietts_modules import EOSDetectionMethod
from nemo.utils import logging


def parse_layer_list(layer_str: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of layer indices."""
    if layer_str is None:
        return None
    return [int(l.strip()) for l in layer_str.split(",")]


def write_csv_header_if_needed(csv_path: str, header: str) -> None:
    """Write CSV header if file doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(header + "\n")


def append_metrics_to_csv(csv_path: str, checkpoint_name: str, dataset: str, metrics: dict) -> None:
    """Append metrics to a CSV file."""
    values = [
        checkpoint_name,
        dataset,
        metrics.get('cer_filewise_avg', ''),
        metrics.get('wer_filewise_avg', ''),
        metrics.get('cer_cumulative', ''),
        metrics.get('wer_cumulative', ''),
        metrics.get('ssim_pred_gt_avg', ''),
        metrics.get('ssim_pred_context_avg', ''),
        metrics.get('ssim_gt_context_avg', ''),
        metrics.get('ssim_pred_gt_avg_alternate', ''),
        metrics.get('ssim_pred_context_avg_alternate', ''),
        metrics.get('ssim_gt_context_avg_alternate', ''),
        metrics.get('cer_gt_audio_cumulative', ''),
        metrics.get('wer_gt_audio_cumulative', ''),
        metrics.get('utmosv2_avg', ''),
        metrics.get('total_gen_audio_seconds', ''),
        metrics.get('frechet_codec_distance', ''),
    ]
    with open(csv_path, "a") as f:
        f.write(",".join(str(v) for v in values) + "\n")
    logging.info(f"Metrics appended to: {csv_path}")


def create_formatted_metrics_mean_ci(metrics_mean_ci: dict) -> dict:
    """Create formatted metrics mean CI."""
    for k, v in metrics_mean_ci.items():
        if isinstance(v, list):
            mean, ci = float(v[0]), float(v[1])
            logging.info(f"Metric {k}: {mean:.4f} ± {ci:.4f}")
            metrics_mean_ci[k] = f"{mean:.4f} ± {ci:.4f}"
    return metrics_mean_ci


def filter_datasets(dataset_meta_info: dict, datasets: Optional[List[str]]) -> List[str]:
    """Select datasets from the dataset meta info."""
    if datasets is None:
        # Dataset filtering not specified, return all datasets
        return list(dataset_meta_info.keys())
    else:
        datasets = datasets.split(",")
        # Check if datasets are valid
        for dataset in datasets:
            if dataset not in dataset_meta_info:
                raise ValueError(f"Dataset {dataset} not found in dataset meta info")
        # Return all requsted datasets
        return datasets


def run_inference_and_evaluation(
    model_config: ModelLoadConfig,
    inference_config: InferenceConfig,
    eval_config: EvaluationConfig,
    dataset_meta_info: dict,
    datasets: Optional[List[str]],
    out_dir: str,
    num_repeats: int = 1,
    confidence_level: float = 0.95,
    violin_plot_metrics: Optional[List[str]] = None,
    log_exp_name: bool = False,
    clean_up_disk: bool = False,
    skip_evaluation: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Run inference and optional evaluation on specified datasets.

    Longform inference is automatically detected based on text characteristics
    when longform_mode="auto" (default). Use longform_mode="always" or "never"
    for explicit control.

    Args:
        model_config: Configuration for loading the model.
        inference_config: Configuration for inference.
        eval_config: Configuration for evaluation.
        dataset_meta_info: Dictionary containing dataset metadata.
        datasets: List of dataset names to run inference and evaluation on. If None, all datasets in the
                  dataset meta info will be processed.
        out_dir: Output directory for results.
        num_repeats: Number of times to repeat inference (for CI estimation).
        confidence_level: Confidence level for CI calculation.
        violin_plot_metrics: Metrics to include in violin plots.
        log_exp_name: Whether to include experiment name in output paths.
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

    # Load model
    model, checkpoint_name = load_magpie_model(model_config)

    # Log architecture summary and get MoE info + FLOPs metrics
    moe_info, flops_per_component = log_model_architecture_summary(model)

    # Add experiment name prefix if requested
    if log_exp_name and model_config.checkpoint_file:
        exp_name = get_experiment_name_from_checkpoint_path(model_config.checkpoint_file)
        checkpoint_name = f"{exp_name}__{checkpoint_name}"

    # Build full checkpoint identifier (include MoE info if present)
    full_checkpoint_name = (
        f"{checkpoint_name}_{moe_info}{inference_config.build_identifier()}_SV_{eval_config.sv_model}"
    )

    # Create inference runner (auto-detects longform based on config.longform_mode)
    logging.info(f"Longform mode: {inference_config.longform_mode}")
    runner = MagpieInferenceRunner(model, inference_config)

    # Tracking metrics across datasets
    ssim_per_dataset = []
    cer_per_dataset = []
    all_datasets_filewise_metrics = {}

    # CSV headers
    csv_header = (
        "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,"
        "wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,"
        "ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,"
        "ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative,"
        "utmosv2_avg,total_gen_audio_seconds,frechet_codec_distance"
    )

    for dataset in datasets:
        logging.info(f"Processing dataset: {dataset}")

        meta = dataset_meta_info[dataset]
        manifest_records = read_manifest(meta['manifest_path'])
        language = meta.get('whisper_language', 'en')

        # Prepare dataset metadata (remove evaluation-specific keys)
        dataset_meta_for_dl = copy.deepcopy(meta)
        for key in ["whisper_language", "load_cached_codes_if_available"]:
            dataset_meta_for_dl.pop(key, None)

        # Setup output directories
        eval_dir = os.path.join(out_dir, f"{full_checkpoint_name}_{dataset}")
        audio_dir = os.path.join(eval_dir, "audio")
        os.makedirs(eval_dir, exist_ok=True)

        # Setup CSV files
        per_run_csv = os.path.join(eval_dir, "all_experiment_metrics.csv")
        write_csv_header_if_needed(per_run_csv, csv_header)

        metrics_all_repeats = []
        filewise_metrics_all_repeats = []

        for repeat_idx in range(num_repeats):
            logging.info(f"Repeat {repeat_idx + 1}/{num_repeats} for dataset {dataset}")

            repeat_audio_dir = os.path.join(audio_dir, f"repeat_{repeat_idx}")
            os.makedirs(repeat_audio_dir, exist_ok=True)

            # Create dataset and run inference
            test_dataset = runner.create_dataset({dataset: dataset_meta_for_dl})

            if len(test_dataset) != len(manifest_records):
                raise ValueError(
                    f"Dataset length mismatch: {len(test_dataset)} vs {len(manifest_records)} manifest records"
                )

            rtf_metrics_list, _, codec_file_paths = runner.run_inference_on_dataset(
                dataset=test_dataset,
                output_dir=repeat_audio_dir,
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

            with open(os.path.join(eval_dir, f"{dataset}_rtf_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(mean_rtf, f, indent=4)

            if skip_evaluation:
                logging.info("Skipping evaluation as requested.")
                continue

            # Run evaluation
            eval_config_for_dataset = EvaluationConfig(
                sv_model=eval_config.sv_model,
                asr_model_name=eval_config.asr_model_name,
                language=language,
                with_utmosv2=eval_config.with_utmosv2,
                with_fcd=eval_config.with_fcd,
                codec_model_path=eval_config.codec_model_path,
            )

            metrics, filewise_metrics = evaluate_generated_audio_dir(
                manifest_path=meta['manifest_path'],
                audio_dir=meta['audio_dir'],
                generated_audio_dir=repeat_audio_dir,
                config=eval_config_for_dataset,
            )

            metrics_all_repeats.append(metrics)
            filewise_metrics_all_repeats.extend(filewise_metrics)

            # Save metrics
            with open(os.path.join(eval_dir, f"{dataset}_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            with open(os.path.join(eval_dir, f"{dataset}_filewise_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(filewise_metrics, f, indent=4)

            # Append to per-run CSV
            append_metrics_to_csv(per_run_csv, full_checkpoint_name, dataset, metrics)

            # Create violin plot for this repeat
            violin_path = Path(eval_dir) / f"{dataset}_violin_{repeat_idx}.png"
            create_violin_plot(filewise_metrics, violin_plot_metrics, violin_path)

            # Delete temporary predicted codes files
            for codec_file_path in codec_file_paths:
                os.remove(codec_file_path)

        if skip_evaluation or not metrics_all_repeats:
            continue

        # Store for combined plot
        all_datasets_filewise_metrics[dataset] = filewise_metrics_all_repeats

        # Compute mean with confidence interval across repeats
        metrics_mean_ci = compute_mean_with_confidence_interval(
            metrics_all_repeats,
            STANDARD_METRIC_KEYS,
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
    if len(all_datasets_filewise_metrics) > 1:
        combined_plot_path = os.path.join(out_dir, f"{full_checkpoint_name}_combined_violin_plot.png")
        create_combined_box_plot(all_datasets_filewise_metrics, violin_plot_metrics, combined_plot_path)

    # Clean up if requested
    if clean_up_disk:
        logging.info(f"Cleaning up output directory: {out_dir}")
        shutil.rmtree(out_dir)

    # Return averaged metrics
    if ssim_per_dataset and cer_per_dataset:
        return np.mean(cer_per_dataset), np.mean(ssim_per_dataset)
    return None, None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='MagpieTTS Inference and Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model loading arguments
    model_group = parser.add_argument_group('Model Loading')
    model_group.add_argument(
        '--hparams_files',
        type=str,
        default=None,
        help='Comma-separated paths to hparams.yaml files (use with --checkpoint_files)',
    )
    model_group.add_argument(
        '--checkpoint_files',
        type=str,
        default=None,
        help='Comma-separated paths to .ckpt files (use with --hparams_files)',
    )
    model_group.add_argument(
        '--nemo_files',
        type=str,
        default=None,
        help='Comma-separated paths to .nemo files (alternative to hparams + checkpoint)',
    )
    model_group.add_argument(
        '--codecmodel_path',
        type=str,
        required=True,
        help='Path to the audio codec model',
    )
    model_group.add_argument(
        '--hparams_file_from_wandb',
        action='store_true',
        help='Set if hparams file was exported from wandb',
    )
    model_group.add_argument(
        '--legacy_codebooks',
        action='store_true',
        help='Use legacy codebook indices (for old checkpoints)',
    )
    model_group.add_argument(
        '--legacy_text_conditioning',
        action='store_true',
        help='Use legacy text conditioning (for old checkpoints)',
    )

    # Dataset and output arguments
    data_group = parser.add_argument_group('Dataset and Output')
    data_group.add_argument(
        '--datasets_json_path',
        type=str,
        required=True,
        default=None,
        help='Path to dataset configuration JSON file (will process all datasets in the file if --datasets is not specified)',
    )
    data_group.add_argument(
        '--datasets',
        type=str,
        default=None,
        help='Comma-separated list of dataset names to process using names from the datasets_json_path file.  If not specified, all datasets in the datasets_json_path will be processed.',
    )
    data_group.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Output directory for generated audio and metrics',
    )
    data_group.add_argument(
        '--log_exp_name',
        action='store_true',
        help='Include experiment name in output folder name',
    )
    data_group.add_argument(
        '--clean_up_disk',
        action='store_true',
        help='Delete output directory after completion',
    )

    # Inference arguments
    infer_group = parser.add_argument_group('Inference Parameters')
    # Add model specific parameters
    for field in fields(ModelInferenceParameters):
        extra_args = {"type": field.type}
        if field.type == bool:
            extra_args["action"] = "store_true"
            del extra_args["type"]
        if field.name == "estimate_alignment_from_layers" or field.name == "apply_prior_to_layers":
            extra_args["help"] = "Must be a comma separate string. Not enclosed in brackets"
            extra_args["type"] = str
        elif field.name == "eos_detection_method":
            extra_args["choices"] = [m.value for m in EOSDetectionMethod]
        infer_group.add_argument(f"--{field.name}", **extra_args)
    infer_group.add_argument('--batch_size', type=int, default=32)
    infer_group.add_argument('--use_cfg', action='store_true', help='Enable classifier-free guidance')
    infer_group.add_argument(
        '--longform_mode',
        type=str,
        default='auto',
        choices=['auto', 'always', 'never'],
        help='Longform inference mode: auto (detect from text), always, or never',
    )
    infer_group.add_argument(
        '--longform_word_threshold',
        type=int,
        default=40,
        help='Word threshold for auto-detection of longform text',
    )
    infer_group.add_argument(
        '--longform_max_decoder_steps',
        type=int,
        default=50000,
        help='Maximum decoder steps for longform inference',
    )

    # Local transformer / MaskGit arguments
    infer_group.add_argument('--use_local_transformer', action='store_true')
    infer_group.add_argument('--maskgit_n_steps', type=int, default=3)
    infer_group.add_argument('--maskgit_noise_scale', type=float, default=0.0)
    infer_group.add_argument('--maskgit_fixed_schedule', type=int, nargs='+', default=None)
    infer_group.add_argument(
        '--maskgit_sampling_type',
        default=None,
        choices=["default", "causal", "purity_causal", "purity_default"],
    )

    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument(
        '--run_evaluation',
        action='store_true',
        help='Run evaluation after inference (default: False, inference only)',
    )
    eval_group.add_argument('--sv_model', type=str, default="titanet", choices=["titanet", "wavlm"])
    eval_group.add_argument('--asr_model_name', type=str, default="nvidia/parakeet-tdt-1.1b")
    eval_group.add_argument('--num_repeats', type=int, default=1)
    eval_group.add_argument('--confidence_level', type=float, default=0.95)
    eval_group.add_argument('--disable_utmosv2', action='store_true')
    eval_group.add_argument(
        '--violin_plot_metrics',
        type=str,
        nargs='*',
        default=['cer', 'pred_context_ssim', 'utmosv2'],
    )
    eval_group.add_argument('--disable_fcd', action='store_true', help="Disable Frechet Codec Distance computation")

    # Quality targets (for CI/CD)
    target_group = parser.add_argument_group('Quality Targets')
    target_group.add_argument('--cer_target', type=float, default=None)
    target_group.add_argument('--ssim_target', type=float, default=None)

    return parser


def main(argv=None):
    """Entry point for MagpieTTS inference and evaluation.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.
    """
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    dataset_meta_info = load_evalset_config(args.datasets_json_path)
    datasets = filter_datasets(dataset_meta_info, args.datasets)

    logging.info(f"Loaded {len(datasets)} datasets: {', '.join(datasets)}")

    # Determine mode and validate
    has_checkpoint_mode = (
        args.hparams_files is not None
        and args.checkpoint_files is not None
        and args.hparams_files != "null"
        and args.checkpoint_files != "null"
    )
    has_nemo_mode = args.nemo_files is not None and args.nemo_files != "null"

    if not has_checkpoint_mode and not has_nemo_mode:
        parser.error("You must provide either:\n 1. --hparams_files and --checkpoint_files\n 2. --nemo_files")

    # Build configurations
    # Use higher max_decoder_steps for longform inference when mode is 'always'
    if args.longform_mode == 'always':
        max_decoder_steps = args.longform_max_decoder_steps
    elif args.longform_mode == 'auto':
        # Use longform steps if any text appears long (will be checked in runner)
        max_decoder_steps = args.longform_max_decoder_steps
    else:  # 'never'
        max_decoder_steps = 440
    model_inference_parameters = {}
    for field in fields(ModelInferenceParameters):
        field = field.name
        if field == "max_decoder_steps":
            model_inference_parameters[field] = max_decoder_steps
            continue
        arg_from_cmdline = vars(args)[field]
        if arg_from_cmdline is not None:
            if field in ["estimate_alignment_from_layers", "apply_prior_to_layers"]:
                model_inference_parameters[field] = parse_layer_list(vars(args)[field])
            else:
                model_inference_parameters[field] = vars(args)[field]

    inference_config = InferenceConfig(
        model_inference_parameters=ModelInferenceParameters.from_dict(model_inference_parameters),
        batch_size=args.batch_size,
        use_cfg=args.use_cfg,
        apply_attention_prior=args.apply_attention_prior,
        use_local_transformer=args.use_local_transformer,
        maskgit_n_steps=args.maskgit_n_steps,
        maskgit_noise_scale=args.maskgit_noise_scale,
        maskgit_fixed_schedule=args.maskgit_fixed_schedule,
        maskgit_sampling_type=args.maskgit_sampling_type,
        longform_mode=args.longform_mode,
        longform_word_threshold=args.longform_word_threshold,
    )

    eval_config = EvaluationConfig(
        sv_model=args.sv_model,
        asr_model_name=args.asr_model_name,
        with_utmosv2=not args.disable_utmosv2,
        with_fcd=not args.disable_fcd,
        codec_model_path=args.codecmodel_path if not args.disable_fcd else None,
    )

    cer, ssim = None, None

    # Run for each model (checkpoint or nemo)
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
            )

            cer, ssim = run_inference_and_evaluation(
                model_config=model_config,
                inference_config=inference_config,
                eval_config=eval_config,
                dataset_meta_info=dataset_meta_info,
                datasets=datasets,
                out_dir=args.out_dir,
                num_repeats=args.num_repeats,
                confidence_level=args.confidence_level,
                violin_plot_metrics=args.violin_plot_metrics,
                log_exp_name=args.log_exp_name,
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
            )

            cer, ssim = run_inference_and_evaluation(
                model_config=model_config,
                inference_config=inference_config,
                eval_config=eval_config,
                dataset_meta_info=dataset_meta_info,
                datasets=datasets,
                out_dir=args.out_dir,
                num_repeats=args.num_repeats,
                confidence_level=args.confidence_level,
                violin_plot_metrics=args.violin_plot_metrics,
                log_exp_name=args.log_exp_name,
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
