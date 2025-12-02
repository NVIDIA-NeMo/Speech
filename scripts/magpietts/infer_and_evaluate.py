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
Inference and Evaluation script used for CI and NeMo model evaluation on custom datasets.
Please use this script as an example of how to do inference with MagpieTTS, but this script is otherwise unsupported
for general use cases.
"""
import argparse
import copy
import glob
import json
import os
import shutil
import time
from functools import partial
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scripts.magpietts.evalset_config as evalset_config
import scripts.magpietts.evaluate_generated_audio as evaluate_generated_audio
import soundfile as sf
import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from PIL import Image

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPATokenizer
from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset
from nemo.collections.tts.models import MagpieTTSModel

# EVALUATION_DATASETS is the full list of datasets for evaluation of a new model.
EVALUATION_DATASETS = (
    "riva_hard_digits,riva_hard_letters,riva_hard_money,riva_hard_short,vctk,libritts_seen,libritts_test_clean"
)


def compute_mean_and_confidence_interval(metrics_list, metric_keys, confidence=0.90):
    metrics = {}
    for key in metric_keys:
        measurements = [m[key] for m in metrics_list]
        mean = np.mean(measurements)
        std_err = stats.sem(measurements)

        confidence_interval = std_err * stats.t.ppf((1 + confidence) / 2, len(measurements) - 1)
        print(f"{key}: {mean} +/- {confidence_interval}")
        metrics[key] = "{:.4f} +/- {:.4f}".format(mean, confidence_interval)
    return metrics


def update_config(model_cfg, codecmodel_path, legacy_codebooks=False, legacy_text_conditioning=False):
    '''helper function to rename older yamls from t5 to magpie'''
    model_cfg.codecmodel_path = codecmodel_path
    if hasattr(model_cfg, 'text_tokenizer'):
        # Backward compatibility for models trained with absolute paths in text_tokenizer
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
    model_cfg.train_ds = None
    model_cfg.validation_ds = None
    model_cfg.legacy_text_conditioning = legacy_text_conditioning
    if "t5_encoder" in model_cfg:
        model_cfg.encoder = model_cfg.t5_encoder
        del model_cfg.t5_encoder
    if "t5_decoder" in model_cfg:
        model_cfg.decoder = model_cfg.t5_decoder
        del model_cfg.t5_decoder
    if hasattr(model_cfg, 'decoder') and hasattr(model_cfg.decoder, 'prior_eps'):
        # Added to prevent crash after removing arg from transformer_2501.py in https://github.com/blisc/NeMo/pull/56
        del model_cfg.decoder.prior_eps
    if hasattr(model_cfg, 'use_local_transformer') and model_cfg.use_local_transformer:
        # For older checkpoints trained with a different parameter name
        model_cfg.local_transformer_type = "autoregressive"
        del model_cfg.use_local_transformer
    if hasattr(model_cfg, 'downsample_factor'):
        # Backward compatibility for models trained with the config option`downsample_factor` which was later renamed to `frame_stacking_factor`
        model_cfg.frame_stacking_factor = model_cfg.downsample_factor
        del model_cfg.downsample_factor
    if legacy_codebooks:
        # Added to address backward compatibility arising from
        #  https://github.com/blisc/NeMo/pull/64
        print(
            "WARNING: Using legacy codebook indices for backward compatibility. Should only be used with old checkpoints."
        )
        num_audio_tokens_per_codebook = model_cfg.num_audio_tokens_per_codebook
        model_cfg.forced_num_all_tokens_per_codebook = num_audio_tokens_per_codebook
        model_cfg.forced_audio_eos_id = num_audio_tokens_per_codebook - 1
        model_cfg.forced_audio_bos_id = num_audio_tokens_per_codebook - 2
        if model_cfg.model_type == 'decoder_context_tts':
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 3
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 4
            model_cfg.forced_mask_token_id = num_audio_tokens_per_codebook - 5
        else:
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 1
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 2
    if hasattr(model_cfg, 'sample_rate'):
        # This was removed from the config and is now in the model class
        sample_rate = model_cfg.sample_rate
        del model_cfg.sample_rate
    else:
        sample_rate = None
    return model_cfg, sample_rate


def update_ckpt(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        if 't5_encoder' in key:
            new_key = key.replace('t5_encoder', 'encoder')
            new_state_dict[new_key] = state_dict[key]
        elif 't5_decoder' in key:
            new_key = key.replace('t5_decoder', 'decoder')
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def delete_old_generated_files(output_dir):
    # Delete any leftover generated files from previous runs as these can confuse the evaluation
    print(f"Deleting old generated files in: {output_dir} ...")
    for f in glob.glob(f"{output_dir}/predicted_codes*.pt"):
        os.remove(f)
    for f in glob.glob(f"{output_dir}/predicted_audio*.wav"):
        os.remove(f)
    for f in glob.glob(f"{output_dir}/cross_attn_map_*.png"):
        os.remove(f)


def create_violin_plots(metrics: List[dict], metric_keys: List[str], output_png: str):
    # Create dataframe from list of dicts
    df = pd.DataFrame(metrics)

    # Plot the violin plots for all DataFrames side by side
    num_columns = len(metric_keys)
    width = num_columns * 5
    fig, axs = plt.subplots(1, num_columns, figsize=(width, 4))

    for i, column in enumerate(metric_keys):
        assert column in df
        # Create empty lists to store the parts objects for each DataFrame
        # Plot the violin plots for each DataFrame
        axs[i].violinplot(df[column], showmedians=True, positions=[i], widths=0.5)

        axs[i].set_title(column)
        axs[i].set_xticks([i])
        axs[i].set_xticklabels([column])
        axs[i].grid(True, linestyle="dotted")

        # Calculate and display the mean value for each DataFrame
        mean = df[column].mean()
        sem = df[column].sem()
        axs[i].plot(i, mean, "o", color="red", markersize=4, label="Mean (95%CI)")

        label_numeric = f"{mean:.2f}±{1.96 * sem:.2f}"
        axs[i].text(i + 0.06, mean, label_numeric, ha="center", va="top")

    # Create a single legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_png, format="png", bbox_inches="tight")


def create_combined_violin_plots(dataset_metrics: dict, metric_keys: List[str], output_png: str):
    """
    Create box plots comparing multiple datasets for each metric in a single figure.

    Args:
        dataset_metrics: Dictionary where keys are dataset names and values are lists of metric dictionaries
        metric_keys: List of metric names to plot
        output_png: Output file path for the combined plot
    """
    # Prepare data for plotting
    datasets = list(dataset_metrics.keys())
    num_datasets = len(datasets)
    num_metrics = len(metric_keys)

    # Create figure with subplots for each metric
    fig, axs = plt.subplots(1, num_metrics, figsize=(num_metrics * 6, 6))

    # Handle case where there's only one metric (axs won't be an array)
    if num_metrics == 1:
        axs = [axs]

    # Define colors for different datasets
    colors = plt.cm.Set3(np.linspace(0, 1, num_datasets))

    for metric_idx, metric in enumerate(metric_keys):
        ax = axs[metric_idx]

        # Collect data for all datasets for this metric
        all_data = []
        positions = []
        dataset_labels = []

        for dataset_idx, dataset in enumerate(datasets):
            df = pd.DataFrame(dataset_metrics[dataset])
            if metric in df.columns:
                data = df[metric].dropna()
                all_data.append(data)
                positions.append(dataset_idx + 1)
                dataset_labels.append(dataset)

        # Create box plots
        if all_data:
            bp = ax.boxplot(
                all_data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'red', 'markersize': 6},
            )

            # Color the box plots
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)

            # Add mean labels for each dataset
            for i, (data, pos) in enumerate(zip(all_data, positions)):
                mean = data.mean()
                sem = data.sem()

                label_numeric = f"{mean:.3f}±{1.96 * sem:.3f}"
                ax.text(pos + 0.1, mean, label_numeric, ha="left", va="center", fontsize=8)

        # Set labels and title
        ax.set_title(f"{metric.upper()}", fontsize=12, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(dataset_labels, rotation=45, ha='right')
        ax.grid(True, linestyle="dotted", alpha=0.7)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)

        # Set y-axis limit for CER metrics
        if 'cer' in metric.lower():
            ax.set_ylim(0, 0.3)

    # Add overall title
    fig.suptitle("Performance Comparison Across Datasets", fontsize=14, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Combined violin plot saved to: {output_png}")


def run_inference(
    hparams_file,
    checkpoint_file,
    nemo_file,
    datasets,
    out_dir,
    temperature,
    topk,
    codecmodel_path,
    use_cfg,
    cfg_scale,
    batch_size,
    sv_model,
    asr_model_name,
    num_repeats=1,
    apply_attention_prior=False,
    attention_prior_epsilon=1e-3,
    attention_prior_lookahead_window=10,
    estimate_alignment_from_layers=None,
    apply_prior_to_layers=None,
    start_prior_after_n_audio_steps=10,
    confidence_level=0.95,
    use_local_transformer=False,
    maskgit_n_steps=3,
    maskgit_noise_scale=0.0,
    maskgit_fixed_schedule=None,
    maskgit_sampling_type=None,
    legacy_codebooks=False,
    legacy_text_conditioning=False,
    clean_up_disk=False,
    hparams_file_from_wandb=False,
    log_exp_name=False,
    violin_plot_metrics=None,
    eos_detection_method=None,
    ignore_finished_sentence_tracking=False,
    with_utmosv2=True,
):
    # Avoid lists as default values and apply default value in function
    if violin_plot_metrics is None:
        violin_plot_metrics = ['cer', 'pred_context_ssim', 'utmosv2']
    # Load model
    if hparams_file is not None and checkpoint_file is not None:
        model_cfg = OmegaConf.load(hparams_file)
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg

        if hparams_file_from_wandb:
            model_cfg = model_cfg.value

        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config(
                model_cfg, codecmodel_path, legacy_codebooks, legacy_text_conditioning
            )

        model = MagpieTTSModel(cfg=model_cfg)
        model.use_kv_cache_for_inference = True

        # Load weights from checkpoint file
        print("Loading weights from checkpoint")
        ckpt = torch.load(checkpoint_file, weights_only=False)
        state_dict = update_ckpt(ckpt['state_dict'])
        model.load_state_dict(state_dict)
        checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]
    elif nemo_file is not None:
        model_cfg = MagpieTTSModel.restore_from(nemo_file, return_config=True)
        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config(
                model_cfg, codecmodel_path, legacy_codebooks, legacy_text_conditioning
            )
        model = MagpieTTSModel.restore_from(nemo_file, override_config_path=model_cfg)
        model.use_kv_cache_for_inference = True
        checkpoint_name = nemo_file.split("/")[-1].split(".nemo")[0]
    else:
        raise ValueError("Need either a checkpoint and hparams file, or a nemo file.")

    if cfg_sample_rate is not None and cfg_sample_rate != model.sample_rate:
        raise ValueError("Sample rate in config and model do not match")

    print("Loaded weights.")
    model.cuda()
    model.eval()

    if log_exp_name:
        # the experiment name is the name of the directory two above the checkpoint path,
        # since training produces directories of the form `exp_name/checkpoints/checkpoint_name.ckpt`.
        exp_name = f"{os.path.basename(os.path.dirname(os.path.dirname(checkpoint_file)))}__"
    else:
        exp_name = ""

    # Build checkpoint name
    checkpoint_name = (
        f"{exp_name}{checkpoint_name}_Temp{temperature}_Topk{topk}_Cfg_{use_cfg}_{cfg_scale}_"
        f"Prior_{apply_attention_prior}_"
    )
    if apply_attention_prior:
        # Only add prior config details if prior is enabled (to avoid super long checkpoint names)
        checkpoint_name += (
            f"{attention_prior_epsilon}_{attention_prior_lookahead_window}_{start_prior_after_n_audio_steps}_"
            f"{''.join([str(l) for l in estimate_alignment_from_layers]) if estimate_alignment_from_layers is not None else 'None'}_"
            f"{''.join([str(l) for l in apply_prior_to_layers]) if apply_prior_to_layers is not None else 'None'}_"
        )
    checkpoint_name += (
        f"LT_{use_local_transformer}_"
        f"MaskGit_{maskgit_n_steps}_{maskgit_sampling_type}_{''.join([str(l) for l in maskgit_fixed_schedule]) if maskgit_fixed_schedule is not None else 'None'}_"
        f"SV_{sv_model}"
        f"EOS_{eos_detection_method}"
        f"IgnoreFST_{ignore_finished_sentence_tracking}"
    )

    dataset_meta_info = evalset_config.dataset_meta_info
    ssim_per_dataset = []
    cer_per_dataset = []
    all_datasets_filewise_metrics = {}  # Store filewise metrics for all datasets for combined violin plot
    if (not with_utmosv2) and ('utmosv2' in violin_plot_metrics):
        violin_plot_metrics.remove('utmosv2')
    for dataset in datasets:
        print(f"Evaluating dataset {dataset}")
        metrics_n_repeated = []
        manifest_records = read_manifest(dataset_meta_info[dataset]['manifest_path'])
        language = dataset_meta_info[dataset].get('whisper_language', 'en')
        dataset_meta_for_dl = copy.deepcopy(dataset_meta_info[dataset])
        for key in ["whisper_language", "load_cached_codes_if_available"]:
            if key in dataset_meta_for_dl:
                del dataset_meta_for_dl[key]

        dataset_meta = {dataset: dataset_meta_for_dl}

        eval_dir = os.path.join(out_dir, f"{checkpoint_name}_{dataset}")
        audio_dir = os.path.join(eval_dir, "audio")
        all_experiment_csv = os.path.join(eval_dir, "all_experiment_metrics.csv")
        os.makedirs(eval_dir, exist_ok=True)

        if not os.path.exists(all_experiment_csv):
            with open(all_experiment_csv, "w") as f:
                header = "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative,utmosv2,total_gen_audio_seconds"
                header += "\n"
                f.write(header)

        context_duration_min = model.cfg.get('context_duration_min', 5.0)
        context_duration_max = model.cfg.get('context_duration_max', 5.0)
        if context_duration_min < 5.0 and context_duration_max > 5.0:
            context_duration_min = 5.0
            context_duration_max = 5.0  # @pneekhara - For multiencoder models, I want fixed size contexts for fair eval. Not too important though.

        dataset_filewise_metrics_all_repeats = []  # Store metrics for all repeats of this dataset
        for repeat_idx in range(num_repeats):
            pred_audio_dir = os.path.join(audio_dir, f"repeat_{repeat_idx}")
            os.makedirs(pred_audio_dir, exist_ok=True)
            delete_old_generated_files(pred_audio_dir)

            test_dataset = MagpieTTSDataset(
                dataset_meta=dataset_meta,
                sample_rate=model.sample_rate,
                min_duration=0.5,
                max_duration=20,
                codec_model_samples_per_frame=model.codec_model_samples_per_frame,
                bos_id=model.bos_id,
                eos_id=model.eos_id,
                context_audio_bos_id=model.context_audio_bos_id,
                context_audio_eos_id=model.context_audio_eos_id,
                audio_bos_id=model.audio_bos_id,
                audio_eos_id=model.audio_eos_id,
                num_audio_codebooks=model.num_audio_codebooks,
                prior_scaling_factor=None,
                load_cached_codes_if_available=False,
                dataset_type='test',
                tokenizer_config=None,
                load_16khz_audio=model.model_type == 'single_encoder_sv_tts',
                use_text_conditioning_tokenizer=model.use_text_conditioning_encoder,
                text_conditioning_tokenizer_name=model.text_conditioning_tokenizer_name,
                pad_context_text_to_max_duration=model.pad_context_text_to_max_duration,
                context_duration_min=context_duration_min,
                context_duration_max=context_duration_max,
            )
            assert len(test_dataset) == len(
                manifest_records
            ), f"Dataset length and manifest length should be the same. Dataset length: {len(test_dataset)}, Manifest length: {len(manifest_records)}"

            test_dataset.text_tokenizer = model.tokenizer
            # Set phoneme prob = 1 for g2p
            g2p = None
            if isinstance(model.tokenizer, AggregatedTTSTokenizer):
                g2p = model.tokenizer.tokenizers["english_phoneme"].g2p
            elif isinstance(model.tokenizer, IPATokenizer):
                g2p = model.tokenizer.g2p
            if g2p is not None:
                g2p.phoneme_probability = 1.0

            test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=test_dataset.collate_fn,
                num_workers=2,
                shuffle=False,
            )

            item_idx = 0
            all_rtf_metrics = []
            codec_file_paths = []
            for bidx, batch in enumerate(test_data_loader):
                print(f"Processing batch {bidx} out of {len(test_data_loader)} of dataset {dataset}")
                batch_cuda = {}
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch_cuda[key] = batch[key].cuda()
                    else:
                        batch_cuda[key] = batch[key]

                st = time.time()
                (
                    predicted_audio,
                    predicted_audio_lens,
                    predicted_codes,
                    predicted_codes_lens,
                    rtf_metrics,
                    cross_attention_maps,
                    _,
                ) = model.infer_batch(
                    batch_cuda,
                    max_decoder_steps=440,
                    temperature=temperature,
                    topk=topk,
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                    return_cross_attn_probs=True,
                    apply_attention_prior=apply_attention_prior,
                    prior_epsilon=attention_prior_epsilon,
                    lookahead_window_size=attention_prior_lookahead_window,
                    estimate_alignment_from_layers=estimate_alignment_from_layers,
                    apply_prior_to_layers=apply_prior_to_layers,
                    start_prior_after_n_audio_steps=start_prior_after_n_audio_steps,
                    use_local_transformer_for_inference=use_local_transformer,
                    maskgit_n_steps=maskgit_n_steps,
                    maskgit_noise_scale=maskgit_noise_scale,
                    maskgit_fixed_schedule=maskgit_fixed_schedule,
                    maskgit_sampling_type=maskgit_sampling_type,
                    ignore_finished_sentence_tracking=ignore_finished_sentence_tracking,
                    eos_detection_method=eos_detection_method,
                )

                all_rtf_metrics.append(rtf_metrics)
                et = time.time()
                print(f"Time taken for inference: {et-st}", predicted_audio.size())
                for idx in range(predicted_audio.size(0)):
                    cross_attn_map_image = Image.fromarray(cross_attention_maps[idx])
                    cross_attn_map_image.save(os.path.join(pred_audio_dir, f"cross_attn_map_{item_idx}.png"))

                    predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                    predicted_audio_np = predicted_audio_np[: predicted_audio_lens[idx]]
                    audio_path = os.path.join(pred_audio_dir, f"predicted_audio_{item_idx}.wav")
                    sf.write(audio_path, predicted_audio_np, model.sample_rate)
                    codes_path = os.path.join(pred_audio_dir, f"predicted_codes_{item_idx}.pt")
                    predicted_codes_current = predicted_codes[idx, :, : predicted_codes_lens[idx]]  # C, T'
                    torch.save(predicted_codes_current, codes_path)
                    codec_file_paths.append(codes_path)
                    context_audio_path = manifest_records[item_idx].get('context_audio_filepath', None)
                    target_audio_path = manifest_records[item_idx].get('audio_filepath', None)
                    if context_audio_path is not None:
                        context_audio_path = os.path.join(dataset_meta_info[dataset]['audio_dir'], context_audio_path)
                    if target_audio_path is not None:
                        target_audio_path = os.path.join(dataset_meta_info[dataset]['audio_dir'], target_audio_path)
                    if os.path.exists(context_audio_path):
                        shutil.copy(context_audio_path, os.path.join(audio_dir, f"context_audio_{item_idx}.wav"))
                    if os.path.exists(target_audio_path):
                        shutil.copy(target_audio_path, os.path.join(audio_dir, f"target_audio_{item_idx}.wav"))
                    item_idx += 1

            mean_rtf_metrics = {}
            for key in all_rtf_metrics[0]:
                mean_rtf_metrics[key] = float(np.mean([m[key] for m in all_rtf_metrics]))

            metrics, filewise_metrics = evaluate_generated_audio.evaluate(
                dataset_meta[dataset]['manifest_path'],
                dataset_meta[dataset]['audio_dir'],
                pred_audio_dir,
                language=language,
                sv_model_type=sv_model,
                asr_model_name=asr_model_name,
                with_utmosv2=with_utmosv2,
            )
            metrics_n_repeated.append(metrics)
            dataset_filewise_metrics_all_repeats.extend(
                filewise_metrics
            )  # Collect all filewise metrics for combined plot

            with open(os.path.join(eval_dir, f"{dataset}_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            with open(os.path.join(eval_dir, f"{dataset}_filewise_metrics_{repeat_idx}.json"), "w") as f:
                # Indent for better readability
                json.dump(filewise_metrics, f, indent=4)

            with open(os.path.join(eval_dir, f"{dataset}_rtf_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(mean_rtf_metrics, f, indent=4)

            with open(all_experiment_csv, "a") as f:
                data = f"{checkpoint_name},{dataset},{metrics['cer_filewise_avg']},{metrics['wer_filewise_avg']},{metrics['cer_cumulative']},{metrics['wer_cumulative']},{metrics['ssim_pred_gt_avg']},{metrics['ssim_pred_context_avg']},{metrics['ssim_gt_context_avg']},{metrics['ssim_pred_gt_avg_alternate']},{metrics['ssim_pred_context_avg_alternate']},{metrics['ssim_gt_context_avg_alternate']},{metrics['cer_gt_audio_cumulative']},{metrics['wer_gt_audio_cumulative']},{metrics['utmosv2_avg']},{metrics['total_gen_audio_seconds']}"
                data += "\n"
                f.write(data)
                print(f"Wrote metrics for {checkpoint_name} and {dataset} to {all_experiment_csv}")

            output_png_file = Path(eval_dir) / f"{dataset}_violin_{repeat_idx}.png"
            create_violin_plots(filewise_metrics, violin_plot_metrics, output_png_file)

            # Clean up temporary codec files
            for codes_file in codec_file_paths:
                os.remove(codes_file)

        # Store filewise metrics for this dataset for combined plotting
        all_datasets_filewise_metrics[dataset] = dataset_filewise_metrics_all_repeats

        metric_keys = [
            'cer_filewise_avg',
            'wer_filewise_avg',
            'cer_cumulative',
            'wer_cumulative',
            'ssim_pred_gt_avg',
            'ssim_pred_context_avg',
            'ssim_gt_context_avg',
            'ssim_pred_gt_avg_alternate',
            'ssim_pred_context_avg_alternate',
            'ssim_gt_context_avg_alternate',
            'cer_gt_audio_cumulative',
            'wer_gt_audio_cumulative',
            'utmosv2_avg',
            'total_gen_audio_seconds',
        ]
        metrics_mean_ci = compute_mean_and_confidence_interval(
            metrics_n_repeated, metric_keys, confidence=confidence_level
        )
        all_experiment_csv_with_ci = os.path.join(out_dir, "all_experiment_metrics_with_ci.csv")
        if not os.path.exists(all_experiment_csv_with_ci):
            with open(all_experiment_csv_with_ci, "w") as f:
                header = "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative,utmosv2_avg,total_gen_audio_seconds"
                header += "\n"
                f.write(header)
        with open(all_experiment_csv_with_ci, "a") as f:
            data = f"{checkpoint_name},{dataset},{metrics_mean_ci['cer_filewise_avg']},{metrics_mean_ci['wer_filewise_avg']},{metrics_mean_ci['cer_cumulative']},{metrics_mean_ci['wer_cumulative']},{metrics_mean_ci['ssim_pred_gt_avg']},{metrics_mean_ci['ssim_pred_context_avg']},{metrics_mean_ci['ssim_gt_context_avg']},{metrics_mean_ci['ssim_pred_gt_avg_alternate']},{metrics_mean_ci['ssim_pred_context_avg_alternate']},{metrics_mean_ci['ssim_gt_context_avg_alternate']},{metrics_mean_ci['cer_gt_audio_cumulative']},{metrics_mean_ci['wer_gt_audio_cumulative']},{metrics_mean_ci['utmosv2_avg']},{metrics_mean_ci['total_gen_audio_seconds']}"
            data += "\n"
            f.write(data)
            print(f"Wrote metrics with CI for {checkpoint_name} and {dataset} to {all_experiment_csv_with_ci}")

        measurements = [m['ssim_pred_context_avg'] for m in metrics_n_repeated]
        ssim_current = np.mean(measurements)
        ssim_per_dataset.append(ssim_current)
        measurements = [m['cer_cumulative'] for m in metrics_n_repeated]
        cer_current = np.mean(measurements)
        cer_per_dataset.append(cer_current)

    # Create combined violin plot for all datasets
    if len(all_datasets_filewise_metrics) > 1:  # Only create combined plot if we have multiple datasets
        combined_output_png = os.path.join(out_dir, f"{checkpoint_name}_combined_violin_plot.png")
        create_combined_violin_plots(all_datasets_filewise_metrics, violin_plot_metrics, combined_output_png)

    # Average across datasets
    ssim = np.mean(ssim_per_dataset)
    cer = np.mean(cer_per_dataset)
    if clean_up_disk:
        shutil.rmtree(out_dir)
    return cer, ssim


def main():
    parser = argparse.ArgumentParser(description='Experiment Evaluation')
    parser.add_argument('--hparams_files', type=str, default=None)
    parser.add_argument('--hparams_file_from_wandb', action='store_true')
    parser.add_argument('--checkpoint_files', type=str, default=None)
    parser.add_argument('--nemo_files', type=str, default=None)
    parser.add_argument('--codecmodel_path', type=str, default=None, help="Path to codec model")
    parser.add_argument('--datasets', type=str, default=None)
    # Parameters for running inference experiments locally
    parser.add_argument('--out_dir', type=str, default="/datap/misc/Evals/LocalTransformerAblations2")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--use_cfg', action='store_true')
    parser.add_argument(
        '--use_local_transformer',
        action='store_true',
        help="Enables use of local transformer for inference; applies to both Autoregressive and MaskGit sampling.",
    )
    parser.add_argument('--maskgit_n_steps', type=int, default=3)
    parser.add_argument('--maskgit_noise_scale', type=float, default=0.0)
    parser.add_argument('--maskgit_fixed_schedule', type=int, nargs='+', default=None)
    parser.add_argument(
        '--maskgit_sampling_type', default=None, choices=["default", "causal", "purity_causal", "purity_default"]
    )
    parser.add_argument('--cfg_scale', type=float, default=2.5)
    parser.add_argument('--apply_attention_prior', action='store_true')
    parser.add_argument('--attention_prior_epsilon', type=float, default=0.1)
    parser.add_argument('--attention_prior_lookahead_window', type=int, default=5)
    parser.add_argument('--estimate_alignment_from_layers', type=str, default=None)
    parser.add_argument('--apply_prior_to_layers', type=str, default=None)
    parser.add_argument('--start_prior_after_n_audio_steps', type=int, default=0)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument(
        '--eos_detection_method',
        type=str,
        default="argmax_or_multinomial_any",
        choices=[
            "argmax_any",
            "argmax_or_multinomial_any",
            "argmax_all",
            "argmax_or_multinomial_all",
            "argmax_zero_cb",
            "argmax_or_multinomial_zero_cb",
        ],
    )
    # Parameters for evaluation
    parser.add_argument('--sv_model', type=str, default="titanet")  # titanet, wavlm
    parser.add_argument(
        '--asr_model_name', type=str, default="nvidia/parakeet-tdt-1.1b"
    )  # stt_en_conformer_transducer_large, nvidia/parakeet-ctc-0.6b
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--confidence_level', type=float, default=0.95)
    parser.add_argument('--legacy_codebooks', action='store_true')
    parser.add_argument('--legacy_text_conditioning', action='store_true')
    parser.add_argument('--ignore_finished_sentence_tracking', action='store_true')
    parser.add_argument('--clean_up_disk', action='store_true')
    parser.add_argument('--cer_target', type=float, default=None)
    parser.add_argument('--ssim_target', type=float, default=None)
    parser.add_argument('--disable_utmosv2', action='store_true', help="Disable UTMOSv2 computation")
    parser.add_argument(
        '--log_exp_name',
        action='store_true',
        help="Include the experiment name (derived from the checkpoint path) in the output folder name.",
    )
    parser.add_argument(
        '--violin_plot_metrics',
        type=str,
        nargs='*',
        default=['cer', 'pred_context_ssim', 'utmosv2'],
        help="Which metrics to add the violin plot.",
    )
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = EVALUATION_DATASETS

    estimate_alignment_from_layers = None
    if args.estimate_alignment_from_layers is not None:
        estimate_alignment_from_layers = [int(l.strip()) for l in args.estimate_alignment_from_layers.split(",")]
    apply_prior_to_layers = None
    if args.apply_prior_to_layers is not None:
        apply_prior_to_layers = [int(l.strip()) for l in args.apply_prior_to_layers.split(",")]

    run_inference_w_args = partial(
        run_inference,
        datasets=args.datasets.split(","),
        out_dir=args.out_dir,
        temperature=args.temperature,
        topk=args.topk,
        codecmodel_path=args.codecmodel_path,
        use_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        sv_model=args.sv_model,
        asr_model_name=args.asr_model_name,
        num_repeats=args.num_repeats,
        apply_attention_prior=args.apply_attention_prior,
        attention_prior_epsilon=args.attention_prior_epsilon,
        attention_prior_lookahead_window=args.attention_prior_lookahead_window,
        estimate_alignment_from_layers=estimate_alignment_from_layers,
        apply_prior_to_layers=apply_prior_to_layers,
        start_prior_after_n_audio_steps=args.start_prior_after_n_audio_steps,
        confidence_level=args.confidence_level,
        use_local_transformer=args.use_local_transformer,
        maskgit_n_steps=args.maskgit_n_steps,
        maskgit_noise_scale=args.maskgit_noise_scale,
        maskgit_fixed_schedule=args.maskgit_fixed_schedule,
        maskgit_sampling_type=args.maskgit_sampling_type,
        legacy_codebooks=args.legacy_codebooks,
        legacy_text_conditioning=args.legacy_text_conditioning,
        clean_up_disk=args.clean_up_disk,
        hparams_file_from_wandb=args.hparams_file_from_wandb,
        log_exp_name=args.log_exp_name,
        violin_plot_metrics=args.violin_plot_metrics,
        eos_detection_method=args.eos_detection_method,
        ignore_finished_sentence_tracking=args.ignore_finished_sentence_tracking,
        with_utmosv2=not args.disable_utmosv2,
    )

    cer, ssim = None, None
    # Mode 1: Run inference from provided hparams and checkpoint files
    if (
        (args.hparams_files is not None)
        and (args.checkpoint_files is not None)
        and (args.hparams_files != "null")
        and (args.checkpoint_files != "null")
    ):
        hparam_files = args.hparams_files.split(",")
        checkpoint_files = args.checkpoint_files.split(",")
        print("Running inference for hparams files: ", hparam_files)
        print("Running inference for checkpoint files: ", checkpoint_files)
        assert len(hparam_files) == len(
            checkpoint_files
        ), "Number of hparams files and checkpoint files should be the same."
        for hparams_file, checkpoint_file in zip(hparam_files, checkpoint_files):
            cer, ssim = run_inference_w_args(
                hparams_file=hparams_file,
                checkpoint_file=checkpoint_file,
                nemo_file=None,
            )
    # Mode 2: Run inference from a .nemo file
    elif args.nemo_files:
        print(f"Running inference for nemo file: {args.nemo_files}")
        for nemo_file in args.nemo_files.split(","):
            cer, ssim = run_inference_w_args(
                hparams_file=None,
                checkpoint_file=None,
                nemo_file=nemo_file,
            )
    else:
        parser.error(
            "You must provide a model to run. Please specify either:\n"
            "1. --hparams_files and --checkpoint_files\n"
            "2. --nemo_file\n"
        )
    if cer is not None and args.cer_target is not None and cer > float(args.cer_target):
        raise ValueError(f"cer_cumulative was {cer} which is higher than the target of {args.cer_target}")
    if ssim is not None and args.ssim_target is not None and ssim < float(args.ssim_target):
        raise ValueError(f"ssim_pred_context_avg was {ssim} which is slower than the target of {args.ssim_target}")


if __name__ == '__main__':
    main()
