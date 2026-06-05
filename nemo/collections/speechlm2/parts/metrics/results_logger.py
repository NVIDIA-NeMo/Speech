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

import json
import os
import time
from collections import defaultdict
from typing import List, Optional

import soundfile as sf
import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.parts.metrics.mcq_evaluator import MCQEvaluator
from nemo.utils import logging

"""
Utilities for logging evaluation results of SpeechLM2 collection models.

This file provides helper functionality for saving audio outputs and structured
metadata during evaluation or inference of Duplex speech-to-speech / TTS models.
It is primarily responsible for:

    - Writing predicted waveforms to disk.
    - Merging user and model audio into multi-channel WAV files for analysis.
    - Exporting metadata (reference text, predictions, ASR output) into JSONL format.
    - Saving auxiliary debug artifacts such as:
        * teacher-forced predictions,
        * reference audio,
        * trimmed outputs,
        * end-of-utterance (EOU) probability signals.

Unlike other files in this directory, which focus on metric evaluation, this module
is dedicated to persisting model outputs — including predicted audio samples and
their associated metadata — for later inspection and analysis.

Key abstraction:
    - `ResultsLogger`: A lightweight utility class that manages audio dumping
      and metadata bookkeeping across inference batches.
"""


def get_rank():
    """Get the current rank in distributed training"""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0
    except:
        return 0


def get_world_size():
    """Get the world size in distributed training"""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        else:
            return 1
    except:
        return 1


class ResultsLogger:
    """
    Saves audios and a json file with the model outputs.
    Now supports distributed training with result merging across ranks.
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.audio_save_path = os.path.join(save_path, "pred_wavs")
        os.makedirs(self.audio_save_path, exist_ok=True)
        self.metadata_save_path = os.path.join(save_path, "metadatas")
        os.makedirs(self.metadata_save_path, exist_ok=True)
        self.cached_results = defaultdict(list)
        self.normalizer = EnglishTextNormalizer()

        # Initialize MCQ evaluator with manifest directory in the same folder
        self.mcq_evaluator = None
        # Get the directory where this file is located
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        mcq_manifest_dir = os.path.join(current_file_dir, 'manifest_files')

        if os.path.exists(mcq_manifest_dir):
            self.mcq_evaluator = MCQEvaluator(mcq_manifest_dir)
            logging.info(f"MCQ evaluator initialized with manifest directory: {mcq_manifest_dir}")
        else:
            logging.warning(
                f"MCQ manifest directory not found at {mcq_manifest_dir}. "
                f"MCQ evaluation (openbookqa, mmsu) will be skipped."
            )

    def reset(self):
        self.cached_results = defaultdict(list)
        metadata_files = os.listdir(self.metadata_save_path)
        for f in metadata_files:
            open(os.path.join(self.metadata_save_path, f), 'w').close()

        # clean out any existing .wav predictions safely
        try:
            audio_files = os.listdir(self.audio_save_path)
            for f in audio_files:
                if f.lower().endswith(".wav"):
                    try:
                        os.remove(os.path.join(self.audio_save_path, f))
                    except FileNotFoundError:
                        pass  # already gone
                    except Exception:
                        logging.warning(f"Failed to remove audio file {f} during reset.", stack_info=False)
        except FileNotFoundError:
            # directory somehow missing: recreate it
            os.makedirs(self.audio_save_path, exist_ok=True)

        return self

    @staticmethod
    def merge_and_save_audio(
        out_audio_path: str, pred_audio: torch.Tensor, pred_audio_sr: int, user_audio: torch.Tensor, user_audio_sr: int
    ) -> None:
        # Handle case where user_audio might be None
        if user_audio is not None:
            user_audio = resample(user_audio.float(), user_audio_sr, pred_audio_sr)
            T1, T2 = pred_audio.shape[0], user_audio.shape[0]
            max_len = max(T1, T2)
            pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
            user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)

            # Combine audio in a multichannel audio
            combined_wav = torch.cat(
                [
                    user_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                    pred_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                ],
                dim=0,
            ).squeeze()
        else:
            combined_wav = pred_audio.unsqueeze(0).detach().cpu()

        # Save audio using soundfile
        os.makedirs(os.path.dirname(out_audio_path), exist_ok=True)
        sf.write(out_audio_path, combined_wav.numpy().astype('float32').T, pred_audio_sr)
        logging.info(f"Audio saved at: {out_audio_path}")

    def update(
        self,
        name: str,
        refs: list[str],
        hyps: list[str],
        samples_id: list[str],
        asr_hyps: Optional[list[str]] = None,
        pred_audio: Optional[torch.Tensor] = None,
        pred_audio_sr: Optional[int] = None,
        user_audio: Optional[torch.Tensor] = None,
        user_audio_sr: Optional[int] = None,
        src_refs: Optional[list[str]] = None,
        src_hyps: Optional[list[str]] = None,
        target_audio: Optional[torch.Tensor] = None,
        pred_audio_tf: Optional[torch.Tensor] = None,
        pre_audio_trimmed: Optional[torch.Tensor] = None,
        eou_pred: Optional[torch.Tensor] = None,
        fps: Optional[float] = None,
        results=None,
        tokenizer=None,
        reference_audio: Optional[torch.Tensor] = None,
    ):
        rank = get_rank()

        for i in range(len(refs)):
            sample_id = samples_id[i][:150]
            # Add rank info to audio filename to avoid conflicts
            if pred_audio is not None:
                out_audio_path = os.path.join(self.audio_save_path, f"{name}_{sample_id}_rank{rank}.wav")
                self.merge_and_save_audio(
                    out_audio_path,
                    pred_audio[i],
                    pred_audio_sr,
                    user_audio[i] if user_audio is not None else None,
                    user_audio_sr,
                )

            # Save additional audio artifacts if provided (from upstream)
            if pred_audio_tf is not None:
                out_audio_path_tf = os.path.join(self.audio_save_path, f"{name}_{sample_id}_rank{rank}_tf.wav")
                self.merge_and_save_audio(
                    out_audio_path_tf,
                    pred_audio_tf[i],
                    pred_audio_sr,
                    user_audio[i] if user_audio is not None else None,
                    user_audio_sr,
                )

            if target_audio is not None:
                out_audio_path_gt = os.path.join(self.audio_save_path, f"{name}_{sample_id}_rank{rank}_GT.wav")
                self.merge_and_save_audio(
                    out_audio_path_gt,
                    target_audio[i],
                    pred_audio_sr,
                    user_audio[i] if user_audio is not None else None,
                    user_audio_sr,
                )

            # Create a wav with eou prediction for debug purposes
            if eou_pred is not None and fps is not None:
                out_audio_path_eou = os.path.join(self.audio_save_path, f"{name}_{sample_id}_rank{rank}_eou.wav")
                repeat_factor = int(pred_audio_sr / fps)
                eou_pred_wav = (
                    eou_pred[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, repeat_factor)
                )  # (B, T, repeat_factor)
                eou_pred_wav = eou_pred_wav.view(1, -1)  # (B, T * repeat_factor)
                eou_pred_wav = eou_pred_wav.float() * 0.8  # make 1 audible and keep 0 as total silence
                sf.write(
                    out_audio_path_eou,
                    eou_pred_wav.squeeze().unsqueeze(0).detach().cpu().numpy().astype('float32').T,
                    pred_audio_sr,
                )

            if pre_audio_trimmed is not None:
                out_audio_path_trimmed = os.path.join(
                    self.audio_save_path, f"{name}_{sample_id}_rank{rank}_pred_trimmed.wav"
                )
                sf.write(
                    out_audio_path_trimmed,
                    pre_audio_trimmed[i].squeeze().unsqueeze(0).detach().cpu().numpy().astype('float32').T,
                    pred_audio_sr,
                )

            if reference_audio is not None:
                out_audio_path_ref = os.path.join(
                    self.audio_save_path, f"{name}_{sample_id}_rank{rank}_spk_reference.wav"
                )
                sf.write(
                    out_audio_path_ref,
                    reference_audio[i].squeeze().unsqueeze(0).detach().cpu().numpy().astype('float32').T,
                    pred_audio_sr,
                )

            # Build metadata dictionary
            out_dict = {
                "id": sample_id,
                "target_text": refs[i],
                "pred_text": hyps[i],
                "pred_audio": asr_hyps[i] if asr_hyps is not None else None,
            }

            # Add source text fields if provided (DuplexSTTModel)
            if src_refs is not None:
                out_dict["src_text"] = src_refs[i]
            if src_hyps is not None:
                out_dict["pred_src_text"] = src_hyps[i] if src_hyps[i] is not None else ""

            # Add tokenizer results if provided (from upstream)
            if results is not None:
                if tokenizer is not None:
                    out_dict['tokens_text'] = " ".join(tokenizer.ids_to_tokens(results['tokens_text'][i]))
                else:
                    out_dict['tokens_text'] = results['tokens_text'][i].tolist()

            self.cached_results[name].append(out_dict)

    def _merge_rank_files(self, dataset_name: str) -> List[dict]:
        """
        Merge results from all ranks for a given dataset.
        Only executed by rank 0.
        """
        rank = get_rank()
        world_size = get_world_size()

        if rank != 0:
            return []

        all_results = []

        # Wait a bit for all ranks to finish writing their files
        time.sleep(2)

        # Collect results from all ranks
        for r in range(world_size):
            rank_file = os.path.join(self.metadata_save_path, f"{dataset_name}_rank{r}.json")

            # Wait for the file to exist (with timeout)
            wait_time = 0
            max_wait = 30  # seconds
            while not os.path.exists(rank_file) and wait_time < max_wait:
                time.sleep(1)
                wait_time += 1

            if os.path.exists(rank_file):
                try:
                    with open(rank_file, 'r', encoding='utf-8') as fin:
                        rank_results = [json.loads(line) for line in fin if line.strip()]
                        if isinstance(rank_results, list):
                            all_results.extend(rank_results)
                        else:
                            logging.warning(f"Unexpected format in {rank_file}: {type(rank_results)}")
                except Exception as e:
                    logging.warning(f"Failed to read {rank_file}: {e}")
            else:
                logging.warning(f"Rank file {rank_file} not found after waiting {max_wait} seconds")

        # logging.info(f"Total merged results for {dataset_name}: {len(all_results)} items")
        return all_results

    def compute_and_save(
        self, special_subset_names: Optional[List[str]] = None, mcq_subset_names: Optional[List[str]] = None
    ):
        """
        Saves all cached results. Now supports distributed training:
        1. Each rank saves its own results with rank suffix
        2. Rank 0 collects and merges all results into final files
        3. Computes metrics on the merged results

        Args:
            special_subset_names: A list of validation subset names to compute accuracy for (QA datasets).
            mcq_subset_names: A list of MCQ dataset names to compute MCQ accuracy for.

        Returns:
            A dictionary of calculated metrics (accuracy and empty_rate) for the special subsets.
            E.g., {'web-qa': {'acc': 0.8, 'empty_rate': 0.1}, 'openbookqa': {'mcq_acc': 0.75, 'empty_rate': 0.05}, ...}
        """
        if special_subset_names is None:
            special_subset_names = ['web-qa', 'llama-qa', 'trivia-qa']

        if mcq_subset_names is None:
            mcq_subset_names = ['openbookqa', 'mmsu']

        rank = get_rank()
        world_size = get_world_size()
        metrics_results = {}

        # Step 1: Each rank saves its own results with rank suffix
        for name, results_list in self.cached_results.items():
            rank_json_path = os.path.join(self.metadata_save_path, f"{name}_rank{rank}.json")
            with open(rank_json_path, 'w', encoding='utf-8') as fout:
                for item in results_list:
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            logging.info(f"Rank {rank} metadata file for {name} dataset saved at: {rank_json_path}")

        # Step 2: Synchronize all ranks before merging
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Step 3: Only rank 0 merges all results and computes final metrics
        if rank == 0:
            for name in self.cached_results.keys():
                # Merge results from all ranks
                merged_results = self._merge_rank_files(name)

                # Save merged results
                final_json_path = os.path.join(self.metadata_save_path, f"{name}.json")
                with open(final_json_path, 'w', encoding='utf-8') as fout:
                    for item in merged_results:
                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                logging.info(f"Final merged metadata file for {name} dataset saved at: {final_json_path}")

                # Compute metrics on merged results
                if name in special_subset_names and merged_results:
                    correct_count = 0
                    empty_count = 0
                    total_count = len(merged_results)

                    for item in merged_results:
                        pred_text = item["pred_text"].strip()
                        normalized_pred = self.normalizer(pred_text)

                        if not normalized_pred:
                            empty_count += 1
                            continue

                        pred_words = set(normalized_pred.split())

                        target_text = item["target_text"]
                        possible_targets = target_text.split(';')

                        is_correct = False
                        for target_option in possible_targets:
                            normalized_target_option = self.normalizer(target_option.strip())
                            target_words = set(normalized_target_option.split())

                            if not target_words or target_words.issubset(pred_words):
                                is_correct = True
                                break

                        if is_correct:
                            correct_count += 1

                    acc = correct_count / total_count if total_count > 0 else 0.0
                    empty_rate = empty_count / total_count if total_count > 0 else 0.0

                    metrics_results[name] = {'acc': torch.tensor(acc), 'empty_rate': torch.tensor(empty_rate)}
                    logging.info(
                        f"Metrics for special subset '{name}': Accuracy={acc}, Empty Rate={empty_rate} (total samples: {total_count})"
                    )

                # Compute MCQ metrics for MCQ datasets
                if name in mcq_subset_names and merged_results and self.mcq_evaluator:
                    try:
                        mcq_metrics = self.mcq_evaluator.evaluate(name, merged_results)

                        # Log empty rate info
                        empty_rate = mcq_metrics['empty_rate']
                        logging.info(
                            f"MCQ empty rate for '{name}': {empty_rate*100:.1f}% ({mcq_metrics['num_empty']}/{mcq_metrics['num_samples']})"
                        )

                        # Store only the accuracy metric for wandb logging
                        # Use the naming convention: [dataset name]_mcq_acc
                        metrics_results[name] = {
                            'mcq_acc': torch.tensor(mcq_metrics['acc']),
                            'empty_rate': torch.tensor(empty_rate),
                        }
                        logging.info(
                            f"MCQ metrics for '{name}': Accuracy={mcq_metrics['acc']*100:.2f}% ({mcq_metrics['num_correct']}/{mcq_metrics['num_samples']})"
                        )
                    except Exception as e:
                        logging.error(f"Failed to compute MCQ metrics for '{name}': {e}")
                elif name in mcq_subset_names and not self.mcq_evaluator:
                    logging.warning(f"MCQ evaluator not initialized, skipping MCQ evaluation for '{name}'")

        # Step 4: Broadcast metrics from rank 0 to all other ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
            # Convert metrics to a format that can be broadcasted
            if rank == 0:
                metrics_to_broadcast = {}
                for name, metrics in metrics_results.items():
                    metrics_to_broadcast[name] = {}
                    # Handle both regular acc and mcq_acc
                    if 'acc' in metrics:
                        metrics_to_broadcast[name]['acc'] = metrics['acc'].item()
                    if 'mcq_acc' in metrics:
                        metrics_to_broadcast[name]['mcq_acc'] = metrics['mcq_acc'].item()
                    if 'empty_rate' in metrics:
                        metrics_to_broadcast[name]['empty_rate'] = metrics['empty_rate'].item()
            else:
                metrics_to_broadcast = {}

            # Broadcast the metrics
            broadcast_list = [metrics_to_broadcast]
            torch.distributed.broadcast_object_list(broadcast_list, src=0)

            # Reconstruct metrics_results on all ranks
            if rank != 0:
                metrics_results = {}
                for name, metrics in broadcast_list[0].items():
                    metrics_results[name] = {}
                    if 'acc' in metrics:
                        metrics_results[name]['acc'] = torch.tensor(metrics['acc'])
                    if 'mcq_acc' in metrics:
                        metrics_results[name]['mcq_acc'] = torch.tensor(metrics['mcq_acc'])
                    if 'empty_rate' in metrics:
                        metrics_results[name]['empty_rate'] = torch.tensor(metrics['empty_rate'])

        return metrics_results
