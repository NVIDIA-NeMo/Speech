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
This script extends the Lhotse shards with audio codec codes.

Example of input shards:
    $ tree ${CUTS_DIR}
    ${CUTS_DIR}/
        cuts.000000.jsonl.gz
        cuts.000001.jsonl.gz
        ...

    $ tree ${TARGET_AUDIO_DIR}
    ${TARGET_AUDIO_DIR}/
        recording.000000.tar
        recording.000001.tar
        ...

    $ tree ${CONTEXT_AUDIO_DIR}
    ${CONTEXT_AUDIO_DIR}/
        recording.000000.tar
        recording.000001.tar
        ...

Example usage:
    export WANDB_API_KEY=${WANDB}
    python -u ${CODE_DIR}/scripts/magpietts/extend_lhotse_shards_with_audio_codes.py \
        --cuts-dir ${CUTS_DIR} \
        --target-audio-dir ${TARGET_AUDIO_DIR} \
        --context-audio-dir ${CONTEXT_AUDIO_DIR} \
        --output-dir ${RESULTS} \
        --codec-model-name ${CODEC_MODEL_NAME} \
        --codec-model-path ${CODEC_MODEL_PATH} \
        --codec-frame-rate ${CODEC_FRAME_RATE} \
        --devices ${DEVICES} \
        --num-nodes ${NUM_NODES} \
        --batch-size ${BATCH_SIZE} \
        --buffer-size ${BUFFER_SIZE} \
        --wandb-entity ${WANDB_ENTITY} \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-name ${WANDB_NAME} \
        --log-level "DEBUG" \
    2>&1 | tee ${LOG}/${WANDB_NAME}.stdout

Expected output:
    $ tree ${RESULTS}
    ${RESULTS}/
        21fpsCausalDecoder/
            target_codes/
                codes.000000.tar
                codes.000001.tar
                ...
            context_codes/
                codes.000000.tar
                codes.000001.tar
                ...
"""

import argparse
import glob
import logging
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
import wandb
from lhotse import CutSet
from lhotse.array import Array, TemporalArray
from lhotse.dataset import IterableDatasetWrapper, SimpleCutSampler
from lhotse.shar.writers.array import ArrayTarWriter
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nemo.collections.tts.models import AudioCodecModel


def compute_effective_audio_length(original_audio_tensor: torch.Tensor, samples_per_frame: int) -> int:
    """Computes the effective length of an audio tensor, padded to be a multiple of samples_per_frame."""
    original_len = original_audio_tensor.shape[0]
    effective_len = original_len
    if samples_per_frame > 0:
        effective_len = ((original_len + samples_per_frame - 1) // samples_per_frame) * samples_per_frame
    return effective_len


def collate_audio_vectors(
    audio_list: List[torch.Tensor], audio_lens_list: List[int], padding_value: Union[float, int]
) -> torch.Tensor:
    """
    Collate a list of audio vectors into a single tensor, handling padding for variable lengths.
    Returns a padded tensor.
    """
    assert all(len(t.shape) == 1 for t in audio_list), "Expected only 1-D input tensors."
    assert len(audio_list) == len(audio_lens_list), "Expected the same number of audio vectors and lengths."

    # Create a padded tensor with the maximum audio length from audio_lens_list, where its max length could be longer than
    # max length of `audio_list``. For example, `audio_lens_list` could be a multiple of the codec model samples per frame.
    result = audio_list[0].new_ones(len(audio_lens_list), max(audio_lens_list)) * padding_value
    for i, t in enumerate(audio_list):
        result[i, : t.shape[0]] = t
    return result


class AudioPairLhotseDataset(Dataset):
    """
    A Lhotse Dataset that processes a batch of MonoCuts (received as a CutSet)
    containing target and context audio.
    Designed to be used with a Lhotse sampler yielding CutSet batches.
    Handles loading audio and collating the batch within __getitem__.
    """

    def __init__(self, target_sample_rate: int, codec_model_samples_per_frame: int):
        self.target_sample_rate = target_sample_rate
        self.codec_model_samples_per_frame = codec_model_samples_per_frame

    def __getitem__(self, cuts: CutSet) -> Optional[Dict[str, Any]]:
        original_target_audios_list = []
        effective_target_lengths_list = []
        original_context_audios_list = []
        effective_context_lengths_list = []
        target_cut_ids_list = []
        shard_indices_list = []

        for cut in cuts:
            if not cut.has_custom("shard_origin"):
                err_msg = f"Cut {cut} is missing required key 'shard_origin'."
                logging.error(err_msg)
                raise ValueError(err_msg)
            if not cut.has_custom("context_recording"):
                err_msg = f"Cut {cut} is missing required key 'context_recording'."
                logging.error(err_msg)
                raise ValueError(err_msg)

            # Parse shard index from the custom field, handling potential errors
            origin_path = cut.custom["shard_origin"]
            match = re.search(r"cuts\.(\d+)\.jsonl\.gz$", origin_path)
            if match is None:
                raise ValueError(f"Could not parse shard index from shard_origin: {origin_path}")
            shard_idx_origin = int(match.group(1))

            # audio shape: (num_channels (1), num_samples) -> (num_samples)
            # resample to target sample rate
            target_audio = torch.from_numpy(cut.recording.resample(self.target_sample_rate).load_audio().squeeze(0))
            context_audio = torch.from_numpy(
                cut.context_recording.resample(self.target_sample_rate).load_audio().squeeze(0)
            )
            original_target_audios_list.append(target_audio)
            original_context_audios_list.append(context_audio)

            eff_target_len = compute_effective_audio_length(target_audio, self.codec_model_samples_per_frame)
            effective_target_lengths_list.append(eff_target_len)

            eff_context_len = compute_effective_audio_length(context_audio, self.codec_model_samples_per_frame)
            effective_context_lengths_list.append(eff_context_len)

            target_cut_ids_list.append(cut.id)
            shard_indices_list.append(shard_idx_origin)

        # Ensure lists are not empty before calling collate_audio_vectors.
        if not original_target_audios_list:
            err_msg = "AudioPairLhotseDataset.__getitem__ processed an empty CutSet or failed to load any audio data, resulting in an empty audio list."
            logging.error(err_msg)
            raise ValueError(err_msg)

        target_audio_padded_batch = collate_audio_vectors(
            original_target_audios_list, effective_target_lengths_list, padding_value=0.0
        )
        context_audio_padded_batch = collate_audio_vectors(
            original_context_audios_list, effective_context_lengths_list, padding_value=0.0
        )

        # TODO: is it really necessary to convert lengths to torch.int64? currently applying torch.int32.
        target_audio_lens_collated = torch.IntTensor(effective_target_lengths_list)
        context_audio_lens_collated = torch.IntTensor(effective_context_lengths_list)

        return {
            "target_audios": target_audio_padded_batch,
            "target_audio_lens": target_audio_lens_collated,
            "context_audios": context_audio_padded_batch,
            "context_audio_lens": context_audio_lens_collated,
            "target_cut_id": target_cut_ids_list,
            "shard_idx_origin": shard_indices_list,
        }


class CodecExtractor(pl.LightningModule):
    """
    LightningModule to extract codec codes. Manages DataLoader creation and
    distribution via predict_dataloader hook.
    """

    def __init__(
        self,
        model_path: str,
        cuts_dir: str,
        target_audio_dir: str,
        context_audio_dir: str,
        batch_size: int,
    ):
        super().__init__()
        self.model_path = model_path
        self.cuts_dir = Path(cuts_dir)
        self.target_audio_dir = Path(target_audio_dir)
        self.context_audio_dir = Path(context_audio_dir)
        self.batch_size = batch_size

        logging.info(f"Initializing `AudioPairLhotseDataset` with model path: {self.model_path}")
        # load the model. mapping to cpu is to avoid GPU mem spikes when initializing the model
        self.codec_model = AudioCodecModel.restore_from(restore_path=self.model_path, map_location='cpu', strict=False)
        self.codec_model.eval()
        logging.info("Codec model loaded.")

        # Placeholder for the rank-specific list of dataloaders
        self._rank_dataloaders: Optional[List[DataLoader]] = None

    def predict_dataloader(self) -> List[DataLoader]:
        """
        Creates and returns the list of DataLoaders assigned to the current rank.
        Caches the result to avoid redundant creation.

        This function is called by the Trainer to get the dataloaders for the current rank. This happens after
        intializing `model.predict()` but before any actual prediction steps (ie. calls to `model.predict_step()`) are executed.
        """
        # Return cached dataloaders if already created for this rank
        if self._rank_dataloaders is not None:
            return self._rank_dataloaders

        # Determine rank and world size
        try:
            # Prefer trainer attributes if available
            current_global_rank = self.global_rank
            world_size = self.trainer.world_size
        except AttributeError:
            # Fallback to torch.distributed if trainer attributes aren't set yet
            current_global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        logging.info(f"[Rank {current_global_rank}/{world_size}] Creating assigned subset of dataloaders...")

        # Find all shard files globally
        cuts_shard_pattern = str(self.cuts_dir / "cuts.*.jsonl.gz")
        all_cuts_shard_paths = sorted(glob.glob(cuts_shard_pattern))

        if not all_cuts_shard_paths:
            msg = f"[Rank {current_global_rank}/{world_size}] No input cut shards found matching pattern: {cuts_shard_pattern}. Cannot proceed."
            logging.error(msg)
            raise FileNotFoundError(msg)

        num_total_shards = len(all_cuts_shard_paths)

        # Verify shard indices are contiguous and start from 0 based on filenames (globally)
        first_idx_str = re.search(r"cuts\.(\d+)\.jsonl\.gz$", all_cuts_shard_paths[0]).group(1)
        last_idx_str = re.search(r"cuts\.(\d+)\.jsonl\.gz$", all_cuts_shard_paths[-1]).group(1)
        first_idx = int(first_idx_str)
        last_idx = int(last_idx_str)
        expected_last_idx = num_total_shards - 1
        if first_idx != 0:
            raise ValueError(f"Expected first shard index to be 0, but found {first_idx} in {all_cuts_shard_paths[0]}")
        if last_idx != expected_last_idx:
            raise ValueError(
                f"Expected last shard index to be {expected_last_idx}, but found {last_idx} in {all_cuts_shard_paths[-1]}"
            )
        logging.info(
            f"[Rank {current_global_rank}/{world_size}] Verified {num_total_shards} total shard files globally, with indices from {first_idx} to {last_idx}."
        )

        # Calculate the slice of original shard indices assigned to this rank
        is_distributed = world_size > 1
        assigned_shard_indices_for_rank = []

        if num_total_shards > 0:
            if not is_distributed:
                assigned_shard_indices_for_rank = list(range(num_total_shards))
                logging.info(
                    f"[Rank {current_global_rank}/{world_size}] Non-distributed mode. Will process all {num_total_shards} shards."
                )
            else:
                num_per_rank_base = num_total_shards // world_size
                num_with_extra = num_total_shards % world_size

                if current_global_rank < num_with_extra:
                    start_shard_offset = current_global_rank * (num_per_rank_base + 1)
                    num_shards_for_rank = num_per_rank_base + 1
                else:
                    # Offset by the shards handled by ranks with an extra one
                    start_shard_offset = num_with_extra + current_global_rank * num_per_rank_base
                    num_shards_for_rank = num_per_rank_base

                end_shard_offset = start_shard_offset + num_shards_for_rank
                assigned_shard_indices_for_rank = list(range(start_shard_offset, end_shard_offset))

                logging.info(
                    f"[Rank {current_global_rank}/{world_size}] Assigned original shard indices "
                    f"{start_shard_offset} through {end_shard_offset -1} "
                    f"({len(assigned_shard_indices_for_rank)} shards)"
                )

        if not assigned_shard_indices_for_rank:
            logging.info(
                f"[Rank {current_global_rank}/{world_size}] No shards assigned to this rank. Returning empty dataloader list. This usually happens when the number of shards is less than the number of ranks."
            )
            self._rank_dataloaders = []
            return []

        # Create DataLoaders only for the shards assigned to this rank
        dataloaders_for_rank = []
        for original_shard_idx in tqdm(
            assigned_shard_indices_for_rank,
            total=len(assigned_shard_indices_for_rank),
            desc=f">>> [Rank {current_global_rank}/{world_size}] Creating DataLoaders for its assigned shards",
        ):
            logging.debug(f"[Rank {current_global_rank}] Processing original shard {original_shard_idx}...")
            fields = {
                "cuts": [str(self.cuts_dir / f"cuts.{original_shard_idx:06d}.jsonl.gz")],
                "recording": [str(self.target_audio_dir / f"recording.{original_shard_idx:06d}.tar")],
                "context_recording": [str(self.context_audio_dir / f"recording.{original_shard_idx:06d}.tar")],
            }
            # Verify if all files exist
            if not all(Path(shard_filepaths[0]).is_file() for shard_filepaths in fields.values()):
                err_msg = f"[Rank {current_global_rank}/{world_size}] Missing one or more files for shard {original_shard_idx}. Files: {fields}"
                logging.error(err_msg)
                raise FileNotFoundError(err_msg)

            try:
                logging.debug(
                    f"[Rank {current_global_rank}] Loading CutSet for original shard {original_shard_idx}..."
                )
                shard_cutset = CutSet.from_shar(fields=fields)
                logging.debug(f"[Rank {current_global_rank}] Loaded CutSet for original shard {original_shard_idx}.")
            except Exception as e:
                logging.critical(
                    f"[Rank {current_global_rank}/{world_size}] CRITICAL ERROR: Failed to load CutSet from shar for original shard index {original_shard_idx}. \
                    Files attempted: {fields}. \
                    Error: {e}",
                    exc_info=True,
                )
                raise

            logging.debug(f"[Rank {current_global_rank}] Creating Sampler for original shard {original_shard_idx}...")
            # Explicitly set rank=0, world_size=1 to ensure sampler iterates the whole shard_cutset
            sampler = SimpleCutSampler(
                shard_cutset, max_cuts=self.batch_size, shuffle=False, drop_last=False, rank=0, world_size=1
            )
            logging.debug(f"[Rank {current_global_rank}] Creating Dataset for original shard {original_shard_idx}...")
            shard_dataset = AudioPairLhotseDataset(
                target_sample_rate=self.codec_model.sample_rate,
                codec_model_samples_per_frame=self.codec_model.samples_per_frame,
            )
            logging.debug(f"[Rank {current_global_rank}] Wrapping Dataset for original shard {original_shard_idx}...")
            iterable_dataset = IterableDatasetWrapper(
                dataset=shard_dataset,
                sampler=sampler,
            )
            logging.debug(
                f"[Rank {current_global_rank}] Creating DataLoader for original shard {original_shard_idx}..."
            )
            dl = DataLoader(
                dataset=iterable_dataset,
                batch_size=None,
                num_workers=1,  # Keep num_workers=1 for `IterableDatasetWrapper + SimpleCutSampler` to avoid duplicate batches.
                pin_memory=True,
            )
            logging.debug(
                f"[Rank {current_global_rank}] Appending DataLoader for original shard {original_shard_idx}..."
            )
            dataloaders_for_rank.append(dl)
            logging.debug(f"[Rank {current_global_rank}] Finished processing original shard {original_shard_idx}.")

        logging.info(
            f"[Rank {current_global_rank}/{world_size}] Created {len(dataloaders_for_rank)} DataLoaders for this rank."
        )
        # Cache the created dataloaders for this rank
        self._rank_dataloaders = dataloaders_for_rank
        return self._rank_dataloaders

    def forward(
        self,
        target_audios: torch.Tensor,
        target_audio_lens: torch.Tensor,
        context_audios: torch.Tensor,
        context_audio_lens: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        try:
            target_audios = target_audios.to(self.device)
            target_audio_lens = target_audio_lens.to(self.device)
            context_audios = context_audios.to(self.device)
            context_audio_lens = context_audio_lens.to(self.device)
            # NOTE: we avoided directly calling `self.codec_model.encode()` because it pads audios again.
            with torch.inference_mode():
                target_audios_encoded, target_audios_encoded_len = self.codec_model.audio_encoder(
                    audio=target_audios, audio_len=target_audio_lens
                )
                target_tokens = self.codec_model.quantize(
                    encoded=target_audios_encoded, encoded_len=target_audios_encoded_len
                )
                context_audios_encoded, context_audios_encoded_len = self.codec_model.audio_encoder(
                    audio=context_audios, audio_len=context_audio_lens
                )
                context_tokens = self.codec_model.quantize(
                    encoded=context_audios_encoded, encoded_len=context_audios_encoded_len
                )
            return {
                "target_codes": target_tokens.to(dtype=torch.int16, device="cpu"),
                "target_codes_lengths": target_audios_encoded_len.to(device="cpu"),
                "context_codes": context_tokens.to(dtype=torch.int16, device="cpu"),
                "context_codes_lengths": context_audios_encoded_len.to(device="cpu"),
            }
        except Exception as e:
            logging.error(
                f"[Rank {self.global_rank}/{self.world_size}] Error during batched codec encoding: {e}", exc_info=True
            )
            raise e

    def predict_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Optional[List[Dict[str, Any]]]:
        codes_dict = self(
            target_audios=batch["target_audios"],
            target_audio_lens=batch["target_audio_lens"],
            context_audios=batch["context_audios"],
            context_audio_lens=batch["context_audio_lens"],
        )

        target_codes_batch = codes_dict["target_codes"]
        target_codes_lens = codes_dict["target_codes_lengths"]
        context_codes_batch = codes_dict["context_codes"]
        context_codes_lens = codes_dict["context_codes_lengths"]

        target_cut_ids = batch["target_cut_id"]
        shard_indices_in_batch = batch["shard_idx_origin"]

        # The shard_indices list should ideally contain the *same* original index
        # for all items in a batch, because each DataLoader loads from only one shard.
        results = []
        batch_size = batch["target_audios"].shape[0]
        original_shard_idx = shard_indices_in_batch[0]
        if not all(idx == original_shard_idx for idx in shard_indices_in_batch):
            raise ValueError(
                f"Inconsistent shard indices within batch! Batch Index: {batch_idx}, Dataloader Index: {dataloader_idx}. Indices: {shard_indices_in_batch}."
            )

        if len(target_cut_ids) != batch_size or target_codes_batch.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch after inference! Input IDs: {len(target_cut_ids)}, "
                f"Input Audio Batch: {batch_size}, Output Codes Batch: {target_codes_batch.shape[0]}. "
                f"Batch Index: {batch_idx}, Dataloader Index: {dataloader_idx}"
            )

        for target_cut_id, target_codes, context_codes, target_codes_len, context_codes_len in zip(
            target_cut_ids, target_codes_batch, context_codes_batch, target_codes_lens, context_codes_lens
        ):
            results.append(
                {
                    "target_cut_id": target_cut_id,
                    "shard_idx": original_shard_idx,
                    "target_codes": target_codes[:, :target_codes_len],
                    "context_codes": context_codes[:, :context_codes_len],
                }
            )

        return results


class CodecPredictionWriter(BasePredictionWriter):
    """
    Writes codec predictions (target and context codes) to ArrayTarWriter shards asynchronously.
    Uses a ThreadPoolExecutor with a single worker to serialize writes and closing operations per shard,
    allowing potential overlap between prediction computation and I/O while closing writers early.
    """

    def __init__(
        self,
        output_dir: str,
        codec_model_name: str,
        codec_frame_rate: float,
    ):
        super().__init__(write_interval="batch")
        self.output_dir_base = Path(output_dir)
        self.codec_model_name = codec_model_name
        self.codec_frame_rate = codec_frame_rate
        self.rank: int = -1
        self.world_size: int = -1
        self.target_writers: Dict[int, ArrayTarWriter] = {}
        self.context_writers: Dict[int, ArrayTarWriter] = {}
        self.target_codes_dir: Optional[Path] = None
        self.context_codes_dir: Optional[Path] = None

        # Attributes for asynchronous writing and closing
        self.writer_lock: Optional[threading.Lock] = None
        self.bg_worker_thread: Optional[ThreadPoolExecutor] = None
        self.futures_per_shard: Optional[Dict[int, List[Future]]] = None
        self.closer_futures: Optional[List[Future]] = None  # Futures for the _wait_and_close_worker tasks
        self.last_processed_shard_idx: int = -1

    def setup(self, trainer: Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        self.rank = trainer.global_rank
        self.world_size = trainer.world_size
        logging.info(
            f"[Rank {self.rank}/{self.world_size}] Setting up CodecPredictionWriter for async writing with early close."
        )

        # Initialize async components
        self.writer_lock = threading.Lock()
        # Single worker ensures sequential execution of writes AND closes
        self.bg_worker_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'CodecWriterRank{self.rank}')
        self.futures_per_shard = defaultdict(list)
        self.closer_futures = []
        self.last_processed_shard_idx = -1

        # Create directories
        self.target_codes_dir = self.output_dir_base / self.codec_model_name / "target_codes"
        self.context_codes_dir = self.output_dir_base / self.codec_model_name / "context_codes"
        if self.rank == 0:
            self.target_codes_dir.mkdir(parents=True, exist_ok=True)
            self.context_codes_dir.mkdir(parents=True, exist_ok=True)
        if trainer.world_size > 1:
            torch.distributed.barrier()
        logging.info(f"[Rank {self.rank}/{self.world_size}] Setup complete. Writers will be created on demand.")

    def _get_or_create_writer(
        self, writer_dict: Dict[int, ArrayTarWriter], shard_idx: int, base_dir: Path
    ) -> ArrayTarWriter:
        # Lock needed as this might be called from main thread while closer task modifies dicts
        with self.writer_lock:
            if shard_idx not in writer_dict:
                output_filename = str(base_dir / f"codes.{shard_idx:06d}.tar")
                logging.debug(
                    f"[Rank {self.rank}/{self.world_size}] Creating writer for shard {shard_idx} (Thread-safe check): {output_filename}"
                )
                try:
                    writer = ArrayTarWriter(pattern=output_filename, shard_size=None, compression="numpy")
                    writer.__enter__()
                    writer_dict[shard_idx] = writer
                    logging.info(f"[Rank {self.rank}/{self.world_size}] Created writer for shard {shard_idx}")
                except Exception as e:
                    msg = f"[Rank {self.rank}/{self.world_size}] Failed to create writer for shard {shard_idx} (file: {output_filename}): {e}"
                    logging.error(msg, exc_info=True)
                    raise ValueError(msg)

            # Return writer even if it might be closed soon by a background task
            # The background task handles the actual closing.
            return writer_dict[shard_idx]

    def _write_worker(
        self,
        target_cut_id: str,
        shard_idx: int,
        target_codes: torch.Tensor,
        context_codes: torch.Tensor,
        target_writer: ArrayTarWriter,
        context_writer: ArrayTarWriter,
    ):
        """Worker function executed by the background thread to write a single item."""
        # Assuming target_writer and context_writer are valid when this task starts
        try:
            target_codes_array_manifest = TemporalArray(
                array=Array(storage_type="shar", storage_path="", storage_key="", shape=list(target_codes.shape)),
                temporal_dim=-1,
                frame_shift=1 / self.codec_frame_rate,
                start=0,
            )
            context_codes_array_manifest = TemporalArray(
                array=Array(storage_type="shar", storage_path="", storage_key="", shape=list(context_codes.shape)),
                temporal_dim=-1,
                frame_shift=1 / self.codec_frame_rate,
                start=0,
            )
            target_writer.write(key=target_cut_id, value=target_codes.numpy(), manifest=target_codes_array_manifest)
            context_writer.write(key=target_cut_id, value=context_codes.numpy(), manifest=context_codes_array_manifest)
            logging.debug(f"[Worker Rank {self.rank}] Wrote item {target_cut_id} for shard {shard_idx}")
        except Exception as e:
            msg = f"[Worker Rank {self.rank}] CRITICAL I/O ERROR writing item {target_cut_id} for shard {shard_idx}: {e}. Writer might be closed prematurely?"
            logging.error(msg, exc_info=True)
            raise ValueError(msg)

    def _wait_and_close_worker(self, shard_idx_to_close: int):
        """Waits for all write tasks of a shard, then closes and removes its writers."""
        logging.info(f"[Worker Rank {self.rank}] Starting closure process for shard {shard_idx_to_close}")
        # 1. Retrieve and remove the list of write futures for this shard
        # Do this early to prevent new futures being added for this closing shard?
        # No, write_on_batch_end logic prevents submission for old shards.
        write_futures = self.futures_per_shard.pop(shard_idx_to_close, [])

        # 2. Wait for all write operations for this shard to complete
        logging.info(
            f"[Worker Rank {self.rank}] Waiting for {len(write_futures)} write tasks for shard {shard_idx_to_close}..."
        )
        processed_write_futures = 0
        if write_futures:
            for f in write_futures:
                try:
                    f.result()  # Wait for completion
                    processed_write_futures += 1
                except Exception as e:
                    # Write worker already logged this, but log context here
                    logging.error(
                        f"[Worker Rank {self.rank}] Exception during write future.result() for shard {shard_idx_to_close}: {e}",
                        exc_info=False,
                    )
            logging.info(
                f"[Worker Rank {self.rank}] Completed {processed_write_futures}/{len(write_futures)} write tasks for shard {shard_idx_to_close}."
            )
        else:
            logging.warning(
                f"[Worker Rank {self.rank}] No write futures found to wait for shard {shard_idx_to_close} during close."
            )

        # 3. Safely remove and close the writers
        writers_closed_count = 0
        with self.writer_lock:  # Protect access to the writer dictionaries
            target_writer = self.target_writers.pop(shard_idx_to_close, None)
            context_writer = self.context_writers.pop(shard_idx_to_close, None)

        if target_writer:
            try:
                target_writer.close()
                logging.info(f"[Worker Rank {self.rank}] Closed target writer for shard {shard_idx_to_close}.")
                writers_closed_count += 1
            except Exception as e:
                logging.error(
                    f"[Worker Rank {self.rank}] Error closing target writer for shard {shard_idx_to_close}: {e}",
                    exc_info=True,
                )
        else:
            logging.warning(
                f"[Worker Rank {self.rank}] Target writer for shard {shard_idx_to_close} not found during close."
            )

        if context_writer:
            try:
                context_writer.close()
                logging.info(f"[Worker Rank {self.rank}] Closed context writer for shard {shard_idx_to_close}.")
                writers_closed_count += 1
            except Exception as e:
                logging.error(
                    f"[Worker Rank {self.rank}] Error closing context writer for shard {shard_idx_to_close}: {e}",
                    exc_info=True,
                )
        else:
            logging.warning(
                f"[Worker Rank {self.rank}] Context writer for shard {shard_idx_to_close} not found during close."
            )

        logging.info(
            f"[Worker Rank {self.rank}] Finished closure process for shard {shard_idx_to_close}. Closed {writers_closed_count} writers."
        )

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        predictions: Optional[List[Dict[str, Any]]],
        batch_indices: Optional[List[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if not predictions:
            err_msg = f"[Rank {self.rank}/{self.world_size}] Received empty predictions list for batch_idx {batch_idx}, dataloader_idx {dataloader_idx}. Skipping."
            logging.error(err_msg)
            raise ValueError(err_msg)

        current_shard_idx = predictions[0]["shard_idx"]
        if not all(p["shard_idx"] == current_shard_idx for p in predictions):
            raise ValueError(
                f"[Rank {self.rank}] Inconsistent shard indices within batch! Batch Index: {batch_idx}, Dataloader Index: {dataloader_idx}."
            )

        # Check for shard change and submit closer task for the previous shard
        if current_shard_idx != self.last_processed_shard_idx and self.last_processed_shard_idx != -1:
            logging.info(
                f"[Rank {self.rank}] Shard index changed from {self.last_processed_shard_idx} to {current_shard_idx}. "
                f"Submitting closure task for shard {self.last_processed_shard_idx}."
            )
            try:
                closer_future = self.bg_worker_thread.submit(
                    self._wait_and_close_worker, self.last_processed_shard_idx
                )
                self.closer_futures.append(closer_future)
            except Exception as e:
                msg = f"[Rank {self.rank}] Failed to submit closer task for shard {self.last_processed_shard_idx}: {e}"
                logging.error(msg, exc_info=True)
                raise ValueError(msg)

        self.last_processed_shard_idx = current_shard_idx

        # Submit write tasks for each item in the current batch
        for prediction in predictions:
            try:
                target_cut_id = prediction["target_cut_id"]
                shard_idx = prediction["shard_idx"]
                target_codes = prediction["target_codes"]
                context_codes = prediction["context_codes"]

                # This needs the lock because the closer task might be removing entries concurrently
                target_writer = self._get_or_create_writer(self.target_writers, shard_idx, self.target_codes_dir)
                context_writer = self._get_or_create_writer(self.context_writers, shard_idx, self.context_codes_dir)

                # Submit the writing task
                write_future = self.bg_worker_thread.submit(
                    self._write_worker,
                    target_cut_id,
                    shard_idx,
                    target_codes,
                    context_codes,
                    target_writer,
                    context_writer,
                )
                self.futures_per_shard[shard_idx].append(write_future)
                logging.debug(f"[Rank {self.rank}] Submitted write task for item {target_cut_id}, shard {shard_idx}")

            except Exception as e:
                msg = f"[Rank {self.rank}] Error processing prediction item {prediction.get('target_cut_id', 'UNKNOWN')} from batch {batch_idx}: {e}"
                logging.error(msg, exc_info=True)
                raise ValueError(msg)

    def teardown(self, trainer: Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        logging.info(
            f"[Rank {self.rank}/{self.world_size}] Tearing down CodecPredictionWriter. Handling final shard and waiting for closers..."
        )

        # 1. Submit closer task for the very last processed shard (if any)
        final_shard_processed = self.last_processed_shard_idx
        if final_shard_processed != -1 and final_shard_processed in self.futures_per_shard:
            logging.info(
                f"[Rank {self.rank}] Submitting final closure task for last processed shard {final_shard_processed}."
            )
            try:
                closer_future = self.bg_worker_thread.submit(self._wait_and_close_worker, final_shard_processed)
                self.closer_futures.append(closer_future)
            except Exception as e:
                msg = f"[Rank {self.rank}] Failed to submit final closer task for shard {final_shard_processed}: {e}"
                logging.error(msg, exc_info=True)
                raise ValueError(msg)

        # 2. Wait for all closer tasks to complete
        num_closer_futures = len(self.closer_futures)
        logging.info(
            f"[Rank {self.rank}/{self.world_size}] Waiting for {num_closer_futures} background closer tasks to complete."
        )
        processed_closer_futures = 0
        if self.closer_futures:
            for future in tqdm(
                self.closer_futures,
                total=num_closer_futures,
                desc=f"[Rank {self.rank}/{self.world_size}] Finalizing Shard Closures",
                leave=False,
            ):
                try:
                    future.result()  # Wait and check for exceptions from the closer worker
                    processed_closer_futures += 1
                except Exception as e:
                    msg = f"[Rank {self.rank}/{self.world_size}] Exception caught during closer future.result(): {e}"
                    logging.error(msg, exc_info=True)
                    raise ValueError(msg)

            logging.info(
                f"[Rank {self.rank}/{self.world_size}] Completed {processed_closer_futures}/{num_closer_futures} closer tasks."
            )
        else:
            logging.info(f"[Rank {self.rank}/{self.world_size}] No closer tasks were submitted.")

        # 3. Shutdown the executor gracefully (all tasks should be done now)
        if self.bg_worker_thread:
            logging.info(f"[Rank {self.rank}/{self.world_size}] Shutting down background worker thread.")
            self.bg_worker_thread.shutdown(wait=True)
            self.bg_worker_thread = None

        # 4. Final sanity checks and cleanup
        remaining_writers = len(self.target_writers) + len(self.context_writers)
        if remaining_writers > 0:
            msg = f"[Rank {self.rank}/{self.world_size}] {remaining_writers} writers remain after teardown! This should not happen. Keys: Target {list(self.target_writers.keys())}, Context {list(self.context_writers.keys())}"
            logging.error(msg)
            raise ValueError(msg)

        remaining_futures = sum(len(futs) for futs in self.futures_per_shard.values())
        if remaining_futures > 0:
            msg = f"[Rank {self.rank}/{self.world_size}] {remaining_futures} write futures remain after teardown! This should not happen. Shards: {list(self.futures_per_shard.keys())}"
            logging.error(msg)
            raise ValueError(msg)

        self.target_writers.clear()
        self.context_writers.clear()
        self.futures_per_shard.clear()
        self.closer_futures.clear()

        logging.info(f"[Rank {self.rank}/{self.world_size}] Teardown complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuts-dir", type=str, required=True, help="Directory containing input cuts/cuts.*.jsonl.gz shards."
    )
    parser.add_argument(
        "--target-audio-dir", type=str, required=True, help="Directory containing target_audio/recording.*.tar shards."
    )
    parser.add_argument(
        "--context-audio-dir",
        type=str,
        required=True,
        help="Directory containing context_audio/recording.*.tar shards.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Base directory to save the output code shards.")
    parser.add_argument(
        "--codec-model-name",
        type=str,
        default="21fpsCausalDecoder",
        help="Name for codec model (used in output path).",
    )
    parser.add_argument(
        "--codec-model-path", type=str, required=True, help="Path to the NeMo codec model (.nemo file)."
    )
    parser.add_argument("--codec-frame-rate", type=float, default=21.5, help="Frame rate for codec model.")
    parser.add_argument("--devices", type=int, default=-1, help="Number of GPUs per node (-1 for all).")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed processing.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size PER GPU for codec inference.")
    parser.add_argument(
        "--buffer-size", type=int, default=256, help="Number of items to buffer before writing to TAR files."
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity.")
    parser.add_argument("--wandb-project", type=str, default="lhotse_codes_extraction", help="Wandb project.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    log_level_val = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level_val, format=log_format)

    codec_extractor = CodecExtractor(
        model_path=args.codec_model_path,
        cuts_dir=args.cuts_dir,
        target_audio_dir=args.target_audio_dir,
        context_audio_dir=args.context_audio_dir,
        batch_size=args.batch_size,
    )

    pred_writer = CodecPredictionWriter(
        output_dir=args.output_dir,
        codec_model_name=args.codec_model_name,
        codec_frame_rate=args.codec_frame_rate,
    )

    wandb_logger = None
    if args.wandb_entity and args.wandb_project:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"extract_codes_{args.codec_model_name}_{os.path.basename(args.cuts_dir)}",
            log_model=False,
        )
        logging.info(f"Wandb logging enabled to {args.wandb_entity}/{args.wandb_project}")

    strategy = DDPStrategy(find_unused_parameters=False) if torch.cuda.is_available() and args.devices != 1 else "auto"
    trainer = Trainer(
        devices=args.devices if torch.cuda.is_available() else 1,
        num_nodes=args.num_nodes,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        logger=wandb_logger,
        callbacks=[pred_writer],
        use_distributed_sampler=False,  # we should disable replacing or wrapping Lhostse CutSampler with a `DistributedSamplerWrapper` since Lhotse's sampler already handles distributed sampling.
    )

    logging.info(f"Starting prediction with {trainer.world_size} ranks.")
    trainer.predict(codec_extractor, return_predictions=False)
    logging.info("Prediction finished.")

    if trainer.is_global_zero and wandb_logger:
        wandb.finish()
        logging.info("Wandb run finished.")


if __name__ == "__main__":
    import torch.multiprocessing

    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # This exception occurs if the start method has already been set. We can safely ignore it.
        pass
    main()
