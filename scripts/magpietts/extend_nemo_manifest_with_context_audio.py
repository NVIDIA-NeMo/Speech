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

import argparse
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import lightning.pytorch as pl
import torch
import wandb
from lhotse.dataset.collation import collate_vectors
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

logger = logging.getLogger(__name__)

# Constant for masking identical items in similarity matrix
# Set below valid cosine similarity range [-1, 1] to ensure masked items are never selected
MASKED_SIMILARITY_VALUE = -2.0

"""
Usage:
python scripts/magpietts/extend_manifest_with_context_audio.py
    --manifest /path/to/input.json
    --audio-base-dir /path/to/audio
    --output-dir /path/to/output_sharded_manifests
    --batch-size 16
    --devices 2
    --num-nodes 1
    --flush-threshold-items 20000
    --num-workers 4
    --context-min-duration 3.0
    --context-min-ssim 0.6
    --max-speaker-items 20000  # optional, prevents OOM for large speakers

This script distributes speakers across DDP ranks. Each rank processes its assigned speakers
and writes a partial manifest. Rank 0 then merges these into a final manifest.

The --max-speaker-items parameter limits the size of the context pool per speaker to prevent OOM
when computing similarity matrices. If a speaker has more items than this limit, a random
sample will be used as the context pool, but all items will still be processed to find
their best context from this pool.

Input manifest example entry:
{
    "audio_filepath": "NVYT_40K_audios_wav/_8Kirz57BTY.wav",
    "text": "the face.",
    "speaker": "| Language:en Dataset:NVYT_2505 Speaker:_8Kirz57BTY_SPEAKER_01 |",
    "offset": 2.8,
    "duration": 0.48,
    "bandwidth": 10125,
    "stoi_squim": 0.98,
    "sisdr_squim": 18.235,
    "pesq_squim": 2.349,
    "dataset_id": "369a9f1a-65eb-4c09-8de3-8babea29da4c",
    "dataset_version": "2024_11_02_092919",
    "dataset_name": "yt_mixed",
    "normalized_text": "the face."
}

Output manifest example entry:
{
    "audio_filepath": "NVYT_40K_audios_wav/_8Kirz57BTY.wav",
    "text": "the face.",
    "speaker": "| Language:en Dataset:NVYT_2505 Speaker:_8Kirz57BTY_SPEAKER_01 |",
    "offset": 2.8,
    "duration": 0.48,
    "bandwidth": 10125,
    "stoi_squim": 0.98,
    "sisdr_squim": 18.235,
    "pesq_squim": 2.349,
    "dataset_id": "369a9f1a-65eb-4c09-8de3-8babea29da4c",
    "dataset_version": "2024_11_02_092919",
    "dataset_name": "yt_mixed",
    "normalized_text": "the face.",
    "context_audio_filepath": "NVYT_40K_audios_wav/_8Kirz57BTY.wav",
    "context_audio_offset": 5.6,
    "context_audio_duration": 6.0,
    "context_audio_text": "would you mind..", 
    "context_audio_normalized_text": "would you mind..",
    "context_audio_speaker_similarity": 0.85
}
"""


def check_speaker_format(item: str):
    """Enforce speaker format like '| Language:en Dataset:HiFiTTS Speaker:9136_other |'"""
    pattern = r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|"
    if not isinstance(item, str):
        return False
    return bool(re.match(pattern, item))


class SpeakerShardedAudioDataset(Dataset):
    def __init__(self, assigned_records_list, base_audio_dir, sample_rate=16000):
        self.sample_rate = sample_rate
        self.base_audio_dir = base_audio_dir
        self.processed_records = assigned_records_list

    def __len__(self):
        return len(self.processed_records)

    def get_wav_from_filepath(self, audio_filepath_rel, offset_in_sec=0, duration_in_sec=None):
        full_audio_filepath = os.path.join(self.base_audio_dir, audio_filepath_rel)
        try:
            features = AudioSegment.from_file(
                audio_file=full_audio_filepath,
                target_sr=self.sample_rate,
                int_values=False,  # TODO: if input is FLAC, then we should set this to True.
                offset=offset_in_sec,
                duration=duration_in_sec,
            )
        except Exception as e:
            logger.warning(
                f"[Skipping Wav Load] Failed for `{full_audio_filepath}` (relative: `{audio_filepath_rel}`, offset={offset_in_sec}, duration={duration_in_sec}): {e}"
            )
            return None, None
        audio_samples = features.samples
        return torch.tensor(audio_samples), torch.tensor(len(audio_samples)).long()

    def __getitem__(self, idx):
        item_info = self.processed_records[idx]

        audio, audio_length = self.get_wav_from_filepath(
            item_info["audio_filepath"], item_info["offset"], item_info["duration"]
        )
        if audio is None or audio_length is None:
            return None

        output_item = item_info.copy()
        output_item.update(
            {
                "audio": audio,
                "audio_length": audio_length,
            }
        )
        return output_item

    def collate_fn(self, batch):
        valid_items = [item for item in batch if item is not None]
        if not valid_items:
            return {
                "audios": torch.empty(0),
                "audio_lengths": torch.empty(0),
                "metadata_list": [],
                "parsed_speaker_ids_list": [],
            }

        audio_padded = collate_vectors([item["audio"] for item in valid_items], padding_value=0.0)
        audio_lengths = torch.tensor([item["audio_length"] for item in valid_items])
        metadata_list = [
            {k: v for k, v in item.items() if k not in ['audio', 'audio_length', 'parsed_speaker_id']}
            for item in valid_items
        ]
        parsed_speaker_ids_for_batch = [item['parsed_speaker_id'] for item in valid_items]

        return {
            "audios": audio_padded,
            "audio_lengths": audio_lengths,
            "metadata_list": metadata_list,
            "parsed_speaker_ids_list": parsed_speaker_ids_for_batch,
        }


class EmbeddingSimilarityExtractorSharded(pl.LightningModule):
    def __init__(
        self,
        output_dir: str,
        output_file_prefix: str,
        flush_threshold_items: int,
        context_min_duration: float,
        context_min_ssim: float,
        speaker_expected_counts_map: dict,
        initial_assigned_count: int,
        max_speaker_items: int = None,
    ):
        super().__init__()
        self.sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            'titanet_large', map_location=torch.device('cpu')
        )
        self.sv_model.eval()

        self.output_dir = Path(output_dir)
        self.output_file_prefix = output_file_prefix
        self.flush_threshold_items = flush_threshold_items
        self.context_min_duration = context_min_duration
        self.context_min_ssim = context_min_ssim
        self.speaker_expected_counts = speaker_expected_counts_map
        self.initial_assigned_count = initial_assigned_count
        self.max_speaker_items = max_speaker_items

        # Per-rank attributes
        self.output_file_path = None
        self.speaker_data_accumulator = defaultdict(list)
        self.total_accumulated_items = 0  # total number of items accumulated across all speakers for this rank
        self.processed_speakers_set = set()  # set of speakers that have been processed and flushed
        self.ready_to_flush_speaker_ids = set()  # set of speakers that have accumulated enough items to be flushed
        self.output_manifest_file = None
        # total num of items discarded due to no suitable context for this rank
        self.total_discarded_no_suitable_context_this_rank = 0
        self.total_items_written_this_rank = 0  # total items written to manifest by this rank

    def setup(self, stage: str):
        if stage == "predict":
            self.sv_model.to(self.device)
            self.output_file_path = self.output_dir / f"{self.output_file_prefix}_rank{self.global_rank}.json"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_manifest_file = open(self.output_file_path, "w", encoding="utf-8")
            logger.info(f"Writing partial manifest to: `{self.output_file_path}`")
            if self.max_speaker_items:
                logger.info(f"Max speaker items limit set to: {self.max_speaker_items}")
            else:
                logger.info("No max speaker items limit set (potential OOM risk for very large speakers)")
            logger.debug(f"Expected speaker counts for model: {self.speaker_expected_counts}")

    def forward(self, batch):
        with torch.no_grad():
            _, speaker_embeddings = self.sv_model.forward(
                input_signal=batch['audios'],
                input_signal_length=batch['audio_lengths'],
            )
            return speaker_embeddings

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if batch['audios'].nelement() == 0:
            return []

        speaker_embeddings_gpu = self(batch)

        output_items_for_batch_end = []
        for i, single_metadata_item in enumerate(batch["metadata_list"]):
            embedding_cpu_fp32 = speaker_embeddings_gpu[i].cpu().type(torch.float32)
            base_speaker_id_for_item = batch["parsed_speaker_ids_list"][i]

            processed_item = {
                "speaker_id_for_grouping": base_speaker_id_for_item,
                "embedding": embedding_cpu_fp32,
                "metadata": single_metadata_item,
            }
            output_items_for_batch_end.append(processed_item)

        return output_items_for_batch_end

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        for item in outputs:
            base_speaker_id = item['speaker_id_for_grouping']

            if base_speaker_id not in self.processed_speakers_set:
                self.speaker_data_accumulator[base_speaker_id].append(
                    {'embedding': item['embedding'], 'metadata': item['metadata']}
                )
                self.total_accumulated_items += 1

                expected_count = self.speaker_expected_counts[base_speaker_id]
                current_count = len(self.speaker_data_accumulator[base_speaker_id])

                if current_count == expected_count:
                    self.ready_to_flush_speaker_ids.add(base_speaker_id)
                    logger.debug(
                        f"Speaker {base_speaker_id} is complete with {current_count} items. Added to `ready_to_flush_speaker_ids`."
                    )
                elif current_count > expected_count:
                    msg = f"Speaker {base_speaker_id} has {current_count} items, but expected {expected_count}. Possible data inconsistency or error in expected counts."
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                msg = f"Received new item for already processed speaker '{base_speaker_id}'. This may indicate issues with data sharding, expected counts, or duplicate data."
                logger.error(msg)
                raise ValueError(msg)

        if self.total_accumulated_items >= self.flush_threshold_items and self.ready_to_flush_speaker_ids:
            self._process_and_flush_speakers_local()

    def _process_and_flush_speakers_local(self):
        speakers_to_process_now = list(self.ready_to_flush_speaker_ids)
        self.ready_to_flush_speaker_ids.clear()

        if not speakers_to_process_now:
            msg = "_process_and_flush_speakers_local called, but `speakers_to_process_now` is empty after list conversion. This is unexpected."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            f"Flushing {len(speakers_to_process_now)} completed speakers. "
            f"Current total accumulated items: {self.total_accumulated_items}"
        )

        for speaker_id in speakers_to_process_now:
            speaker_items = self.speaker_data_accumulator.pop(speaker_id)
            self.total_accumulated_items -= len(speaker_items)
            self.processed_speakers_set.add(speaker_id)

            # Apply speaker size limit to prevent OOM while processing all items
            all_items_to_process = speaker_items  # We want to process ALL items

            # Create context pool with original indices for easy identification
            if self.max_speaker_items and len(speaker_items) > self.max_speaker_items:
                logger.warning(
                    f"Speaker {speaker_id} has {len(speaker_items)} items, exceeding max limit of {self.max_speaker_items}. "
                    f"Using random sample of {self.max_speaker_items} items as context pool, but processing all {len(speaker_items)} items."
                )
                # Randomly sample with original indices preserved
                random.seed(12345)  # For reproducibility
                indexed_items = [(idx, item) for idx, item in enumerate(speaker_items)]
                sampled_indexed_items = random.sample(indexed_items, self.max_speaker_items)
                context_pool_items = [item for _, item in sampled_indexed_items]
                context_pool_original_indices = [idx for idx, _ in sampled_indexed_items]
            else:
                # Use all items as context pool
                context_pool_items = speaker_items
                context_pool_original_indices = list(range(len(speaker_items)))

            # NOTE: Now we compute similarities between ALL items and the context pool.
            # This limits the similarity matrix to N×M instead of N×N where M <= max_speaker_items.
            # Memory usage: N×M×4 bytes instead of N×N×4 bytes.
            all_embeddings = torch.stack([item['embedding'] for item in all_items_to_process])
            context_embeddings = torch.stack([item['embedding'] for item in context_pool_items])

            all_embeddings_norm = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
            context_embeddings_norm = torch.nn.functional.normalize(context_embeddings, p=2, dim=1)

            # Compute N×M similarity matrix: each row is similarities for one item against all context candidates
            similarity_matrix = torch.matmul(all_embeddings_norm, context_embeddings_norm.transpose(0, 1))

            # Mask positions where items are identical (same item appearing in both N and M sets)
            # Using original indices as identifiers. This prevents an item from being selected as its own context.
            # Create a mapping from original indices to context pool positions
            original_index_to_context_position = {}
            for context_pos, original_idx in enumerate(context_pool_original_indices):
                original_index_to_context_position[original_idx] = context_pos

            # Mask similarities for identical items
            for n_idx in range(len(all_items_to_process)):
                if n_idx in original_index_to_context_position:
                    context_pos = original_index_to_context_position[n_idx]
                    similarity_matrix[n_idx, context_pos] = MASKED_SIMILARITY_VALUE

            # Sort all similarities for each item to iterate through candidates
            # sorted_similarities_tensor will contain sorted similarities for each row (original item)
            # sorted_indices_tensor will contain indices in the context_pool
            sorted_similarities_tensor, sorted_indices_tensor = torch.sort(similarity_matrix, dim=1, descending=True)

            num_records_written_for_speaker = 0
            # Initialize a counter for items discarded for this specific speaker
            num_discarded_for_this_speaker_no_context = 0

            for i, current_item_data in enumerate(all_items_to_process):
                output_record = current_item_data['metadata'].copy()
                write_this_record = False

                # Iterate through potential candidates from context pool, sorted by similarity
                for candidate_rank in range(sorted_indices_tensor.size(1)):
                    candidate_ssim = sorted_similarities_tensor[i, candidate_rank].item()
                    context_pool_idx = sorted_indices_tensor[i, candidate_rank].item()

                    # if ANY candidate has similarity ≤ MASKED_SIMILARITY_VALUE, all subsequent ones will be ≤ MASKED_SIMILARITY_VALUE
                    # since similarities are sorted in descending order, we can break early
                    if candidate_ssim <= MASKED_SIMILARITY_VALUE:
                        break

                    # If SSIM is below threshold, stop searching for this item
                    if candidate_ssim < self.context_min_ssim:
                        break

                    # Check duration if SSIM is acceptable
                    best_meta_dict = context_pool_items[context_pool_idx]['metadata']
                    candidate_duration = best_meta_dict["duration"]

                    if candidate_duration >= self.context_min_duration:
                        # Found a suitable candidate, update record and stop searching for this item
                        record_update_dict = {
                            "context_speaker_similarity": candidate_ssim,
                            "context_audio_filepath": best_meta_dict["audio_filepath"],
                            "context_audio_offset": best_meta_dict["offset"],
                            "context_audio_duration": candidate_duration,
                            "context_audio_text": best_meta_dict["text"],
                        }
                        normalized_text_candidate = best_meta_dict.get("normalized_text", None)
                        if normalized_text_candidate is not None:
                            record_update_dict["context_audio_normalized_text"] = normalized_text_candidate

                        output_record.update(record_update_dict)
                        write_this_record = True
                        break

                if write_this_record:
                    self.output_manifest_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    num_records_written_for_speaker += 1
                else:
                    # This item will be discarded as no suitable context was found
                    num_discarded_for_this_speaker_no_context += 1

            # Accumulate to rank-level total
            self.total_discarded_no_suitable_context_this_rank += num_discarded_for_this_speaker_no_context
            self.total_items_written_this_rank += num_records_written_for_speaker

        if len(speakers_to_process_now) > 0:
            self.output_manifest_file.flush()  # ensure all data currently held in the buffer is immediately written to disk.
        logger.info(f"Flushing of completed speakers done. Local remaining items: {self.total_accumulated_items}")

    def on_predict_epoch_end(self):
        logger.info(
            f"Epoch end: Identifying remaining speakers to flush. "
            f"Speakers in accumulator: {len(self.speaker_data_accumulator)}, Already processed: {len(self.processed_speakers_set)}"
        )

        for speaker_id, items in list(self.speaker_data_accumulator.items()):
            if speaker_id not in self.processed_speakers_set:
                expected_count = self.speaker_expected_counts[speaker_id]
                actual_count = len(items)
                if actual_count == expected_count:
                    logger.info(
                        f"Epoch end: Speaker {speaker_id} is complete ({actual_count}/{expected_count}). Adding to ready set."
                    )
                    self.ready_to_flush_speaker_ids.add(speaker_id)
                else:
                    msg = f"Epoch end: Speaker {speaker_id} is still in accumulator with {actual_count} items, but expected {expected_count}. This indicates an issue, e.g., not all data for this speaker was received or processed during the epoch."
                    logger.error(msg)
                    raise ValueError(msg)

        if self.ready_to_flush_speaker_ids:
            logger.info(
                f"Epoch end: Calling `_process_and_flush_speakers_local` for {len(self.ready_to_flush_speaker_ids)} ready speakers."
            )
            self._process_and_flush_speakers_local()
        else:
            logger.info(f"Epoch end: No remaining speakers identified as ready to flush.")

        if self.speaker_data_accumulator:  # Should be empty if all went well
            msg = f"Epoch end: {len(self.speaker_data_accumulator)} speakers still in accumulator post-final flush attempt: {list(self.speaker_data_accumulator.keys())}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            f"Total items discarded on this rank due to no suitable context found (failed SSIM or duration): {self.total_discarded_no_suitable_context_this_rank}"
        )
        logger.info(f"Total items written to manifest on this rank: {self.total_items_written_this_rank}")

        # Verification step
        expected_total_processed = (
            self.total_items_written_this_rank + self.total_discarded_no_suitable_context_this_rank
        )
        if self.initial_assigned_count == expected_total_processed:
            logger.info(
                f"Verification successful: Initial items ({self.initial_assigned_count}) == Written ({self.total_items_written_this_rank}) + Discarded ({self.total_discarded_no_suitable_context_this_rank})"
            )
        else:
            msg = f"VERIFICATION FAILED: Initial items ({self.initial_assigned_count}) != Written ({self.total_items_written_this_rank}) + Discarded ({self.total_discarded_no_suitable_context_this_rank}) --- Difference: {self.initial_assigned_count - expected_total_processed}"
            logger.error(msg)
            raise RuntimeError(msg)

        if self.output_manifest_file:
            self.output_manifest_file.close()
            self.output_manifest_file = None
        logger.info(f"Local processing complete. Partial manifest closed.")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # Wait for all ranks to finish writing files


def _parse_speaker_id_libritts(record):
    """
    libritts format: audio_filepath = "{subset}/{speaker_id}/{chapter_id}/{speaker_id}_{chapter_id}_{utterance_id}_{segment_id}.wav"
        e.g. "train-clean-100/89/218/89_218_000014_000003.wav"
    re-organized speaker_id: "{subset}_{speaker_id}_{chapter_id}"
        e.g. "train-clean-100_89_218"
    """
    parts = record['audio_filepath'].lower().split('/')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"


def _parse_speaker_id_hifitts(record):
    """
    hifitts format: audio_filepath = "{speaker_id}_{audio_quality}/{book_id}/{chapter_name}_{segment_id}.wav"
        e.g. "11614_other/12352/prideofjennico_01_castle_0000.flac"
    re-organized speaker_id: "{speaker_id}_{audio_quality}_{book_id}_{chapter_name}"
        e.g. "11614_other_12352_prideofjennico_01_castle"
    """
    parts = record['audio_filepath'].lower().split('/')
    chapter_name = parts[-1].rsplit('_', 1)[0]
    return f"{parts[0]}_{parts[1]}_{chapter_name}"


def _parse_speaker_id_hifitts2(record):
    """
    hifitts2 format: audio_filepath = "{speaker_id}/{book_id}/{speaker_id}_{book_id}_{chapter_name}_{segment_id}.wav"
        e.g. "100/2315/100_2315_sea_fairies_0812_librivox-01_baum_sea_fairies_0.flac"
    re-organized speaker_id: "{speaker_id}_{book_id}_{chapter_name}"
        e.g. "100_2315_sea_fairies_0812_librivox-01_baum_sea_fairies"
    """
    parts = record['audio_filepath'].lower().split('/')
    return parts[-1].rsplit('_', 1)[0]


def _parse_speaker_id_nvyt2505(record):
    """
    nvyt2505 format: audio_filepath = "NVYT_40K_audios_wav/{utterance_id}.wav", which does not contain speaker_id.
        e.g. "NVYT_40K_audios_wav/Thg50o7gmsk.wav"
    But we can parse the speaker_id from: speaker = "| Language:en Dataset:NVYT_2505 Speaker:Thg50o7gmsk_SPEAKER_00 |".
    re-organized speaker_id: "{parsed_speaker_id}"
        e.g. "thg50o7gmsk_speaker_00"
    """
    speaker_regex = re.compile(r'Speaker:([^ |]+)')
    match = speaker_regex.search(record['speaker'])
    if not match:
        raise ValueError(f"Failed to parse speaker_id from record: {record}")
    return match.group(1).lower()


def _parse_speaker_id_rivaLindyRodney(record):
    """
    rivaLindyRodney format: audio_filepath = "{speaker}/44khz/{emotion}/{speaker}_{emotion}_{utterance_id}.wav"
        e.g. "Lindy/44khz/WIZWIKI/LINDY_WIZWIKI_004161.wav"
    re-organized speaker_id: "{speaker}_{emotion}"
        e.g. "lindy_wizwiki"
    """
    parts = record['audio_filepath'].lower().split('/')
    return f"{parts[0]}_{parts[2]}"


def _parse_speaker_id_rivaEmmaMeganSeanTom(record):
    """
    rivaEmmaMeganSeanTom format: audio_filepath = "{speaker}/22_kHz/{speaker}_{emotion}_{utterance_id}.wav"
        e.g. "Emma/22_kHz/Emma_Sad_Intense_Correlated_00147.wav"
    re-organized speaker_id: "{speaker}_{emotion}"
        e.g. "emma_sad_intense_correlated"
    """
    parts = record['audio_filepath'].lower().split('/')
    return parts[2].rsplit('_', 1)[0]


def _parse_speaker_id_jhsdGtc20Amp20Keynote(record):
    """
    jhsdGtc20Amp20Keynote format: audio_filepath = "{keynote_event}_KEYNOTE-VOOnly-44khz-16bit-mono_{utterance_id}.wav"
        e.g. "AMP20_KEYNOTE-VOOnly-44khz-16bit-mono_12.wav"
    re-organized speaker_id: "{keynote_event}"
        e.g. "AMP20"
    """
    return record['audio_filepath'].lower().rsplit('_', 2)[0]


def _get_parsed_speaker_id_for_dataset(dataset_name_arg, record):
    """Routes to the appropriate speaker ID parsing function based on dataset_name."""
    if dataset_name_arg == "libritts":
        return _parse_speaker_id_libritts(record)
    elif dataset_name_arg == "librittsDevClean":
        return _parse_speaker_id_libritts(record)
    elif dataset_name_arg == "hifitts":
        return _parse_speaker_id_hifitts(record)
    elif dataset_name_arg == "hifitts2":
        return _parse_speaker_id_hifitts2(record)
    elif dataset_name_arg == "nvyt2505":
        return _parse_speaker_id_nvyt2505(record)
    elif dataset_name_arg == "rivaLindyRodney":
        return _parse_speaker_id_rivaLindyRodney(record)
    elif dataset_name_arg == "rivaEmmaMeganSeanTom":
        return _parse_speaker_id_rivaEmmaMeganSeanTom(record)
    elif dataset_name_arg == "jhsdGtc20Amp20Keynote":
        return _parse_speaker_id_jhsdGtc20Amp20Keynote(record)
    else:
        logger.error(
            f"Unsupported dataset_name '{dataset_name_arg}' provided. Please check the --dataset-name argument."
        )
        raise ValueError(f"Unsupported dataset_name: {dataset_name_arg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--audio-base-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save rank-specific manifests.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=[
            "libritts",
            "librittsDevClean",
            "hifitts",
            "hifitts2",
            "nvyt2505",
            "rivaLindyRodney",
            "rivaEmmaMeganSeanTom",
            "jhsdGtc20Amp20Keynote",
        ],
        help="Name of the dataset being processed. This determines the speaker ID parsing logic.",
    )
    parser.add_argument("--flush-threshold-items", type=int, default=20000)
    parser.add_argument(
        "--context-min-duration", type=float, default=3.0, help="Minimum duration for a context audio segment."
    )
    parser.add_argument(
        "--context-min-ssim", type=float, default=0.6, help="Minimum cosine similarity for a context audio segment."
    )
    parser.add_argument(
        "--max-speaker-items",
        type=int,
        default=None,
        help="Maximum size of context pool per speaker to prevent OOM. If a speaker has more items, a random sample will be used as context pool, but all items will still be processed. Default: None (no limit, potential OOM risk).",
    )
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="speaker_similarity_sharded")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Initialize DDP early to get rank and world_size for sharding
    # PyTorch Lightning Trainer will handle DDP initialization if not done explicitly,
    # but we need rank/world_size for data sharding before Trainer setup.
    ddp_env_vars_detected = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ

    user_intended_distributed = False
    if isinstance(args.devices, int) and args.devices not in [0, 1]:  # 0 for CPU, 1 for single GPU. -1 means all GPUs.
        user_intended_distributed = True
    if args.num_nodes > 1:
        user_intended_distributed = True

    if user_intended_distributed and not ddp_env_vars_detected:
        logger.warning(
            f"Warning: A distributed run seems intended (num_nodes={args.num_nodes}, devices='{args.devices}'), "
            f"but standard DDP environment variables (e.g., `LOCAL_RANK`, `WORLD_SIZE`) were not detected pre-Trainer initialization. "
            f"If launching on SLURM, ensure you are using `srun` or have correctly configured your sbatch script. "
            f"For local multi-GPU, consider using `torchrun`. "
            f"PyTorch Lightning will now attempt to initialize the distributed environment. "
            f"If it defaults to a single process, data sharding will be ineffective (all data processed by one rank)."
        )

    strategy = (
        DDPStrategy(find_unused_parameters=False)
        if (isinstance(args.devices, int) and args.devices != 1 and args.devices != 0)
        else "auto"
    )

    trainer = Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        accelerator="gpu",
        strategy=strategy,
        logger=None,
        max_epochs=1,
        use_distributed_sampler=False,
    )

    world_size = trainer.world_size
    global_rank = trainer.global_rank

    log_format = f"%(asctime)s [RANK {global_rank}] [%(levelname)s] %(message)s"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    if global_rank == 0:
        logger.info("Reading and sharding manifest ...")

        temp_sv_model_for_config = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            'titanet_large', map_location=torch.device('cpu')
        )
        # Initialize sample_rate for all ranks; rank 0 will populate it.
        # This variable will be broadcast if in distributed mode.
        sample_rate = temp_sv_model_for_config.preprocessor._sample_rate
        min_duration_in_sec_required = temp_sv_model_for_config.preprocessor.featurizer.hop_length * 2 / sample_rate
        del temp_sv_model_for_config
        logger.info(
            f"Calculated sample_rate: {sample_rate}, min_duration_in_sec_required: {min_duration_in_sec_required:.3f}s"
        )

        speaker_to_records = defaultdict(list)
        num_processed_records = 0
        total_initial_records = 0

        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                total_initial_records += 1
                try:
                    rec = json.loads(line.strip())
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line: `{line.strip()}`")
                    continue

                # 1. Apply duration filter
                if rec.get("duration") is None or rec.get("duration") < min_duration_in_sec_required:
                    continue

                # 2. Apply speaker format check
                if not check_speaker_format(rec["speaker"]):
                    msg = f"Invalid speaker format for record: {rec['speaker']}, File: {rec['audio_filepath']}(offset={rec['offset']}, duration={rec['duration']})."
                    logger.error(msg)
                    raise ValueError(msg)

                # 3. Parse speaker ID and add to map
                rec['parsed_speaker_id'] = _get_parsed_speaker_id_for_dataset(args.dataset_name, rec)

                speaker_to_records[rec['parsed_speaker_id']].append(rec)
                num_processed_records += 1

        num_filtered_out_initial_pass = total_initial_records - num_processed_records

        logger.info(
            f"Initial pass filtered out {num_filtered_out_initial_pass} records (e.g., duration). Processing {num_processed_records} records before speaker count filter."
        )

        # Filter speakers with less than 2 segments
        speakers_before_count_filter = len(speaker_to_records)

        speakers_with_segment_counts = [
            {"count": len(rec_list), "records": rec_list}
            for _, rec_list in speaker_to_records.items()
            if len(rec_list) >= 2
        ]
        del speaker_to_records

        speakers_after_count_filter = len(speakers_with_segment_counts)
        records_after_count_filter = sum(item["count"] for item in speakers_with_segment_counts)

        num_speakers_filtered_by_count = speakers_before_count_filter - speakers_after_count_filter
        num_records_filtered_by_speaker_count = num_processed_records - records_after_count_filter

        logger.info(
            f"Filtered out {num_speakers_filtered_by_count} speakers (and {num_records_filtered_by_speaker_count} corresponding records) with < 2 segments. "
            f"Now processing {records_after_count_filter} records from {speakers_after_count_filter} speakers for sharding."
        )

        # Greedy Bin-Packing for speaker distribution
        # 1. Sort speakers by segment count in descending order
        speakers_with_segment_counts.sort(key=lambda x: x["count"], reverse=True)

        # 2. Initialize rank loads and assignments
        rank_loads = [0] * world_size
        rank_assignments = [[] for _ in range(world_size)]

        # 3. Assign speakers to ranks using greedy approach
        for speaker_info in speakers_with_segment_counts:
            # Find the rank with the minimum current load
            min_load_rank_idx = rank_loads.index(min(rank_loads))

            # Assign all records of this speaker to that rank
            rank_assignments[min_load_rank_idx].extend(speaker_info["records"])
            # Update the load of that rank
            rank_loads[min_load_rank_idx] += speaker_info["count"]

        data_to_distribute = rank_assignments
        logger.info(
            f"Sharding complete. {sum(len(r) for r in data_to_distribute)} records distributed among {world_size} ranks."
        )
        for r, recs in enumerate(data_to_distribute):
            logger.info(f"Plan for rank {r} = {len(recs)} records.")

        per_rank_speaker_counts = []  # [{"spk_0": 10, "spk_1": 5}, {"spk_2": 6, "spk_3": 8}, ...]
        for rank_idx in range(world_size):
            counts_for_rank = defaultdict(int)
            for record in data_to_distribute[rank_idx]:
                counts_for_rank[record['parsed_speaker_id']] += 1
            per_rank_speaker_counts.append(dict(counts_for_rank))

    else:  # Other ranks prepare to receive
        data_to_distribute = [None] * world_size
        per_rank_speaker_counts = [None] * world_size
        sample_rate = None  # Initialize for non-rank 0 before broadcast

    # Broadcast the list of lists of records. Each rank will then pick its part.
    if world_size > 1 and not torch.distributed.is_initialized():
        logger.warning(
            f"Distributed run (world_size={world_size}) detected, but `torch.distributed` not yet initialized. "
            f"Attempting to trigger environment setup via `trainer.strategy.setup_environment()`."
        )
        # The trainer's strategy is responsible for setting up the distributed environment.
        # This typically happens implicitly during trainer.fit/predict/test/validate calls.
        trainer.strategy.setup_environment()
        if torch.distributed.is_initialized():
            logger.info(
                f"`torch.distributed` successfully initialized after `trainer.strategy.setup_environment()`. Synchronizing ranks."
            )
            torch.distributed.barrier()  # Ensure all ranks have completed setup before proceeding.
        else:
            msg = f"[Rank {global_rank}] FATAL: Failed to initialize `torch.distributed` even after calling `trainer.strategy.setup_environment()` for world_size={world_size}. Cannot proceed with distributed data sharding."
            logger.error(msg)
            raise RuntimeError(msg)
    elif world_size == 1 and torch.distributed.is_initialized():
        # This case should ideally not happen (DDP initialized for a single process run by Lightning).
        logger.warning(f"Warning: `torch.distributed` is initialized, but world_size is 1. This is unusual.")
    elif world_size > 1 and torch.distributed.is_initialized():
        logger.info(f"`torch.distributed` was already initialized. world_size={world_size}. Synchronizing ranks.")
        torch.distributed.barrier()

    # Now, proceed with the data distribution logic, expecting `torch.distributed` to be initialized if world_size > 1.
    my_speaker_expected_counts = {}
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(data_to_distribute, src=0)
        assigned_records_for_this_rank = data_to_distribute[global_rank]
        torch.distributed.broadcast_object_list(per_rank_speaker_counts, src=0)
        my_speaker_expected_counts = per_rank_speaker_counts[global_rank]

        # Broadcast sample_rate
        if global_rank == 0:
            sample_rate_to_broadcast = [sample_rate]
        else:
            sample_rate_to_broadcast = [None]
        torch.distributed.broadcast_object_list(sample_rate_to_broadcast, src=0)
        sample_rate = sample_rate_to_broadcast[0]
        logger.info(f"Received {len(assigned_records_for_this_rank)} records for processing.")
        logger.debug(f"Expected speaker counts for this rank: {my_speaker_expected_counts}")
        logger.info(f"Received sample_rate via broadcast: {sample_rate}")
    elif world_size == 1:
        # data_to_distribute is already prepared by rank 0 code block if world_size was 1 from start
        assigned_records_for_this_rank = data_to_distribute[0] if data_to_distribute and data_to_distribute[0] else []
        my_speaker_expected_counts = (
            per_rank_speaker_counts[0] if per_rank_speaker_counts and per_rank_speaker_counts[0] else {}
        )
        if not assigned_records_for_this_rank:
            msg = f"[Rank {global_rank}] Error: No records were assigned for processing in single process mode. Issue in initial data prep."
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Single process, assigned {len(assigned_records_for_this_rank)} records.")
        logger.debug(f"Expected speaker counts: {my_speaker_expected_counts}")
        logger.info(f"Using sample_rate from rank 0 execution: {sample_rate}")
    else:
        msg = f"[Rank {global_rank}] Critical: DDP not initialized for sharding, and not a single process run. Cannot determine configuration."
        logger.error(msg)
        raise ValueError(msg)

    # Validate that sample_rate is now available on all ranks before use
    if sample_rate is None:
        msg = f"[Rank {global_rank}] Critical error: sample_rate was not correctly set or broadcasted. Value is None."
        logger.error(msg)
        raise RuntimeError(msg)

    wandb_logger = None
    if args.wandb_entity and args.wandb_project and global_rank == 0:
        run_name = args.wandb_name or f"sharded_similarity_{Path(args.manifest).stem}"
        wandb_logger = WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity, name=run_name, log_model=False
        )
        logger.info(f"Wandb logging enabled to {args.wandb_entity}/{args.wandb_project}, run name: {run_name}")
    trainer.logger = wandb_logger

    dataset = SpeakerShardedAudioDataset(
        assigned_records_list=assigned_records_for_this_rank,
        base_audio_dir=args.audio_base_dir,
        sample_rate=sample_rate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    model = EmbeddingSimilarityExtractorSharded(
        output_dir=args.output_dir,
        output_file_prefix=Path(args.manifest).stem,
        flush_threshold_items=args.flush_threshold_items,
        context_min_duration=args.context_min_duration,
        context_min_ssim=args.context_min_ssim,
        speaker_expected_counts_map=my_speaker_expected_counts,
        initial_assigned_count=len(assigned_records_for_this_rank),
        max_speaker_items=args.max_speaker_items,
    )
    logger.info(
        f"Starting prediction with {len(assigned_records_for_this_rank)} records ({len(my_speaker_expected_counts)} unique speakers for this rank according to counts)."
    )
    trainer.predict(model, dataloaders=dataloader)

    # Rank 0 merges the partial manifests
    if global_rank == 0:
        final_manifest_path = Path(args.output_dir) / (
            Path(args.manifest).stem
            + f"_withContextAudioMinDur{args.context_min_duration}MinSSIM{args.context_min_ssim}.json"
        )
        logger.info(f"Merging partial manifest files to `{final_manifest_path}`...")
        with open(final_manifest_path, "w", encoding="utf-8") as final_out_f:
            for i in range(world_size):
                partial_file_path = Path(args.output_dir) / f"{Path(args.manifest).stem}_rank{i}.json"
                if partial_file_path.exists():
                    with open(partial_file_path, "r", encoding="utf-8") as pf:
                        for line in pf:
                            final_out_f.write(line)
                    logger.info(f"Merged `{partial_file_path}`")
                else:
                    logger.warning(f"Warning - partial manifest file not found: `{partial_file_path}`")
        logger.info(f"Merging complete. Final manifest: `{final_manifest_path}`")

    if wandb_logger and global_rank == 0:
        wandb.finish()
        logger.info("WandB run finished.")

    logger.info(f"Done.")


if __name__ == "__main__":
    main()
