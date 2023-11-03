# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from omegaconf import DictConfig

from nemo.collections.asr.data.huggingface.hf_audio_to_text import (
    HFAudioToBPEDataset,
    HFAudioToCharDataset,
    HFIterableAudioToBPEDataset,
    HFIterableAudioToCharDataset,
)


def get_hf_audio_to_text_bpe_dataset(
    config: DictConfig, global_rank: int, world_size: int, tokenizer, augmentor=None,
):
    if "streaming" in config and config["streaming"]:
        dataset = HFIterableAudioToBPEDataset(
            audio_key=config.get('audio_key', 'audio.array'),
            text_key=config["text_key"],
            sample_rate_key=config.get('sample_rate_key', 'audio.sampling_rate'),
            tokenizer=tokenizer,
            hf_data_cfg=config["hf_data_cfg"],
            sample_rate=config["sample_rate"],
            augmentor=augmentor,
            trim=config.get('trim_silence', False),
            return_sample_id=config.get('return_sample_id', False),
            id_key=config.get("id_key", None),
            channel_selector=config.get('channel_selector', None),
            normalize_db=config.get('normalize_db', None),
            ref_channel=config.get('ref_channel', None),
            global_rank=global_rank,
            world_size=world_size,
            shuffle_n=config.get("shuffle_n", 2048),
            shuffle_seed=config.get("shuffle_seed", None),
            use_start_end_token=config.get('use_start_end_token', True),
            normalize_text=config.get('normalize_text', False),
            symbols_to_keep=config.get('symbols_to_keep', None),
        )
    else:
        dataset = HFAudioToBPEDataset(
            audio_key=config.get('audio_key', 'audio.array'),
            text_key=config["text_key"],
            sample_rate_key=config.get('sample_rate_key', 'audio.sampling_rate'),
            tokenizer=tokenizer,
            hf_data_cfg=config["hf_data_cfg"],
            sample_rate=config["sample_rate"],
            augmentor=augmentor,
            trim=config.get('trim_silence', False),
            return_sample_id=config.get('return_sample_id', False),
            id_key=config.get("id_key", None),
            channel_selector=config.get('channel_selector', None),
            normalize_db=config.get('normalize_db', None),
            ref_channel=config.get('ref_channel', None),
            use_start_end_token=config.get('use_start_end_token', True),
            normalize_text=config.get('normalize_text', False),
            symbols_to_keep=config.get('symbols_to_keep', None),
        )

    return dataset


def get_hf_audio_to_text_char_dataset(
    config: DictConfig, global_rank: int, world_size: int, augmentor=None,
):
    if "streaming" in config and config["streaming"]:
        dataset = HFIterableAudioToCharDataset(
            labels=config["labels"],
            audio_key=config.get('audio_key', 'audio.array'),
            text_key=config["text_key"],
            sample_rate_key=config.get('sample_rate_key', 'audio.sampling_rate'),
            hf_data_cfg=config["hf_data_cfg"],
            sample_rate=config["sample_rate"],
            augmentor=augmentor,
            trim=config.get('trim_silence', False),
            return_sample_id=config.get('return_sample_id', False),
            id_key=config.get("id_key", None),
            channel_selector=config.get('channel_selector', None),
            normalize_db=config.get('normalize_db', None),
            ref_channel=config.get('ref_channel', None),
            global_rank=global_rank,
            world_size=world_size,
            shuffle_n=config.get("shuffle_n", 2048),
            shuffle_seed=config.get("shuffle_seed", None),
            parser=config.get("parser", "en"),
            blank_index=config.get("blank_index", -1),
            unk_index=config.get("unk_index", -1),
            normalize=config.get("normalize", False),
            normalize_text=config.get('normalize_text', False),
            symbols_to_keep=config.get('symbols_to_keep', None),
            pad_id=config.get('pad_id', 0),
            bos_id=config.get('bos_id', None),
            eos_id=config.get('eos_id', None),
        )
    else:
        dataset = HFAudioToCharDataset(
            labels=config["labels"],
            audio_key=config.get('audio_key', 'audio.array'),
            text_key=config["text_key"],
            sample_rate_key=config.get('sample_rate_key', 'audio.sampling_rate'),
            hf_data_cfg=config["hf_data_cfg"],
            sample_rate=config["sample_rate"],
            augmentor=augmentor,
            trim=config.get('trim_silence', False),
            bos_id=config.get('bos_id', None),
            eos_id=config.get('eos_id', None),
            pad_id=config.get('pad_id', 0),
            return_sample_id=config.get('return_sample_id', False),
            id_key=config.get("id_key", None),
            channel_selector=config.get('channel_selector', None),
            normalize_db=config.get('normalize_db', None),
            ref_channel=config.get('ref_channel', None),
            parser=config.get("parser", "en"),
            blank_index=config.get("blank_index", -1),
            unk_index=config.get("unk_index", -1),
            normalize=config.get("normalize", False),
            normalize_text=config.get('normalize_text', False),
            symbols_to_keep=config.get('symbols_to_keep', None),
        )

    return dataset
