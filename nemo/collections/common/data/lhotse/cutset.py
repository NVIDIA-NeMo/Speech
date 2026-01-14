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

import io
import logging
import random
import re
import warnings
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import KeysView, Mapping, Sequence, Tuple, Union

import numpy as np
import omegaconf
import soundfile as sf
from lhotse import CutSet, Features, MonoCut, Recording, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.cut import Cut, MixedCut, PaddingCut
from lhotse.serialization import load_yaml
from lhotse.utils import fastcopy
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.common.data.lhotse.nemo_adapters import (
    LazyNeMoIterator,
    LazyNeMoTarredIterator,
    expand_sharded_filepaths,
)
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    LhotseTextAdapter,
    LhotseTextPairAdapter,
    NeMoMultimodalConversation,
    NeMoMultimodalConversationJsonlAdapter,
    NeMoMultimodalConversationShareGPTJsonlAdapter,
    NeMoSFTJsonlAdapter,
    TextTurn,
)
from nemo.collections.common.parts.preprocessing.manifest import get_full_path


def read_cutset_from_config(config: Union[DictConfig, dict]) -> Tuple[CutSet, bool]:
    """
    Reads NeMo configuration and creates a CutSet either from Lhotse or NeMo manifests.

    Returns a tuple of ``CutSet`` and a boolean indicating whether the data is tarred (True) or not (False).
    """
    # First, check if the dataset is specified in the new configuration format and use it if possible.
    if not isinstance(config, DictConfig):
        config = DictConfig(config)
    if config.get("input_cfg") is not None:
        cuts, is_tarred = read_dataset_config(config)
    else:
        # Now, we'll figure out if we should read Lhotse manifest or NeMo manifest.
        use_nemo_manifest = all(config.get(opt) is None for opt in ("cuts_path", "shar_path"))
        if use_nemo_manifest:
            if config.get("manifest_filepath") is None:
                raise IncompleteConfigError("You must specify either: manifest_filepath, cuts_path, or shar_path")
            cuts, is_tarred = read_nemo_manifest(config)
        else:
            cuts, is_tarred = read_lhotse_manifest(config)

    return cuts, is_tarred


class IncompleteConfigError(RuntimeError):
    """Placeholder for an error raised."""

    pass


KNOWN_DATA_CONFIG_TYPES = {}


def get_known_config_data_types() -> KeysView[str]:
    """
    Return the names of all registered data type parsers.

    Example:

        >>> get_known_config_data_types()
        ["nemo", "nemo_tarred", "lhotse", ...]
    """
    return KNOWN_DATA_CONFIG_TYPES.keys()


def get_parser_fn(data_type_name: str):
    """
    Return the parsing function for a given data type name.
    Parsing function reads a dataloading config and returns a tuple
    of lhotse ``CutSet`` and boolean indicating whether we should use
    iterable dataset (True) or map dataset (False) mechanism ("is tarred").
    """
    return KNOWN_DATA_CONFIG_TYPES[data_type_name]


def data_type_parser(name: Union[str, list[str]]):
    """
    Decorator used to register data type parser functions.
    Parsing function reads a dataloading config and returns a tuple
    of lhotse ``CutSet`` and boolean indicating whether we should use
    iterable dataset (True) or map dataset (False) mechanism ("is tarred").

    Example:

        >>> @data_type_parser("my_new_format")
        ... def my_new_format(config):
        ...     return CutSet(read_my_format(**config)), True
        ...
        ... fn = get_parser_fn("my_new_format")
        ... cuts, is_tarred = fn({"my_arg_0": ..., "my_arg_1": ..., ...})
    """

    def _decorator(fn):
        global KNOWN_DATA_CONFIG_TYPES
        if isinstance(name, str):
            KNOWN_DATA_CONFIG_TYPES[name] = fn
        else:
            for n in name:
                KNOWN_DATA_CONFIG_TYPES[n] = fn
        return fn

    return _decorator


def read_dataset_config(config) -> tuple[CutSet, bool]:
    """
    Input configuration format examples.
    Example 1. Combine two datasets with equal weights and attach custom metadata in ``tags`` to each cut::
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /path/to/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.5
            tags:
              lang: en
              some_metadata: some_value
          - type: nemo_tarred
            manifest_filepath: /path/to/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.5
            tags:
              lang: pl
              some_metadata: some_value
    Example 2. Combine multiple (4) datasets, with 2 corresponding to different tasks (ASR, AST).
        There are two levels of weights: per task (outer) and per dataset (inner).
        The final weight is the product of outer and inner weight::
        input_cfg:
          - type: group
            weight: 0.7
            tags:
              task: asr
            input_cfg:
              - type: nemo_tarred
                manifest_filepath: /path/to/asr1/manifest__OP_0..512_CL_.json
                tarred_audio_filepath: /path/to/tarred_audio/asr1/audio__OP_0..512_CL_.tar
                weight: 0.6
                tags:
                  lang: en
                  some_metadata: some_value
              - type: nemo_tarred
                manifest_filepath: /path/to/asr2/manifest__OP_0..512_CL_.json
                tarred_audio_filepath: /path/to/asr2/tarred_audio/audio__OP_0..512_CL_.tar
                weight: 0.4
                tags:
                  lang: pl
                  some_metadata: some_value
          - type: group
            weight: 0.3
            tags:
              task: ast
            input_cfg:
              - type: nemo_tarred
                manifest_filepath: /path/to/ast1/manifest__OP_0..512_CL_.json
                tarred_audio_filepath: /path/to/ast1/tarred_audio/audio__OP_0..512_CL_.tar
                weight: 0.2
                tags:
                  src_lang: en
                  tgt_lang: pl
              - type: nemo_tarred
                manifest_filepath: /path/to/ast2/manifest__OP_0..512_CL_.json
                tarred_audio_filepath: /path/to/ast2/tarred_audio/audio__OP_0..512_CL_.tar
                weight: 0.8
                tags:
                  src_lang: pl
                  tgt_lang: en
    """
    propagate_attrs = {
        "shuffle": config.get("shuffle", False),
        "shard_seed": config.get("shard_seed", "trng"),
        "text_field": config.get("text_field", "text"),
        "lang_field": config.get("lang_field", "lang"),
        "metadata_only": config.get("metadata_only", False),
        "force_finite": config.get("force_finite", False),
        "max_open_streams": config.get("max_open_streams", None),
        "audio_locator_tag": config.get("audio_locator_tag", None),
        "token_equivalent_duration": config.get("token_equivalent_duration", None),
        "skip_missing_manifest_entries": config.get("skip_missing_manifest_entries", False),
        "force_map_dataset": config.get("force_map_dataset", False),
        "force_iterable_dataset": config.get("force_iterable_dataset", False),
        "slice_length": config.get("slice_length", None),
    }
    cuts, is_tarred = parse_and_combine_datasets(config.input_cfg, propagate_attrs=propagate_attrs)
    return cuts, is_tarred


def parse_group(grp_cfg: DictConfig, propagate_attrs: dict) -> [CutSet, bool]:
    """Parse a group configuration, potentially combining multiple datasets."""
    assert grp_cfg.type in get_known_config_data_types(), f"Unknown item type in dataset config list: {grp_cfg.type=}"

    # Note: Text data types will return is_tarred=True.
    #       We choose to treat text as-if it was tarred, which tends to be more
    #       efficient as it moves the text file iteration into dataloading subprocess.
    if grp_cfg.type != "group":
        parser_fn = get_parser_fn(grp_cfg.type)
        cuts, is_tarred = parser_fn(grp_cfg)
    else:
        cuts, is_tarred = parse_and_combine_datasets(
            grp_cfg.input_cfg,
            propagate_attrs=propagate_attrs,
        )
    # Attach extra tags to every utterance dynamically, if provided.
    if (extra_tags := grp_cfg.get("tags")) is not None:
        cuts = cuts.map(partial(attach_tags, tags=extra_tags), apply_fn=None)
    return cuts, is_tarred


@data_type_parser("txt")
def read_txt_paths(config: DictConfig) -> tuple[CutSet, bool]:
    """Read paths to text files and create a CutSet."""
    cuts = CutSet(
        LhotseTextAdapter(
            paths=config.paths,
            language=config.language,
            shuffle_shards=config.shuffle,
            shard_seed=config.shard_seed,
        )
    )
    if not config.get("force_finite", False):
        cuts = cuts.repeat(preserve_id=True)
    return cuts, True


@data_type_parser("txt_pair")
def read_txt_pair_paths(config: DictConfig) -> tuple[CutSet, bool]:
    """Read paths to source and target text files and create a CutSet."""
    cuts = CutSet(
        LhotseTextPairAdapter(
            source_paths=config.source_paths,
            target_paths=config.target_paths,
            source_language=config.get("source_language"),
            target_language=config.get("target_language"),
            questions_path=config.get("questions_path"),
            questions_language=config.get("questions_language"),
            shuffle_shards=config.shuffle,
            shard_seed=config.shard_seed,
        )
    )
    if not config.get("force_finite", False):
        cuts = cuts.repeat(preserve_id=True)
    return cuts, True


@data_type_parser("nemo_sft_jsonl")
def read_nemo_sft_jsonl(config: DictConfig) -> tuple[CutSet, bool]:
    """Read paths to Nemo SFT JSONL files and create a CutSet."""
    cuts = CutSet(
        NeMoSFTJsonlAdapter(
            paths=config.paths,
            language=config.get("language"),
            shuffle_shards=config.shuffle,
            shard_seed=config.shard_seed,
        )
    )
    if not config.get("force_finite", False):
        cuts = cuts.repeat(preserve_id=True)
    return cuts, True


@data_type_parser("multimodal_conversation")
def read_multimodal_conversation_jsonl(config: DictConfig) -> tuple[CutSet, bool]:
    """Read paths to multimodal conversation JSONL files and create a CutSet."""
    cuts = CutSet(
        NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=config.manifest_filepath,
            tarred_audio_filepaths=config.get("tarred_audio_filepaths"),
            audio_locator_tag=config.audio_locator_tag,
            token_equivalent_duration=config.get("token_equivalent_duration"),
            shuffle_shards=config.shuffle,
            shard_seed=config.shard_seed,
            system_prompt=config.get("tags", {}).get("system_prompt"),
            slice_length=config.get("slice_length"),
        )
    )
    if not config.get("force_finite", False):
        cuts = cuts.repeat(preserve_id=True)
    return cuts, True


@data_type_parser(["share_gpt"])
def read_share_gpt_as_conversation(config) -> tuple[CutSet, bool]:
    """Read paths to ShareGPT JSONL files and create a CutSet of NeMoMultimodalConversation."""
    cuts = CutSet(
        NeMoMultimodalConversationShareGPTJsonlAdapter(
            manifest_filepath=config.manifest_filepath,
            tarred_audio_filepaths=config.get("tarred_audio_filepaths"),
            audio_locator_tag=config.audio_locator_tag,
            audio_placeholders=config.audio_placeholders,
            token_equivalent_duration=config.get("token_equivalent_duration"),
            shuffle_shards=config.shuffle,
            shard_seed=config.shard_seed,
            slice_length=config.get("slice_length"),
        )
    )
    if not config.get("force_finite", False):
        cuts = cuts.repeat(preserve_id=True)
    return cuts, True


def _resolve_shar_inputs(path: Union[str, Path], only_metadata: bool) -> dict:
    if only_metadata:
        return dict(fields={"cuts": sorted(Path(path).glob("cuts.*"))})
    else:
        return dict(in_dir=path)


def attach_tags(cut, tags: dict):
    """Attach extra tags to a cut dynamically."""
    for key, val in tags.items():
        setattr(cut, key, val)
    return cut


@data_type_parser("group")
def parse_and_combine_datasets(
    config_list: Union[list[DictConfig], ListConfig], propagate_attrs: dict
) -> tuple[CutSet, bool]:
    """Parse a list of dataset configurations, potentially combining multiple datasets."""
    cuts = []
    weights = []
    tarred_status = []

    if isinstance(config_list, (str, Path)):
        # Resolve local filepath /path/to/input_cfg.yaml or remote url s3://bucket/path/to/input_cfg.yaml into config contents if needed.
        config_list = OmegaConf.create(load_yaml(config_list))
    assert len(config_list) > 0, "Empty group in dataset config list."

    for item in config_list:
        # Check if we have any attributes that are propagated downwards to each item in the group.
        # If a key already exists in the item, it takes precedence (we will not overwrite);
        # otherwise we will assign it.
        # We also update propagate_atts for the next sub-groups based on what's present in this group
        next_propagate_attrs = propagate_attrs.copy()
        for k, v in propagate_attrs.items():
            if k not in item:
                item[k] = v
            else:
                next_propagate_attrs[k] = item[k]

        # Load the item (which may also be another group) as a CutSet.
        item_cuts, item_is_tarred = parse_group(item, next_propagate_attrs)
        cuts.append(item_cuts)
        tarred_status.append(item_is_tarred)
        if (w := item.get("weight")) is not None:
            weights.append(w)

    all_same_tarred_status = all(t == tarred_status[0] for t in tarred_status)
    if not all_same_tarred_status:
        if propagate_attrs["force_map_dataset"] or propagate_attrs["force_iterable_dataset"]:
            logging.warning(
                f"Not all datasets in the group have the same tarred status, using provided force_map_dataset "
                f"({propagate_attrs['force_map_dataset']}) and force_iterable_dataset "
                f"({propagate_attrs['force_iterable_dataset']}) to determine the final tarred status."
            )
        else:
            raise ValueError(
                "Mixing tarred and non-tarred datasets is not supported when neither force_map_dataset "
                "nor force_iterable_dataset is True."
            )

    assert len(weights) == 0 or len(cuts) == len(
        weights
    ), "Missing dataset weight. When weighting datasets, every dataset must have a specified weight."

    if len(cuts) > 1:
        cuts = mux(
            *cuts,
            weights=weights if weights else None,
            max_open_streams=propagate_attrs["max_open_streams"],
            seed=propagate_attrs["shard_seed"],
            force_finite=propagate_attrs["force_finite"] or propagate_attrs["metadata_only"],
        )
    else:
        (cuts,) = cuts
    return cuts, tarred_status[0]


@data_type_parser(["lhotse", "lhotse_shar"])
def read_lhotse_manifest(config) -> tuple[CutSet, bool]:
    """Read paths to Lhotse manifest files and create a CutSet."""
    is_tarred = config.get("shar_path") is not None
    if is_tarred:
        # Lhotse Shar is the equivalent of NeMo's native "tarred" dataset.
        # The combination of shuffle_shards, and repeat causes this to
        # be an infinite manifest that is internally reshuffled on each epoch.
        # The parameter ``config.shard_seed`` is used to determine shard shuffling order. Options:
        # - "trng" means we'll defer setting the seed until the iteration
        #   is triggered, and we'll use system TRNG to get a completely random seed for each worker.
        #   This results in every dataloading worker using full data but in a completely different order.
        # - "randomized" means we'll defer setting the seed until the iteration
        #   is triggered, and we'll use config.seed to get a pseudo-random seed for each worker.
        #   This results in every dataloading worker using full data but in a completely different order.
        #   Unlike "trng", this is deterministic, and if you resume training, you should change the seed
        #   to observe different data examples than in the previous run.
        # - integer means we'll set a specific seed in every worker, and data would be duplicated across them.
        #   This is mostly useful for unit testing or debugging.
        shard_seed = config.get("shard_seed", "trng")
        metadata_only = config.get("metadata_only", False)
        force_finite = config.get("force_finite", False)
        if config.get("cuts_path") is not None:
            warnings.warn("Note: lhotse.cuts_path will be ignored because lhotse.shar_path was provided.")
        if isinstance(config.shar_path, (str, Path)):
            logging.info(f"Initializing Lhotse Shar CutSet (tarred) from a single data source: '{config.shar_path}'")
            cuts = CutSet.from_shar(
                **_resolve_shar_inputs(config.shar_path, metadata_only),
                shuffle_shards=True,
                seed=shard_seed,
                slice_length=config.get("slice_length", None),
            )
            if not metadata_only and not force_finite:
                cuts = cuts.repeat(preserve_id=True)
        elif isinstance(config.shar_path, Sequence):
            # Multiple datasets in Lhotse Shar format: we will dynamically multiplex them
            # with probability approximately proportional to their size
            logging.info(
                "Initializing Lhotse Shar CutSet (tarred) from multiple data sources with a weighted multiplexer. "
                "We found the following sources and weights: "
            )
            cutsets = []
            weights = []
            for item in config.shar_path:
                if isinstance(item, (str, Path)):
                    path = item
                    cs = CutSet.from_shar(
                        **_resolve_shar_inputs(path, metadata_only),
                        shuffle_shards=True,
                        seed=shard_seed,
                        slice_length=config.get("slice_length", None),
                    )
                    weight = len(cs)
                else:
                    assert isinstance(item, Sequence) and len(item) == 2 and isinstance(item[1], (int, float)), (
                        "Supported inputs types for config.shar_path are: "
                        "str | list[str] | list[tuple[str, number]] "
                        "where str is a path and number is a mixing weight (it may exceed 1.0). "
                        f"We got: '{item}'"
                    )
                    path, weight = item
                    cs = CutSet.from_shar(
                        **_resolve_shar_inputs(path, metadata_only),
                        shuffle_shards=True,
                        seed=shard_seed,
                        slice_length=config.get("slice_length", None),
                    )
                logging.info(f"- {path=} {weight=}")
                cutsets.append(cs)
                weights.append(weight)

            cuts = mux(
                *cutsets,
                weights=weights,
                max_open_streams=config.get("max_open_streams", None),
                seed=shard_seed,
                force_finite=force_finite,
            )
        elif isinstance(config.shar_path, Mapping):
            fields = {k: expand_sharded_filepaths(v) for k, v in config.shar_path.items()}
            assert "cuts" in config.shar_path.keys(), (
                f"Invalid value for key 'shar_path': a dict was provided, but didn't specify key 'cuts' pointing "
                f"to the manifests. We got the following: {config.shar_path=}"
            )
            if metadata_only:
                fields = {"cuts": fields["cuts"]}
            cuts = CutSet.from_shar(
                fields=fields,
                shuffle_shards=True,
                seed=shard_seed,
                slice_length=config.get("slice_length", None),
            )
            if not metadata_only and not force_finite:
                cuts = cuts.repeat(preserve_id=True)
        else:
            raise RuntimeError(
                f"Unexpected value for key 'shar_path'. We support string, list of strings, "
                f"list of tuples[string,float], and dict[string,list[string]], "
                f"but got: {type(config.shar_path)=} {config.shar_path=}"
            )
    else:
        # Regular Lhotse manifest points to individual audio files (like native NeMo manifest).
        path = config.cuts_path
        cuts = CutSet.from_file(path).map(partial(resolve_relative_paths, manifest_path=path))
    return cuts, is_tarred


def cut_to_conversation(
    cut: Cut, audio_locator_tag: str, token_equivalent_duration: float
) -> NeMoMultimodalConversation:
    """
    Converts a lhotse Cut into a two-turn NeMoMultimodalConversation, where the user turn contains cut's audio,
    and assistant turn contains text response from ``cut.supervisions[0].text``.

    If ``cut`` has a custom field ``context``, it's pre-pended as an extra user text turn before the user's audio turn.
    """
    if isinstance(cut, NeMoMultimodalConversation):
        return cut
    turns = [
        AudioTurn(cut=cut, role="user", audio_locator_tag=audio_locator_tag, text=cut.supervisions[0].text),
        TextTurn(value=cut.supervisions[0].text, role="assistant"),
    ]
    if hasattr(cut, "context"):
        turns = [TextTurn(value=cut.context, role="user")] + turns
    if hasattr(cut, "system_prompt"):
        turns = [TextTurn(value=cut.system_prompt, role="system")] + turns
    return NeMoMultimodalConversation(
        id=cut.id,
        turns=turns,
        token_equivalent_duration=token_equivalent_duration,
        custom=cut.custom,
    )


@data_type_parser(["s2s_duplex_overlap_as_s2s_duplex"])
def read_s2s_duplex_overlap_as_s2s_duplex(config) -> Tuple[CutSet, bool]:
    """
    Convert a CutSet with overlapping agent/user segments into a standard S2S duplex format.

    Args:
        config: Dictionary containing parser options:
            - move_agent_text_back_by (float): Time offset to shift agent text back.
            - filter_samples_starting_with_agent (bool): Whether to remove samples starting with agent.
            - agent_roles (List[str]): Roles considered as agent.

    Returns:
        Tuple[CutSet, bool]: Converted cuts and a flag indicating if the data was tarred.
    """
    move_agent_text_back_by = config.get("move_agent_text_back_by", 0)
    filter_samples_starting_with_agent = config.get("filter_samples_starting_with_agent", False)
    agent_roles = config.get("agent_roles", ["agent", "Assistant", "assistant"])

    cuts, is_tarred = read_cutset_from_config(config)

    def filter_cuts_starting_with_agent_fn(cuts: CutSet, agent_roles: Tuple[str, ...]) -> CutSet:
        """Remove cuts where the first supervision belongs to an agent role."""

        def _filter_fn(cut: Cut) -> bool:
            if not cut.supervisions:
                return False
            cut.supervisions = sorted(cut.supervisions, key=lambda s: s.start)
            return cut.supervisions[0].speaker not in agent_roles

        return cuts.filter(_filter_fn)

    def convert_overlap_cut_fn(cut: Cut) -> Cut:
        """Convert agent/user overlapping segments into sequential SupervisionSegments."""
        agent_segments = [
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=seg["start"] - move_agent_text_back_by,
                duration=seg["end"] - seg["start"] + move_agent_text_back_by,
                text=seg["text"],
                speaker="agent",
            )
            for seg in cut.agent_segments
        ]

        user_segments = [
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=seg["start"],
                duration=seg["end"] - seg["start"],
                text=seg["text"],
                speaker="user",
            )
            for seg in cut.user_segments
        ]

        cut.supervisions = sorted(agent_segments + user_segments, key=lambda s: s.start)
        cut.formatter = "s2s_duplex_overlap_as_s2s_duplex"
        return cut

    cuts = cuts.map(convert_overlap_cut_fn)
    if filter_samples_starting_with_agent:
        cuts = filter_cuts_starting_with_agent_fn(cuts, tuple(agent_roles))

    return cuts, is_tarred


@data_type_parser(["lhotse_magpietts_data_as_continuation"])
def read_lhotse_magpietts_data_as_s2s_duplex(config) -> Tuple[CutSet, bool]:
    """
    Convert MagpieTTS dataset cuts into the Duplex S2S format, with optional
    `context_audio` that can be used as a speaker reference.

    Args:
        config: Dictionary containing parser options:
            - add_extra_end_silence (bool): Whether to add extra silence at the end.
            - extra_end_silence_range (List[float]): Range of extra silence duration.
            - max_cer (float): Maximum allowed character error rate.
            - min_context_speaker_similarity (float): Minimum similarity score.
            - target_speaker (str, optional): Target speaker filter.
            - sample_rate (int): Audio sample rate for resampling.

    Returns:
        Tuple[CutSet, bool]: Converted cuts and a flag indicating if data was tarred.
    """
    cuts, is_tarred = read_cutset_from_config(config)

    add_extra_end_sil = config.get("add_extra_end_silence", False)
    extra_end_silence_range = config.get("extra_end_silence_range", [0.5, 6.0])
    sample_rate = config.get("sample_rate", 22050)

    max_cer = config.get("max_cer", 0.03)
    min_context_speaker_similarity = config.get("min_context_speaker_similarity", 0.6)
    target_speaker = config.get("target_speaker", None)
    keep_flag = "pass"

    def create_recording_from_array(samples: np.ndarray, sampling_rate: int, recording_id: str) -> Recording:
        """Convert a numpy array into a Lhotse Recording object."""
        with io.BytesIO() as buffer:
            sf.write(buffer, samples.T, samplerate=sampling_rate, format='WAV')
            buffer.seek(0)
            return Recording.from_bytes(buffer.read(), recording_id=recording_id)

    def convert_cut_fn(cut: Cut) -> Cut:
        """Convert a single cut into the continuation format."""
        orig_agent_sup = fastcopy(cut.supervisions[0])
        target_audio_orig_dur = cut.target_audio.duration

        # Resample audios
        cut.target_audio = cut.target_audio.resample(sample_rate)
        cut.context_audio = cut.context_audio.resample(sample_rate)
        total_duration = cut.target_audio.duration

        # Prepare MonoCuts
        cut_target = MonoCut(
            id=f"{cut.id}_target",
            start=0.0,
            duration=total_duration,
            channel=0,
            recording=cut.target_audio,
            supervisions=[],
        )

        zero_audio = np.zeros((1, int(total_duration * sample_rate)), dtype=np.float32)
        source_recording = create_recording_from_array(zero_audio, sample_rate, recording_id=f"{cut.id}_source")

        cut_source = MonoCut(
            id=f"{cut.id}_source",
            start=0.0,
            duration=total_duration,
            channel=0,
            recording=source_recording,
            supervisions=[],
        )

        # Save to memory
        cut_source = cut_source.move_to_memory(audio_format='wav')
        cut_target = cut_target.move_to_memory(audio_format='wav')

        # Create user and agent supervisions
        user_sup = fastcopy(orig_agent_sup, start=0.0, duration=0.08, speaker="user", text="dummy text")
        agent_sup = fastcopy(orig_agent_sup, start=0.0, duration=target_audio_orig_dur - 0.08, speaker="agent")

        # Optionally add extra silence
        if add_extra_end_sil:
            sil_duration = random.uniform(*extra_end_silence_range)
            cut_target = cut_target.pad(duration=total_duration + sil_duration, direction="right")
            cut_source = cut_source.pad(duration=total_duration + sil_duration, direction="right")
            cut_source = cut_source.to_mono().move_to_memory(audio_format='wav')
            cut_target = cut_target.to_mono().move_to_memory(audio_format='wav')
            agent_sup.duration += sil_duration + 1.0
            user_sup.duration += sil_duration

        # Assemble final cut
        cut_source.supervisions = [user_sup, agent_sup]
        cut_source.target_audio = cut_target.recording
        cut_source.duration = cut_target.duration
        cut_source.context_audio = cut.context_audio
        cut_source.formatter = "lhotse_magpietts_data_as_continuation"

        return cut_source

    # Filters
    def filter_cer_fn(cut: Cut) -> bool:
        return (
            len(cut.supervisions) == 0
            or not cut.supervisions[0].has_custom("cer")
            or cut.supervisions[0].cer <= max_cer
        )

    def filter_val_flag_fn(cut: Cut) -> bool:
        return not cut.has_custom("validation_status") or cut.validation_status == keep_flag

    def filter_secs_fn(cut: Cut) -> bool:
        return (
            len(cut.supervisions) == 0
            or not cut.supervisions[0].has_custom("context_speaker_similarity")
            or cut.supervisions[0].context_speaker_similarity >= min_context_speaker_similarity
        )

    def filter_target_speaker_fn(cut: Cut) -> bool:
        return len(cut.supervisions) == 0 or target_speaker is None or target_speaker in cut.supervisions[0].speaker

    # Apply filters
    cuts = (
        cuts.filter(filter_cer_fn).filter(filter_val_flag_fn).filter(filter_secs_fn).filter(filter_target_speaker_fn)
    )

    # Convert cuts
    cuts = cuts.map(convert_cut_fn)

    return cuts, is_tarred


@data_type_parser(["lhotse_as_conversation"])
def read_lhotse_as_conversation(config) -> tuple[CutSet, bool]:
    cuts, is_tarred = read_cutset_from_config(config)
    # Attach extra tags to every utterance dynamically, if provided.
    # We need to attach them before cuts are converted to conversations.
    if (extra_tags := config.get("tags")) is not None:
        cuts = cuts.map(partial(attach_tags, tags=extra_tags), apply_fn=None)
    cuts = cuts.map(
        partial(
            cut_to_conversation,
            audio_locator_tag=config.audio_locator_tag,
            token_equivalent_duration=config.token_equivalent_duration,
        )
    )
    return cuts, is_tarred


def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces


class FailedConversion:
    pass


def s2s_cut_to_conversation(
    cut: Cut,
    audio_locator_tag: str,
    token_equivalent_duration: float,
    input_roles: Sequence[str] = ("user", "User"),
    output_roles: Sequence[str] = ("assistant", "Assistant", "agent", "Agent"),
    strip_timestamp_tokens: bool = True,
) -> NeMoMultimodalConversation:
    """
    Converts a lhotse Cut representing multi-turn speech-to-speech conversation (with multiple supervision segments)
    into a multi-turn NeMoMultimodalConversation, where the user has AudioTurns and assistant responds in TextTurns.

    Args:
        cut: lhotse Cut to convert.
        audio_locator_tag: special token indicating audio will be inserted in this location in the token sequence.
        token_equivalent_duration: how much speech duration is counted as one token.
        input_roles: when supervision.speaker is set to one of these values, we consider it user's turn.
        output_roles: when supervision.speaker is set to one of these values, we consider it assistant's turn.
        strip_timestamp_tokens: strips tokens like <|0|>, <|1|>, etc indicating timestamps from the text.
    """
    turn_cuts = cut.trim_to_supervisions(keep_overlapping=False)
    turns = []
    idx = 0
    for per_turn_cut in turn_cuts:
        assert (
            len(per_turn_cut.supervisions) >= 1
        ), f"Expected at least one supervision per turn, got none in cut {cut.id}"
        # If len(per_turn_cut.supervisions) > 1, only the first turn is considered for cut creation
        # We assume that len(per_turn_cut.supervisions) >= 1 happens because one of the turns is completely contained within
        # another turn
        turn_speaker = per_turn_cut.supervisions[0].speaker
        turn_text = per_turn_cut.supervisions[0].text
        if strip_timestamp_tokens:
            turn_text = _strip_timestamps(turn_text)
        if len(per_turn_cut.supervisions) > 1:
            assert per_turn_cut.supervisions[1].text == turn_cuts[idx - 1].supervisions[0].text
        if turn_speaker in input_roles:
            turns.append(AudioTurn(cut=per_turn_cut, role="user", audio_locator_tag=audio_locator_tag, text=turn_text))
        elif turn_speaker in output_roles:
            turns.append(TextTurn(value=turn_text, role="assistant"))
        else:
            logging.warning(f"Speaker '{turn_speaker}' not found in user or agent roles for cut {cut.id}")
            return FailedConversion()
        idx += 1
    if hasattr(cut, "system_prompt") and all(t.role != "system" for t in turns):
        turns = [TextTurn(value=cut.system_prompt, role="system")] + turns

    return NeMoMultimodalConversation(
        id=cut.id,
        turns=turns,
        token_equivalent_duration=token_equivalent_duration,
        custom=cut.custom,
    )


@data_type_parser(["s2s_as_conversation"])
def read_s2s_as_conversation(config) -> tuple[CutSet, bool]:
    cuts, is_tarred = read_cutset_from_config(config)
    cuts = cuts.map(
        partial(
            s2s_cut_to_conversation,
            audio_locator_tag=config.audio_locator_tag,
            token_equivalent_duration=config.token_equivalent_duration,
            input_roles=config.get("input_roles", ["user", "User"]),
            output_roles=config.get("output_roles", ["assistant", "Assistant", "agent", "Agent"]),
            strip_timestamp_tokens=config.get("strip_timestamp_tokens", True),
        )
    ).filter(lambda ex: not isinstance(ex, FailedConversion))
    return cuts, is_tarred


def _resolve_shar_inputs(path: Union[str, Path], only_metadata: bool) -> dict:
    if only_metadata:
        return dict(fields={"cuts": sorted(Path(path).glob("cuts.*"))})
    else:
        return dict(in_dir=path)


def resolve_relative_paths(cut: Cut, manifest_path: str) -> Cut:
    """Resolve relative paths in a Cut object to their full paths."""
    if isinstance(cut, PaddingCut):
        return cut

    if isinstance(cut, MixedCut):
        for track in cut.tracks:
            track.cut = resolve_relative_paths(track.cut, manifest_path)
        return cut

    def resolve_recording(value):
        for audio_source in value.sources:
            if audio_source.type == "file":
                audio_source.source = get_full_path(audio_source.source, manifest_file=manifest_path)

    def resolve_array(value):
        if isinstance(value, TemporalArray):
            value.array = resolve_array(value.array)
        else:
            if value.storage_type in ("numpy_files", "lilcom_files"):
                abspath = Path(
                    get_full_path(str(Path(value.storage_path) / value.storage_key), manifest_file=manifest_path)
                )
                value.storage_path = str(abspath.parent)
                value.storage_key = str(abspath.name)
            elif value.storage_type in (
                "kaldiio",
                "chunked_lilcom_hdf5",
                "lilcom_chunky",
                "lilcom_hdf5",
                "numpy_hdf5",
            ):
                value.storage_path = get_full_path(value.storage_path, manifest_file=manifest_path)
            # ignore others i.e. url, in-memory data, etc.

    if cut.has_recording:
        resolve_recording(cut.recording)
    if cut.has_features:
        resolve_array(cut.features)
    if cut.custom is not None:
        for key, value in cut.custom.items():
            if isinstance(value, Recording):
                resolve_recording(value)
            elif isinstance(value, (Array, TemporalArray, Features)):
                resolve_array(value)

    return cut


@data_type_parser(["nemo", "nemo_tarred"])
def read_nemo_manifest(config) -> tuple[CutSet, bool]:
    """Read NeMo manifest and return a Lhotse CutSet."""
    common_kwargs = {}
    for key in ("text_field", "lang_field", "shuffle", "shard_seed", "extra_fields"):
        if key in config:
            if key == "shuffle":
                common_kwargs["shuffle_shards"] = config[key]
            else:
                common_kwargs[key] = config[key]
    # The option below is to allow a special case of NeMo manifest iteration as Lhotse CutSet
    # without performing any I/O. NeMo manifests typically don't have sampling_rate information required by Lhotse,
    # so lhotse has to look up the headers of audio files to fill it on-the-fly.
    # (this only has an impact on non-tarred data; tarred data is read into memory anyway).
    # This is useful for utility scripts that iterate metadata and estimate optimal batching settings
    # and other data statistics.
    metadata_only = config.get("metadata_only", False)
    force_finite = config.get("force_finite", False)
    notar_kwargs = {"metadata_only": metadata_only}
    is_tarred = config.get("tarred_audio_filepaths") is not None
    if isinstance(config.manifest_filepath, (str, Path)):
        logging.info(
            f"""Initializing Lhotse CutSet from a single NeMo manifest
            (is_tarred={is_tarred}): '{config.manifest_filepath}'"""
        )
        if is_tarred and not metadata_only:
            cuts = CutSet(
                LazyNeMoTarredIterator(
                    config.manifest_filepath,
                    tar_paths=config.tarred_audio_filepaths,
                    skip_missing_manifest_entries=config.get("skip_missing_manifest_entries", False),
                    slice_length=config.get("slice_length", None),
                    **common_kwargs,
                )
            )
            if not force_finite:
                cuts = cuts.repeat(preserve_id=True)
        else:
            cuts = CutSet(LazyNeMoIterator(config.manifest_filepath, **notar_kwargs, **common_kwargs))
    else:
        # Format option 1:
        #   Assume it's [[path1], [path2], ...] (same for tarred_audio_filepaths).
        #   This is the format for multiple NeMo buckets.
        #   Note: we set "weights" here to be proportional to the number of utterances in each data source.
        #         this ensures that we distribute the data from each source uniformly throughout each epoch.
        #         Setting equal weights would exhaust the shorter data sources closer the towards the beginning
        #         of an epoch (or over-sample it in the case of infinite CutSet iteration with .repeat()).
        # Format option 2:
        #   Assume it's [[path1, weight1], [path2, weight2], ...] (while tarred_audio_filepaths remain unchanged).
        #   Note: this option allows to manually set the weights for multiple datasets.
        # Format option 3:
        #   i.e., NeMo concatenated dataset
        #   Assume it's [path1, path2, ...] (while tarred_audio_filepaths in the same format).
        logging.info(
            f"""Initializing Lhotse CutSet from multiple NeMo manifest
            (is_tarred={is_tarred}) sources with a weighted multiplexer.
            We found the following sources and weights: """
        )
        cutsets = []
        weights = []
        tar_paths = config.tarred_audio_filepaths if is_tarred else repeat((None,))
        # Create a stream for each dataset.
        for manifest_info, tar_path in zip(config.manifest_filepath, tar_paths):
            if is_tarred and isinstance(tar_path, (list, tuple, ListConfig)):
                # if it's in option 1 or 2
                (tar_path,) = tar_path
                manifest_path = manifest_info[0]
            else:
                # if it's in option 3
                manifest_path = manifest_info
            # First, convert manifest_path[+tar_path] to an iterator.
            if is_tarred and not metadata_only:
                nemo_iter = LazyNeMoTarredIterator(
                    manifest_path=manifest_path,
                    tar_paths=tar_path,
                    skip_missing_manifest_entries=config.get("skip_missing_manifest_entries", False),
                    slice_length=config.get("slice_length", None),
                    **common_kwargs,
                )
            else:
                nemo_iter = LazyNeMoIterator(manifest_path, **notar_kwargs, **common_kwargs)
            # Then, determine the weight or use one provided
            if isinstance(manifest_info, str) or len(manifest_info) == 1:
                weight = len(nemo_iter)
            else:
                assert (
                    isinstance(manifest_info, Sequence)
                    and len(manifest_info) == 2
                    and isinstance(manifest_info[1], (int, float))
                ), (
                    "Supported inputs types for config.manifest_filepath are: "
                    "str | list[list[str]] | list[tuple[str, number]] "
                    "where str is a path and number is a mixing weight (it may exceed 1.0). "
                    f"We got: '{manifest_info}'"
                )
                weight = manifest_info[1]
            logging.info(f"- {manifest_path=} {weight=}")
            # [optional] When we have a limit on the number of open streams,
            #   split the manifest to individual shards if applicable.
            #   This helps the multiplexing achieve closer data distribution
            #   to the one desired in spite of the limit.
            if config.get("max_open_streams") is not None:
                for subiter in nemo_iter.to_shards():
                    cutsets.append(CutSet(subiter))
                    weights.append(weight)
            else:
                cutsets.append(CutSet(nemo_iter))
                weights.append(weight)
        cuts = mux(
            *cutsets,
            weights=weights,
            max_open_streams=config.get("max_open_streams"),
            seed=config.get("shard_seed", "trng"),
            force_finite=force_finite or metadata_only,
        )
    return cuts, is_tarred


@data_type_parser("multi_speaker_simulator")
def read_multi_speaker_simulator(config: DictConfig) -> tuple[CutSet, bool]:
    # Import here to avoid circular dependency
    from nemo.collections.asr.parts.utils.asr_multispeaker_utils import MultiSpeakerMixtureGenerator

    multi_speaker_cuts = CutSet(
        MultiSpeakerMixtureGenerator(
            manifest_filepath=config.manifest_filepath,
            simulator_type=config.simulator_type,
            sample_rate=config.get("sample_rate", 16000),
            min_delay=config.get("min_delay", 0.5),
            min_duration=config.get("min_duration", 0.1),
            max_duration=config.get("max_duration", 60),
            num_speakers=config.get("num_speakers", 2),
            global_rank=config.get("global_rank", 0),
            world_size=config.get("world_size", 1),
        )
    )
    is_tarred = config.get("is_tarred", False)
    return multi_speaker_cuts, is_tarred


def mux(
    *cutsets: CutSet,
    weights: list[Union[int, float]],
    max_open_streams: Union[int, None] = None,
    seed: Union[str, int] = "trng",
    force_finite: bool = False,
) -> CutSet:
    """
    Helper function to call the right multiplexing method flavour in lhotse.
    The result is always an infinitely iterable ``CutSet``, but depending on whether ``max_open_streams`` is set,
    it will select a more appropriate multiplexing strategy.
    """
    if max_open_streams is not None:
        assert not force_finite, "max_open_streams and metadata_only/force_finite options are not compatible"
        cuts = CutSet.infinite_mux(*cutsets, weights=weights, seed=seed, max_open_streams=max_open_streams)
    else:
        if not force_finite:
            cutsets = [cs.repeat(preserve_id=True) for cs in cutsets]
        if len(cutsets) == 1:
            # CutSet.mux must take more than one CutSet.
            cuts = cutsets[0]
        else:
            cuts = CutSet.mux(*cutsets, weights=weights, seed=seed)
    return cuts


def guess_parse_cutset(inp: Union[str, dict, omegaconf.DictConfig]) -> CutSet:
    """
    Utility function that supports opening a CutSet from:
    * a string path to YAML input spec (see :func:`read_dataset_config` for details)
    * a string path to Lhotse non-tarred JSONL manifest
    * a string path to NeMo non-tarred JSON manifest
    * a dictionary specifying inputs with keys available in
        :class:`nemo.collections.common.data.lhotse.dataloader.LhotseDataLoadingConfig`

    It's intended to be used in a generic context where we are not sure which way the user will specify the inputs.
    """
    from nemo.collections.common.data.lhotse.dataloader import make_structured_with_schema_warnings

    if isinstance(inp, (dict, omegaconf.DictConfig)):
        try:
            config = make_structured_with_schema_warnings(OmegaConf.from_dotlist([f"{k}={v}" for k, v in inp.items()]))
            cuts, _ = read_cutset_from_config(config)
            return cuts
        except Exception as e:
            raise RuntimeError(
                f"Couldn't open CutSet based on dict input {inp} (is it compatible with LhotseDataLoadingConfig?)"
            ) from e
    elif isinstance(inp, str):
        if inp.endswith(".yaml"):
            # Path to YAML file with the input configuration
            config = make_structured_with_schema_warnings(OmegaConf.from_dotlist([f"input_cfg={inp}"]))
        elif inp.endswith(".jsonl") or inp.endswith(".jsonl.gz"):
            # Path to a Lhotse non-tarred manifest
            config = make_structured_with_schema_warnings(OmegaConf.from_dotlist([f"cuts_path={inp}"]))
        else:
            # Assume anything else is a NeMo non-tarred manifest
            config = make_structured_with_schema_warnings(OmegaConf.from_dotlist([f"manifest_filepath={inp}"]))
        cuts, _ = read_cutset_from_config(config)
        return cuts
    else:
        raise RuntimeError(f'Unsupported input type: {type(inp)} (expected a dict or a string)')
