# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import warnings
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Sequence, Tuple, Union

import omegaconf
from lhotse import CutSet, Features, Recording
from lhotse.array import Array, TemporalArray
from lhotse.cut import Cut, MixedCut, PaddingCut
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator, LazyNeMoTarredIterator
from nemo.collections.common.data.lhotse.text_adapters import LhotseTextAdapter, LhotseTextPairAdapter
from nemo.collections.common.parts.preprocessing.manifest import get_full_path


def read_cutset_from_config(config: DictConfig) -> Tuple[CutSet, bool]:
    """
    Reads NeMo configuration and creates a CutSet either from Lhotse or NeMo manifests.

    Returns a tuple of ``CutSet`` and a boolean indicating whether the data is tarred (True) or not (False).
    """
    # First, check if the dataset is specified in the new configuration format and use it if possible.
    if config.get("input_cfg") is not None:
        return read_dataset_config(config)
    # Now, we'll figure out if we should read Lhotse manifest or NeMo manifest.
    use_nemo_manifest = all(config.get(opt) is None for opt in ("cuts_path", "shar_path"))
    if use_nemo_manifest:
        assert (
            config.get("manifest_filepath") is not None
        ), "You must specify either: manifest_filepath, cuts_path, or shar_path"
        is_tarred = config.get("tarred_audio_filepaths") is not None
    else:
        is_tarred = config.get("shar_path") is not None
    if use_nemo_manifest:
        # Read NeMo manifest -- use the right wrapper depending on tarred/non-tarred.
        cuts = read_nemo_manifest(config, is_tarred)
    else:
        # Read Lhotse manifest (again handle both tarred(shar)/non-tarred).
        cuts = read_lhotse_manifest(config, is_tarred)
    return cuts, is_tarred


KNOWN_DATASET_CONFIG_TYPES = frozenset(("nemo", "nemo_tarred", "lhotse", "lhotse_shar", "txt", "txt_pair", "group"))


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
        "shuffle": config.shuffle,
        "shard_seed": config.shard_seed,
        "text_field": config.text_field,
        "lang_field": config.lang_field,
        "metadata_only": config.metadata_only,
        "max_open_streams": config.max_open_streams,
    }
    input_cfg = config.input_cfg
    if isinstance(input_cfg, (str, Path)):
        # Resolve /path/to/input_cfg.yaml into config contents if needed.
        input_cfg = OmegaConf.load(input_cfg)
    cuts, is_tarred = parse_and_combine_datasets(input_cfg, propagate_attrs=propagate_attrs)
    return cuts, is_tarred


def parse_group(grp_cfg: DictConfig, propagate_attrs: dict) -> [CutSet, bool]:
    assert grp_cfg.type in KNOWN_DATASET_CONFIG_TYPES, f"Unknown item type in dataset config list: {grp_cfg.type=}"
    if grp_cfg.type == "nemo_tarred":
        is_tarred = True
        cuts = read_nemo_manifest(grp_cfg, is_tarred=is_tarred)
    elif grp_cfg.type == "nemo":
        is_tarred = False
        cuts = read_nemo_manifest(grp_cfg, is_tarred=is_tarred)
    elif grp_cfg.type == "lhotse_shar":
        is_tarred = True
        cuts = read_lhotse_manifest(grp_cfg, is_tarred=is_tarred)
    elif grp_cfg.type == "lhotse":
        is_tarred = False
        cuts = read_lhotse_manifest(grp_cfg, is_tarred=is_tarred)
    # Note: "txt" and "txt_pair" have "is_tarred" set to True.
    #       The main reason is to enable combination of tarred audio and text dataloading,
    #       since we don't allow combination of tarred and non-tarred datasets.
    #       We choose to treat text as-if it was tarred, which also tends to be more
    #       efficient as it moves the text file iteration into dataloading subprocess.
    elif grp_cfg.type == "txt":
        is_tarred = True
        cuts = read_txt_paths(grp_cfg)
    elif grp_cfg.type == "txt_pair":
        is_tarred = True
        cuts = read_txt_pair_paths(grp_cfg)
    elif grp_cfg.type == "group":
        cuts, is_tarred = parse_and_combine_datasets(
            grp_cfg.input_cfg,
            propagate_attrs=propagate_attrs,
        )
    else:
        raise ValueError(f"Unrecognized group: {grp_cfg.type}")
    # Attach extra tags to every utterance dynamically, if provided.
    if (extra_tags := grp_cfg.get("tags")) is not None:
        cuts = cuts.map(partial(attach_tags, tags=extra_tags), apply_fn=None)
    return cuts, is_tarred


def read_txt_paths(config: DictConfig) -> CutSet:
    return CutSet(
        LhotseTextAdapter(
            paths=config.paths,
            language=config.language,
            shuffle_shards=config.shuffle,
            shard_seed=config.shard_seed,
        )
    ).repeat()


def read_txt_pair_paths(config: DictConfig) -> CutSet:
    return CutSet(
        LhotseTextPairAdapter(
            source_paths=config.source_paths,
            target_paths=config.target_paths,
            source_language=config.source_language,
            target_language=config.target_language,
            shuffle_shards=config.shuffle,
            shard_seed=config.shard_seed,
        )
    ).repeat()


def attach_tags(cut, tags: dict):
    for key, val in tags.items():
        setattr(cut, key, val)
    return cut


def parse_and_combine_datasets(
    config_list: Union[list[DictConfig], ListConfig], propagate_attrs: dict
) -> tuple[CutSet, bool]:
    cuts = []
    weights = []
    tarred_status = []
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

    assert all(t == tarred_status[0] for t in tarred_status), "Mixing tarred and non-tarred datasets is not supported."
    assert len(weights) == 0 or len(cuts) == len(
        weights
    ), "Missing dataset weight. When weighting datasets, every dataset must have a specified weight."
    if len(cuts) > 1:
        cuts = mux(
            *cuts,
            weights=weights if weights else None,
            max_open_streams=propagate_attrs["max_open_streams"],
            seed=propagate_attrs["shard_seed"],
            metadata_only=propagate_attrs["metadata_only"],
        )
    else:
        (cuts,) = cuts
    return cuts, tarred_status[0]


def read_lhotse_manifest(config, is_tarred: bool) -> CutSet:
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
        shard_seed = config.shard_seed
        metadata_only = config.metadata_only
        if config.get("cuts_path") is not None:
            warnings.warn("Note: lhotse.cuts_path will be ignored because lhotse.shar_path was provided.")
        if isinstance(config.shar_path, (str, Path)):
            logging.info(f"Initializing Lhotse Shar CutSet (tarred) from a single data source: '{config.shar_path}'")
            cuts = CutSet.from_shar(
                **_resolve_shar_inputs(config.shar_path, metadata_only), shuffle_shards=True, seed=shard_seed
            )
            if not metadata_only:
                cuts = cuts.repeat()
        else:
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
                        **_resolve_shar_inputs(path, metadata_only), shuffle_shards=True, seed=shard_seed
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
                        **_resolve_shar_inputs(path, metadata_only), shuffle_shards=True, seed=shard_seed
                    )
                logging.info(f"- {path=} {weight=}")
                cutsets.append(cs)
                weights.append(weight)
            cuts = mux(
                *cutsets,
                weights=weights,
                max_open_streams=config.max_open_streams,
                seed=config.shard_seed,
                metadata_only=metadata_only,
            )
    else:
        # Regular Lhotse manifest points to individual audio files (like native NeMo manifest).
        path = config.cuts_path
        cuts = CutSet.from_file(path).map(partial(resolve_relative_paths, manifest_path=path))
    return cuts


def _resolve_shar_inputs(path: str | Path, only_metadata: bool) -> dict:
    if only_metadata:
        return dict(fields={"cuts": sorted(Path(path).glob("cuts.*"))})
    else:
        return dict(in_dir=path)


def resolve_relative_paths(cut: Cut, manifest_path: str) -> Cut:
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


def read_nemo_manifest(config, is_tarred: bool) -> CutSet:
    common_kwargs = {
        "text_field": config.text_field,
        "lang_field": config.lang_field,
        "shuffle_shards": config.shuffle,
        "shard_seed": config.shard_seed,
    }
    # The option below is to allow a special case of NeMo manifest iteration as Lhotse CutSet
    # without performing any I/O. NeMo manifests typically don't have sampling_rate information required by Lhotse,
    # so lhotse has to look up the headers of audio files to fill it on-the-fly.
    # (this only has an impact on non-tarred data; tarred data is read into memory anyway).
    # This is useful for utility scripts that iterate metadata and estimate optimal batching settings
    # and other data statistics.
    notar_kwargs = {"metadata_only": config.metadata_only}
    metadata_only = config.metadata_only
    if isinstance(config.manifest_filepath, (str, Path)):
        logging.info(f"Initializing Lhotse CutSet from a single NeMo manifest (tarred): '{config.manifest_filepath}'")
        if is_tarred and not metadata_only:
            cuts = CutSet(
                LazyNeMoTarredIterator(
                    config.manifest_filepath,
                    tar_paths=config.tarred_audio_filepaths,
                    **common_kwargs,
                )
            ).repeat()
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
        logging.info(
            f"Initializing Lhotse CutSet from multiple tarred NeMo manifest sources with a weighted multiplexer. "
            f"We found the following sources and weights: "
        )
        cutsets = []
        weights = []
        tar_paths = config.tarred_audio_filepaths if is_tarred else repeat((None,))
        # Create a stream for each dataset.
        for manifest_info, (tar_path,) in zip(config.manifest_filepath, tar_paths):
            # First, convert manifest_path[+tar_path] to an iterator.
            manifest_path = manifest_info[0]
            if is_tarred and not metadata_only:
                nemo_iter = LazyNeMoTarredIterator(
                    manifest_path=manifest_path,
                    tar_paths=tar_path,
                    **common_kwargs,
                )
            else:
                nemo_iter = LazyNeMoIterator(manifest_path, **notar_kwargs, **common_kwargs)
            # Then, determine the weight or use one provided
            if len(manifest_info) == 1:
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
            if config.max_open_streams is not None:
                for subiter in nemo_iter.to_shards():
                    cutsets.append(CutSet(subiter))
                    weights.append(weight)
            else:
                cutsets.append(CutSet(nemo_iter))
                weights.append(weight)
        # Finally, we multiplex the dataset streams to mix the data.
        cuts = mux(
            *cutsets,
            weights=weights,
            max_open_streams=config.max_open_streams,
            seed=config.shard_seed,
            metadata_only=metadata_only,
        )
    return cuts


def mux(
    *cutsets: CutSet,
    weights: list[int | float],
    max_open_streams: int | None = None,
    seed: str | int = "trng",
    metadata_only: bool = False,
) -> CutSet:
    """
    Helper function to call the right multiplexing method flavour in lhotse.
    The result is always an infinitely iterable ``CutSet``, but depending on whether ``max_open_streams`` is set,
    it will select a more appropriate multiplexing strategy.
    """
    if max_open_streams is not None:
        assert not metadata_only, "max_open_streams and metadata_only options are not compatible"
        cuts = CutSet.infinite_mux(*cutsets, weights=weights, seed=seed, max_open_streams=max_open_streams)
    else:
        if not metadata_only:
            cutsets = [cs.repeat() for cs in cutsets]
        cuts = CutSet.mux(*cutsets, weights=weights, seed=seed)
    return cuts


def guess_parse_cutset(inp: Union[str, dict, omegaconf.DictConfig]) -> CutSet:
    """
    Utility function that supports opening a CutSet from:
    * a string path to YAML input spec (see :func:`read_dataset_config` for details)
    * a string path to Lhotse non-tarred JSONL manifest
    * a string path to NeMo non-tarred JSON manifest
    * a dictionary specifying inputs with keys available in :class:`nemo.collections.common.data.lhotse.dataloader.LhotseDataLoadingConfig`

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
