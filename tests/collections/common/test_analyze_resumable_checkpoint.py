# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import sys
from pathlib import Path

from lhotse.index_pack import IndexPackCollectionSpec, index_pack_collection_key, write_index_pack
from lhotse.indexing import create_jsonl_index


def _load_analyzer_module():
    path = Path(__file__).parents[3] / "scripts" / "dataloading" / "analyze_resumable_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_analyze_resumable_checkpoint_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mixed_legacy_packed_and_unknown_leaf_states_preserve_source_indices():
    analyzer = _load_analyzer_module()
    tree = [
        {"position": 4, "shard_id": 0, "num_shards": 2, "epoch": 1},
        {
            "global_position": 3,
            "global_shard_id": 1,
            "global_num_shards": 3,
            "num_iters": 2,
            "global_seed": 123,
        },
        # A partial/future terminal schema is not decoded, but it must still
        # reserve source index 2 so source 3 is not mislabeled as source 2.
        {"position": "unsupported", "shard_id": 0, "num_shards": 1},
        {"position": 7, "shard_id": 0, "num_shards": 1, "epoch": 0},
    ]

    leaves, span = analyzer._collect_leaf_states_with_span(tree, rank=0, worker="0")

    assert span == 4
    assert [leaf.source_index for leaf in leaves] == [0, 1, 3]
    packed = leaves[1]
    assert packed.epoch == 2
    assert packed.position == 3
    assert packed.shard_id == 1
    assert packed.num_shards == 3
    assert packed.state_type.endswith("/packed-global")
    # Partition 1/3 of an 11-record collection contains four records.
    assert analyzer._consumed_items(packed, total_len=11) == 11


def test_index_pack_catalog_supplies_exact_leaf_totals(tmp_path):
    analyzer = _load_analyzer_module()
    collections = []
    expected = [3, 5]
    manifests = []
    for index, count in enumerate(expected):
        manifest = tmp_path / f"manifest-{index}.jsonl"
        manifest.write_text("".join(f'{{"id": {item}}}\n' for item in range(count)), encoding="utf-8")
        create_jsonl_index(manifest)
        manifests.append(manifest)
        collections.append(
            IndexPackCollectionSpec(
                role="manifest",
                kind="jsonl",
                source_spec=str(manifest),
                paths=(str(manifest),),
            )
        )
    pack_path = tmp_path / "dataset.idxpack"
    # Exact identity, not catalog insertion order, must associate leaf totals.
    write_index_pack(pack_path, list(reversed(collections)))
    specs = [
        analyzer.DatasetSpec(
            source_index=index,
            name=f"leaf-{index}",
            index_pack_path=str(pack_path),
            index_pack_collection_keys=[index_pack_collection_key("manifest", "jsonl", str(manifests[index])).hex()],
        )
        for index in range(2)
    ]

    analyzer._fill_totals_from_index_packs(specs)

    assert [spec.total_items for spec in specs] == expected
    assert [spec.matched_index_pack_collection_key for spec in specs] == [
        index_pack_collection_key("manifest", "jsonl", str(manifest)).hex() for manifest in manifests
    ]


def test_nested_sampler_states_preserve_unknown_leaf_span():
    analyzer = _load_analyzer_module()
    sampler_state = {
        "samplers": [
            {
                "cuts_state": [
                    {"position": 2, "shard_id": 0, "num_shards": 1, "epoch": 0},
                ]
            },
            {
                "cuts_state": [
                    {"position": "future-schema", "shard_id": 0, "num_shards": 1},
                ]
            },
            {
                "cuts_state": [
                    {
                        "global_position": 4,
                        "global_shard_id": 0,
                        "global_num_shards": 1,
                        "num_iters": 1,
                        "global_seed": 123,
                    }
                ]
            },
        ]
    }

    leaves, span = analyzer._collect_leaves_from_sampler(
        sampler_state,
        rank=0,
        worker="0",
        path="$",
    )

    assert span == 3
    assert [leaf.source_index for leaf in leaves] == [0, 2]


def test_recipe_index_pack_root_precedes_default_but_not_explicit_override(tmp_path):
    analyzer = _load_analyzer_module()
    config = {
        "data": {
            "train_ds": {
                "index_pack_root": "/recipe/packs",
                "input_cfg": [
                    {
                        "type": "lhotse_as_conversation",
                        "manifest_filepath": "/data/train.jsonl",
                        "index_pack": "dataset.idxpack",
                        "weight": 1.0,
                    }
                ],
            }
        }
    }

    recipe_specs = analyzer.collect_dataset_specs(
        config,
        config_path=tmp_path / "recipe.yaml",
        indexes_root=None,
        default_index_pack_root="/cluster/default/packs",
    )
    explicit_specs = analyzer.collect_dataset_specs(
        config,
        config_path=tmp_path / "recipe.yaml",
        indexes_root=None,
        index_pack_root="/explicit/packs",
        default_index_pack_root="/cluster/default/packs",
    )

    assert recipe_specs[0].index_pack_path == "/recipe/packs/dataset.idxpack"
    assert explicit_specs[0].index_pack_path == "/explicit/packs/dataset.idxpack"


def test_collect_dataset_specs_preserves_generic_leaf_tags(tmp_path):
    analyzer = _load_analyzer_module()
    config = {
        "data": {
            "train_ds": {
                "input_cfg": [
                    {
                        "type": "group",
                        "tags": {"group": "alpha", "priority": 2},
                        "weight": 1.0,
                        "input_cfg": [
                            {
                                "type": "lhotse_as_conversation",
                                "manifest_filepath": "/data/train.jsonl",
                                "weight": 1.0,
                            }
                        ],
                    }
                ]
            }
        }
    }

    specs = analyzer.collect_dataset_specs(
        config,
        config_path=tmp_path / "recipe.yaml",
        indexes_root=None,
    )

    assert len(specs) == 1
    assert specs[0].tags == {"group": "alpha", "priority": 2}


def test_group_index_pack_root_overrides_recipe_but_not_explicit_cli_root(tmp_path):
    analyzer = _load_analyzer_module()
    config = {
        "data": {
            "train_ds": {
                "index_pack_root": "/recipe/packs",
                "input_cfg": [
                    {
                        "type": "group",
                        "index_pack_root": "/outer-group/packs",
                        "weight": 1.0,
                        "input_cfg": [
                            {
                                "type": "group",
                                "index_pack_root": "/inner-group/packs",
                                "weight": 1.0,
                                "input_cfg": [
                                    {
                                        "type": "lhotse_as_conversation",
                                        "manifest_filepath": "/data/train.jsonl",
                                        "index_pack": "dataset.idxpack",
                                        "weight": 1.0,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    }

    inherited_specs = analyzer.collect_dataset_specs(
        config,
        config_path=tmp_path / "recipe.yaml",
        indexes_root=None,
    )
    explicit_specs = analyzer.collect_dataset_specs(
        config,
        config_path=tmp_path / "recipe.yaml",
        indexes_root=None,
        index_pack_root="/explicit/packs",
    )

    assert inherited_specs[0].index_pack_path == "/inner-group/packs/dataset.idxpack"
    assert explicit_specs[0].index_pack_path == "/explicit/packs/dataset.idxpack"


def test_index_pack_collection_keys_use_structural_source_fields():
    analyzer = _load_analyzer_module()
    path_keys = analyzer._index_pack_collection_keys({"type": "custom", "paths": ["/data/a.jsonl", "/data/b.jsonl"]})
    directory_keys = analyzer._index_pack_collection_keys({"type": "custom", "data_dir": "/data/shards"})

    assert len(path_keys) == 2
    assert len(set(path_keys)) == 2
    assert len(directory_keys) == 1


def test_sequential_packed_state_uses_partitioned_prior_shard_lengths():
    analyzer = _load_analyzer_module()
    state = {
        "current_iter_idx": 2,
        "packed_current_position": 1,
        "num_iters": 0,
        "iter_order": None,
        "global_position": 0,
        "global_seed": None,
        "global_shard_id": 1,
        "global_num_shards": 2,
    }

    leaf = analyzer._leaf_from_state(0, 0, "0", "$", "LazyPackedManifestIterator", state)

    assert leaf.state_type.endswith("/packed-sequential")
    assert leaf.packed_current_shard == 2
    assert leaf.packed_current_position == 1
    # Worker partition 1/2 consumed 2 records from each prior shard, then one
    # record in shard 2. The serialized global_position=0 must be ignored.
    assert analyzer._consumed_items(leaf, 15, packed_shard_lengths=[5, 4, 6]) == 5
    assert analyzer._consumed_items(leaf, 15) is None


def test_sequential_lazy_chain_state_is_reserved_without_false_progress():
    analyzer = _load_analyzer_module()
    tree = [
        {
            "current_iter_idx": 2,
            "num_iters": 0,
            "iter_order": None,
            "global_position": 0,
            "global_seed": None,
            "global_shard_id": 0,
            "global_num_shards": 1,
        }
    ]

    leaves, span = analyzer._collect_leaf_states_with_span(tree, rank=0, worker="0")

    assert leaves == []
    assert span == 1
