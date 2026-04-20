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
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest
from omegaconf import DictConfig

try:
    from torch.distributed._local_tensor import LocalTensorMode  # noqa: E402
except ImportError:
    pytest.skip("Local tensor mode requires PyTorch >= 2.10", allow_module_level=True)

from lightning.pytorch.strategies.model_parallel import _setup_device_mesh

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model
from nemo.collections.speechlm2.data import DataModule

nemo_automodel = pytest.importorskip("nemo_automodel")
from nemo_automodel.components.distributed.config import FSDP2Config  # noqa: E402

from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_dist():
    """Set up minimal env for fake distributed backend, tear down after test."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "0")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture
def tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("tok")
    text_path = tmpdir / "text.txt"
    text_path.write_text("\n".join(chr(i) for i in range(256)))
    create_spt_model(
        text_path,
        vocab_size=512,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        bos=True,
        eos=True,
        remove_extra_whitespaces=True,
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_rank_mapping(sym_val, world_size):
    """Extract ``{global_rank: value}`` mapping from a SymInt produced by LocalTensorMode.

    Inside ``LocalTensorMode``, ``DeviceMesh.get_local_rank()`` returns a
    ``SymInt`` backed by a ``LocalIntNode`` (per-rank varying values) or a
    ``ConstantIntNode`` (same value on every rank).  Arithmetic on these
    ``SymInt`` values preserves the per-rank structure.

    This helper unwraps the result into a plain dict for easy assertions.
    """
    if isinstance(sym_val, int):
        return {r: sym_val for r in range(world_size)}
    node = sym_val.node
    if hasattr(node, "_local_ints"):
        return dict(node._local_ints)
    # ConstantIntNode – same value for every rank.
    val = node.maybe_as_int() if hasattr(node, "maybe_as_int") else node.val
    return {r: val for r in range(world_size)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pp,dp_rep,dp_shard,cp,tp",
    [
        (1, 1, 1, 1, 1),
        (1, 2, 2, 1, 1),
        (1, 1, 2, 1, 1),
        (2, 3, 2, 1, 2),
        (2, 2, 4, 2, 2),
    ],
)
def test_dp_rank_via_strategy(fake_dist, pp, dp_rep, dp_shard, cp, tp):
    """Verify the DP rank formula using the real AutomodelParallelStrategy.

    Uses ``LocalTensorMode`` to obtain per-rank symbolic results for every
    global rank in a single process and checks that:

    * The DataModule formula (``dp_replicate.local_rank * dp_shard.size +
      dp_shard.local_rank``) agrees with the flattened ``"dp"`` submesh.
    * The correct number of unique DP ranks exists.
    * All DP ranks are in ``[0, dp_size)``.
    * Ranks that share the same ``(dp_replicate, dp_shard)`` coordinates are
      assigned the same DP rank.
    """
    dp_size = dp_rep * dp_shard
    world_size = pp * dp_size * cp * tp
    dist.init_process_group(backend="fake", rank=0, world_size=world_size)

    strategy = AutomodelParallelStrategy(
        pp_size=pp,
        tp_size=tp,
        cp_size=cp,
        dp_size=dp_size,
        dp_replicate_size=dp_rep,
        distributed_config=FSDP2Config(backend="gloo"),
    )

    with LocalTensorMode(world_size):
        device_mesh, _ = strategy.create_device_mesh()

        # Method 1: DataModule._get_dp_rank formula
        dp_rank_manual = (
            device_mesh["dp_replicate"].get_local_rank() * device_mesh["dp_shard"].size()
            + device_mesh["dp_shard"].get_local_rank()
        )
        # Method 2: flattened "dp" submesh (used by distributed_sampler_kwargs)
        dp_rank_flat = device_mesh["dp"].get_local_rank()
        dp_world_size = device_mesh["dp_replicate", "dp_shard"].size()

    assert dp_world_size == dp_size

    manual = _extract_rank_mapping(dp_rank_manual, world_size)
    flat = _extract_rank_mapping(dp_rank_flat, world_size)

    # The two computation methods must agree for every global rank.
    assert manual == flat

    # Correct number of unique dp ranks.
    assert len(set(manual.values())) == dp_size

    # All dp_ranks in valid range.
    assert all(0 <= v < dp_size for v in manual.values())

    # Ranks sharing the same (dp_replicate, dp_shard) coordinates must have
    # the same dp_rank; different coordinates must differ.
    mesh_tensor = torch.arange(world_size).reshape(pp, dp_rep, dp_shard, cp, tp)
    for dr in range(dp_rep):
        for ds in range(dp_shard):
            ranks_in_group = mesh_tensor[:, dr, ds, :, :].flatten().tolist()
            dp_ranks_in_group = {manual[r] for r in ranks_in_group}
            assert (
                len(dp_ranks_in_group) == 1
            ), f"DP group (dp_rep={dr}, dp_shard={ds}) maps to multiple dp_ranks: {dp_ranks_in_group}"


def test_non_dp_dims_share_dp_rank(fake_dist):
    """Ranks that differ only in pp / tp / cp get the same dp_rank."""
    pp, dp_rep, dp_shard, cp, tp = 2, 3, 2, 1, 2
    dp_size = dp_rep * dp_shard
    world_size = pp * dp_size * cp * tp
    dist.init_process_group(backend="fake", rank=0, world_size=world_size)

    with LocalTensorMode(world_size):
        strategy = AutomodelParallelStrategy(
            pp_size=pp,
            tp_size=tp,
            cp_size=cp,
            dp_size=dp_size,
            dp_replicate_size=dp_rep,
            distributed_config=FSDP2Config(backend="gloo"),
        )
        device_mesh, _ = strategy.create_device_mesh()
        dp_rank_sym = (
            device_mesh["dp_replicate"].get_local_rank() * device_mesh["dp_shard"].size()
            + device_mesh["dp_shard"].get_local_rank()
        )

    rank_to_dp = _extract_rank_mapping(dp_rank_sym, world_size)

    # Mesh layout: (pp=2, dp_rep=3, dp_shard=2, cp=1, tp=2)
    # Rank 0: (pp=0, dp_rep=0, dp_shard=0, cp=0, tp=0)
    # Rank 1: (pp=0, dp_rep=0, dp_shard=0, cp=0, tp=1)  -- differs in TP
    # Rank 12: (pp=1, dp_rep=0, dp_shard=0, cp=0, tp=0) -- differs in PP
    # Rank 4:  (pp=0, dp_rep=1, dp_shard=0, cp=0, tp=0) -- differs in dp_replicate
    # Rank 2:  (pp=0, dp_rep=0, dp_shard=1, cp=0, tp=0) -- differs in dp_shard

    # Same DP coordinates, different non-DP dims → same dp_rank
    assert rank_to_dp[0] == rank_to_dp[1], "TP variation should not change dp_rank"
    assert rank_to_dp[0] == rank_to_dp[12], "PP variation should not change dp_rank"

    # Different DP coordinates → different dp_rank
    assert rank_to_dp[0] != rank_to_dp[4], "dp_replicate variation must change dp_rank"
    assert rank_to_dp[0] != rank_to_dp[2], "dp_shard variation must change dp_rank"


def test_datamodule_get_dp_rank_automodel(fake_dist, tokenizer):
    """DataModule._get_dp_rank() / _get_world_size() return correct values."""
    pp, dp_rep, dp_shard, cp, tp = 2, 3, 2, 1, 2
    dp_size = dp_rep * dp_shard
    world_size = pp * dp_size * cp * tp
    dist.init_process_group(backend="fake", rank=0, world_size=world_size)

    strategy = AutomodelParallelStrategy(
        pp_size=pp,
        tp_size=tp,
        cp_size=cp,
        dp_size=dp_size,
        dp_replicate_size=dp_rep,
        distributed_config=FSDP2Config(backend="gloo"),
    )
    # Create mesh outside LocalTensorMode → real int values for fake rank 0
    device_mesh, _ = strategy.create_device_mesh()

    cfg = DictConfig({"train_ds": {"batch_size": 2}})
    data = DataModule(cfg, tokenizer=tokenizer, dataset=Identity())

    # Wire up a mock trainer so _get_dp_rank() can find the device mesh.
    data.trainer = SimpleNamespace(model=SimpleNamespace(device_mesh=device_mesh))

    dp_rank = data._get_dp_rank()
    dp_ws = data._get_world_size()

    assert dp_rank is not None, "_get_dp_rank() returned None (missing return?)"
    assert dp_ws is not None, "_get_world_size() returned None (missing return?)"
    assert dp_rank == 0, f"Fake rank 0 should map to dp_rank 0, got {dp_rank}"
    assert dp_ws == dp_size, f"Expected dp_world_size={dp_size}, got {dp_ws}"


def test_dataloader_data_partitioning(tmp_path, tokenizer):
    """Different dp_ranks get disjoint data; same dp_rank is deterministic."""
    dp_world_size = 6
    num_cuts = 24
    cuts_path = str(tmp_path / "cuts.jsonl.gz")
    (
        DummyManifest(CutSet, begin_id=0, end_id=num_cuts, with_data=True)
        .save_audios(tmp_path / "audio")
        .drop_in_memory_data()
        .to_file(cuts_path)
    )

    cfg = DictConfig(
        {
            "input_cfg": [{"type": "lhotse", "cuts_path": cuts_path}],
            "batch_size": 2,
            "force_finite": True,
            "force_map_dataset": True,
            "seed": 0,
            "num_workers": 0,
        }
    )

    # Collect cut IDs from each dp_rank.
    ids_per_rank = {}
    for dp_rank in range(dp_world_size):
        dl = get_lhotse_dataloader_from_config(
            config=cfg,
            global_rank=dp_rank,
            world_size=dp_world_size,
            dataset=Identity(),
            tokenizer=tokenizer,
        )
        ids_per_rank[dp_rank] = [c.id for batch in dl for c in batch]

    # Different DP ranks must receive disjoint cuts.
    for r1 in range(dp_world_size):
        for r2 in range(r1 + 1, dp_world_size):
            overlap = set(ids_per_rank[r1]) & set(ids_per_rank[r2])
            assert not overlap, f"Ranks {r1} and {r2} share cut IDs: {overlap}"

    # Union of all ranks covers the full dataset.
    all_ids = set()
    for ids in ids_per_rank.values():
        all_ids.update(ids)
    assert len(all_ids) == num_cuts

    # Same dp_rank called again → identical sequence (deterministic).
    for dp_rank in (0, dp_world_size - 1):
        dl_again = get_lhotse_dataloader_from_config(
            config=cfg,
            global_rank=dp_rank,
            world_size=dp_world_size,
            dataset=Identity(),
            tokenizer=tokenizer,
        )
        ids_again = [c.id for batch in dl_again for c in batch]
        assert ids_per_rank[dp_rank] == ids_again, f"dp_rank={dp_rank} was not deterministic"


# ---------------------------------------------------------------------------
# Lightning ModelParallelStrategy tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dp_size,tp_size",
    [
        (1, 1),
        (1, 4),
        (4, 1),
        (3, 4),
        (6, 2),
    ],
)
def test_dp_rank_via_lightning_model_parallel(fake_dist, dp_size, tp_size):
    """Verify DP rank using Lightning's ``_setup_device_mesh``.

    Lightning's ``ModelParallelStrategy`` creates a 2D mesh with dimensions
    ``("data_parallel", "tensor_parallel")``.  The DataModule reads DP rank
    via ``dm["data_parallel"].get_local_rank()``.

    Uses ``LocalTensorMode`` to verify every global rank in one process.
    """
    world_size = dp_size * tp_size
    dist.init_process_group(backend="fake", rank=0, world_size=world_size)

    with LocalTensorMode(world_size):
        device_mesh = _setup_device_mesh(dp_size, tp_size, world_size, torch.device("cpu"))

        dp_rank_sym = device_mesh["data_parallel"].get_local_rank()
        dp_world_size = device_mesh["data_parallel"].size()

    assert dp_world_size == dp_size

    rank_to_dp = _extract_rank_mapping(dp_rank_sym, world_size)

    # Correct number of unique dp ranks.
    assert len(set(rank_to_dp.values())) == dp_size

    # All dp_ranks in valid range.
    assert all(0 <= v < dp_size for v in rank_to_dp.values())

    # Ranks sharing the same data_parallel coordinate must have the same
    # dp_rank; different coordinates must differ.
    mesh_tensor = torch.arange(world_size).reshape(dp_size, tp_size)
    for dp_coord in range(dp_size):
        ranks_in_group = mesh_tensor[dp_coord, :].flatten().tolist()
        dp_ranks_in_group = {rank_to_dp[r] for r in ranks_in_group}
        assert (
            len(dp_ranks_in_group) == 1
        ), f"DP group (data_parallel={dp_coord}) maps to multiple dp_ranks: {dp_ranks_in_group}"


def test_lightning_tp_variation_does_not_change_dp_rank(fake_dist):
    """Ranks that differ only in tensor_parallel get the same dp_rank."""
    dp_size, tp_size = 3, 4
    world_size = dp_size * tp_size
    dist.init_process_group(backend="fake", rank=0, world_size=world_size)

    with LocalTensorMode(world_size):
        device_mesh = _setup_device_mesh(dp_size, tp_size, world_size, torch.device("cpu"))
        dp_rank_sym = device_mesh["data_parallel"].get_local_rank()

    rank_to_dp = _extract_rank_mapping(dp_rank_sym, world_size)

    # Mesh layout: (dp=3, tp=4)
    # Rank 0: (dp=0, tp=0)
    # Rank 1: (dp=0, tp=1)  -- differs only in TP
    # Rank 4: (dp=1, tp=0)  -- differs in DP

    assert rank_to_dp[0] == rank_to_dp[1], "TP variation should not change dp_rank"
    assert rank_to_dp[0] == rank_to_dp[2], "TP variation should not change dp_rank"
    assert rank_to_dp[0] == rank_to_dp[3], "TP variation should not change dp_rank"
    assert rank_to_dp[0] != rank_to_dp[4], "DP variation must change dp_rank"


def test_datamodule_get_dp_rank_lightning_model_parallel(fake_dist, tokenizer):
    """DataModule._get_dp_rank() / _get_world_size() with Lightning's 2D mesh."""
    dp_size, tp_size = 3, 4
    world_size = dp_size * tp_size
    dist.init_process_group(backend="fake", rank=0, world_size=world_size)

    # Create mesh outside LocalTensorMode → real int values for fake rank 0
    device_mesh = _setup_device_mesh(dp_size, tp_size, world_size, torch.device("cpu"))

    cfg = DictConfig({"train_ds": {"batch_size": 2}})
    data = DataModule(cfg, tokenizer=tokenizer, dataset=Identity())
    data.trainer = SimpleNamespace(model=SimpleNamespace(device_mesh=device_mesh))

    dp_rank = data._get_dp_rank()
    dp_ws = data._get_world_size()

    assert dp_rank is not None, "_get_dp_rank() returned None (missing return?)"
    assert dp_ws is not None, "_get_world_size() returned None (missing return?)"
    assert dp_rank == 0, f"Fake rank 0 should map to dp_rank 0, got {dp_rank}"
    assert dp_ws == dp_size, f"Expected dp_world_size={dp_size}, got {dp_ws}"
