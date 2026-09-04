# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from nemo.collections.speechlm2.parts.hf_hub import LLM_BACKBONE_DIR
from nemo.core.classes.common import safe_instantiate
from nemo.core.config import hydra_runner
from nemo.utils.dtype import str_to_dtype
from nemo.utils.model_utils import import_class_by_path


@dataclass
class HfExportConfig:
    # Name of the model class to be imported, e.g. nemo.collections.speechlm2.models.SALM
    class_path: str

    # Path to PyTorch Lightning checkpoint file (normal ckpt) or directory (distributed ckpt)
    ckpt_path: str

    # Path to the experiment's config, used to instantiate the model class.
    ckpt_config: str

    # Path where we should save the HuggingFace Hub compatible checkpoint
    output_dir: str

    # Dtype used for stored parameters
    dtype: str = "bfloat16"


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if Path(checkpoint_path).is_dir():
        from torch.distributed.checkpoint import load

        state_dict = {"state_dict": model.state_dict()}
        load(state_dict, checkpoint_id=checkpoint_path)
        model.load_state_dict(state_dict["state_dict"])
    else:
        ckpt_data = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt_data["state_dict"])


def _adapt_strategy_for_conversion_world(strategy_cfg: dict, world_size: int) -> dict:
    """Make an HSDP training mesh valid for the smaller conversion world."""
    strategy_cfg = deepcopy(strategy_cfg)
    tp_size = int(strategy_cfg.get("tp_size") or 1)
    cp_size = int(strategy_cfg.get("cp_size") or 1)
    pp_size = int(strategy_cfg.get("pp_size") or 1)
    non_dp_size = tp_size * cp_size * pp_size
    if world_size % non_dp_size != 0:
        return strategy_cfg

    conversion_dp_size = world_size // non_dp_size
    replicate_size = int(strategy_cfg.get("dp_replicate_size") or 1)
    if replicate_size > 1 and (conversion_dp_size % replicate_size != 0 or replicate_size >= conversion_dp_size):
        strategy_cfg["dp_replicate_size"] = 1
    return strategy_cfg


def setup_distributed_from_config(strategy_cfg: dict) -> Any:
    """Initialize torch.distributed and create a device mesh from a Hydra strategy config.

    Instantiates the strategy from the trainer config dict (as found in the
    experiment YAML), initializes the process group, resolves automodel
    configs, and calls :meth:`strategy.create_device_mesh`.

    Returns:
        An :class:`AutomodelParallelStrategy` with device_mesh ready.
    """
    from nemo.utils.trainer_utils import _resolve_automodel_configs

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    strategy_cfg = _adapt_strategy_for_conversion_world(strategy_cfg, dist.get_world_size())
    strategy = safe_instantiate(strategy_cfg)
    _resolve_automodel_configs(strategy)
    strategy.create_device_mesh()
    return strategy


def consolidate_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Gather a full (non-sharded) state dict from a model with DTensor parameters."""
    from torch.distributed.tensor import DTensor

    consolidated = {}
    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            consolidated[key] = value.full_tensor().cpu()
        else:
            consolidated[key] = value.cpu()
    return consolidated


def _canonical_torch_dtype_name(dtype: str | torch.dtype) -> str:
    """Return the PyTorch dtype name accepted by Transformers configs."""
    return str(str_to_dtype(dtype)).replace("torch.", "")


def _hf_export_config(model: torch.nn.Module, dtype: str | torch.dtype) -> dict[str, Any]:
    """Build the exported root config without mutating the training config."""
    config = OmegaConf.to_container(model.cfg) if isinstance(model.cfg, DictConfig) else deepcopy(model.cfg)
    # Remote-code trust is a runtime security decision. Do not persist a
    # training-time opt-in in checkpoints that may be loaded by another user.
    config.pop("trust_remote_code", None)
    pe_encoder_path = config.get("pe_encoder_path", None)
    pe_encoder_config = config.get("pe_encoder_config", None)
    if pe_encoder_path not in (None, "", False) or pe_encoder_config not in (
        None,
        {},
        "",
        False,
    ):
        pe_encoder = getattr(getattr(model, "perception", None), "encoder", None)
        bundle_config = getattr(pe_encoder, "_bundle_config", None)
        if bundle_config is None:
            raise RuntimeError(
                "Cannot export ParallelExpertEncoder portably: the mounted perception encoder has no architecture bundle config."
            )
        bundle_config = OmegaConf.to_container(bundle_config, resolve=True)
        for obsolete_key in (
            "align_diarization_output_resolution",
            "asr_chunk_size_seconds",
            "diar_chunk_size_seconds",
        ):
            bundle_config.pop(obsolete_key, None)
        # Persist runtime overrides rather than the initialization bundle's
        # defaults. The consolidated root state dict supplies all weights.
        for config_key, attr_name in (
            ("asr_normalize_type", "asr_normalize_type"),
            ("chunk_size_seconds", "chunk_size_seconds"),
            ("diar_normalize_type", "diar_normalize_type"),
            ("frame_shift_seconds", "frame_shift_seconds"),
            ("missing_rttm_target", "missing_rttm_target"),
            ("speaker_feature_mode", "speaker_feature_mode"),
            ("speaker_activity_threshold", "speaker_activity_threshold"),
            ("spk_kernel_scale", "spk_kernel_scale"),
        ):
            if hasattr(pe_encoder, attr_name):
                bundle_config[config_key] = getattr(pe_encoder, attr_name)
        if hasattr(pe_encoder, "speaker_feature_mode"):
            bundle_config["speaker_feature_config_version"] = 1
        config["pe_encoder_config"] = bundle_config
        config["pe_encoder_path"] = None
        config.pop("pe_encoder_overrides", None)

    speaker_encoder_cfg = config.get("speaker_encoder", None)
    if speaker_encoder_cfg not in (None, {}, "", False):
        dual = getattr(getattr(model, "perception", None), "encoder", None)
        auxiliary_encoder_config = getattr(dual, "auxiliary_encoder_config", None)
        if auxiliary_encoder_config is None:
            raise RuntimeError(
                "Cannot export IndependentDualEncoder portably: the mounted auxiliary encoder "
                "has no inline architecture config."
            )
        config["speaker_encoder"] = {
            "encoder_config": deepcopy(auxiliary_encoder_config),
            "frozen": bool(getattr(dual, "freeze_auxiliary", True)),
            "chunk_size_seconds": getattr(dual, "auxiliary_chunk_size_seconds", None),
            "asr_chunk_size_seconds": getattr(dual, "asr_chunk_size_seconds", None),
        }
    dtype_name = _canonical_torch_dtype_name(dtype)
    config["dtype"] = dtype_name
    config["torch_dtype"] = dtype_name

    return config


def save_hf_checkpoint(model: torch.nn.Module, state_dict: dict, cfg: HfExportConfig) -> None:
    """Save a consolidated state dict and model config in HuggingFace Hub format."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dtype = str_to_dtype(cfg.dtype)
    forced_dtypes = {}
    state_dict_adapter = getattr(getattr(model, "llm", None), "state_dict_adapter", None)
    if callable(getattr(state_dict_adapter, "forced_hf_dtype_mapping", None)):
        forced_dtypes = state_dict_adapter.forced_hf_dtype_mapping(state_dict)
    state_dict = {
        key: value.to(torch.float32 if forced_dtypes.get(key) == "F32" else target_dtype)
        for key, value in state_dict.items()
    }

    config = _hf_export_config(model, cfg.dtype)
    save_file(state_dict, output_dir / "model.safetensors")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    save_llm_backbone_config(model, output_dir)


def save_llm_backbone_config(model: torch.nn.Module, output_dir: str | Path) -> None:
    """Save the original LLM config separately from the NeMo wrapper config."""
    llm_config = getattr(getattr(model, "llm", None), "config", None)
    if llm_config is None:
        return

    llm_backbone_dir = Path(output_dir) / LLM_BACKBONE_DIR
    llm_backbone_dir.mkdir(parents=True, exist_ok=True)
    llm_config.save_pretrained(str(llm_backbone_dir))


def _detect_vllm_architecture(model_cfg: dict) -> tuple[str, int]:
    """Determine the vLLM plugin model class and backbone vocabulary size.

    The SALM plugin registers a single architecture name and selects between
    transformer and hybrid backends at instantiation time, so this function
    verifies the backbone config is reachable and returns the unified name
    plus the embedding-table vocabulary bound. The hybrid-vs-transformer split
    is handled inside the plugin.

    Raises:
        ValueError: If the HF config cannot be loaded, has no architecture, or
            declares an invalid vocabulary size.
    """
    pretrained_llm = model_cfg.get("pretrained_llm", "")
    try:
        from transformers import AutoConfig

        llm_cfg = AutoConfig.from_pretrained(pretrained_llm, trust_remote_code=True)
    except Exception as e:
        raise ValueError(
            f"Could not load HF config for pretrained_llm={pretrained_llm!r}: {e}. "
            f"Fix the 'pretrained_llm' field or ensure HF access during conversion."
        ) from e

    archs = getattr(llm_cfg, "architectures", [])
    if not archs:
        raise ValueError(f"HF config for {pretrained_llm!r} has empty 'architectures'.")
    vocab_size = getattr(llm_cfg, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"HF config for {pretrained_llm!r} has invalid 'vocab_size': {vocab_size!r}.")

    return "NeMoSpeechLMForConditionalGeneration", vocab_size


def prepare_for_vllm(output_dir: str, model_cfg: dict) -> None:
    """Patch a saved checkpoint to be vLLM-ready.

    Adds tokenizer (with audio token and chat template), patches config.json
    with model_type/architectures, and writes generation_config.json.

    Args:
        output_dir: Path to the HuggingFace checkpoint directory.
        model_cfg: Model config dict (from experiment YAML).

    Raises:
        ValueError: If required model metadata is missing, or the tokenizer's
            audio token does not fit the SpeechLM embedding table.
    """
    from transformers import AutoTokenizer

    from nemo.collections.speechlm2.vllm.salm.config import _SPEECHLM_EMBED_EXTRA_ROWS
    from nemo.utils import logging as LOG

    output_dir = Path(output_dir)
    pretrained_llm = model_cfg.get("pretrained_llm", "")
    if not pretrained_llm:
        raise ValueError("model config has no 'pretrained_llm'; cannot load tokenizer for vLLM")

    # ``model.audio_locator_tag`` is the SoT for the audio placeholder;
    # fail loud rather than default, since a mismatch is silent at inference.
    audio_token = model_cfg.get("audio_locator_tag")
    if not audio_token:
        raise ValueError("model config has no 'audio_locator_tag' (set it in the training YAML).")

    # 1. Patch config.json (arch, model_type, audio_locator_tag for vLLM plugin).
    arch_model_cfg = dict(model_cfg)
    llm_backbone_dir = output_dir / LLM_BACKBONE_DIR
    llm_backbone_config_path = llm_backbone_dir / "config.json"
    llm_backbone_config = None
    if llm_backbone_config_path.exists():
        arch_model_cfg["pretrained_llm"] = str(llm_backbone_dir)
        llm_backbone_config = json.loads(llm_backbone_config_path.read_text())
    arch, base_vocab_size = _detect_vllm_architecture(arch_model_cfg)
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "nemo_speechlm"
    config["architectures"] = [arch]
    config["audio_locator_tag"] = audio_token
    if llm_backbone_config is not None:
        # Keep the export portable while making the bundled config authoritative.
        # NeMo's HF loader resolves this relative marker to a cached local path;
        # the vLLM config consumes the embedded copy without another Hub lookup.
        config["pretrained_llm"] = LLM_BACKBONE_DIR
        config["llm_config"] = llm_backbone_config
    else:
        config.pop("llm_config", None)
    config.pop("audio_token_index", None)

    # 2. Save tokenizer (backbone chat_template carries over via save_pretrained)
    existing = [
        f.name
        for f in output_dir.iterdir()
        if f.name in ("tokenizer_config.json", "tokenizer.json", "generation_config.json")
    ]
    if existing:
        LOG.info("Overwriting existing files in %s: %s", output_dir, existing)
    tokenizer_src = model_cfg.get("tokenizer_path") or pretrained_llm
    tok = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if audio_token not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [audio_token]})
    audio_token_id = tok.get_vocab().get(audio_token)
    if isinstance(audio_token_id, bool) or not isinstance(audio_token_id, int) or audio_token_id < 0:
        raise ValueError(f"Tokenizer did not assign a valid ID to audio token {audio_token!r}.")
    padded_vocab_size = base_vocab_size + _SPEECHLM_EMBED_EXTRA_ROWS
    if audio_token_id >= padded_vocab_size:
        raise ValueError(
            f"Audio token ID {audio_token_id} is outside the SpeechLM embedding table with "
            f"{padded_vocab_size} rows. Reduce the tokenizer's added-token count before training/export."
        )
    pad_token = model_cfg.get("pad_token", None)
    if pad_token:
        if pad_token not in tok.get_vocab():
            raise ValueError(f"model pad_token={pad_token!r} is absent from the exported tokenizer vocabulary.")
        tok.pad_token = pad_token

    # Preserve the model's training-time padding contract explicitly. Loading
    # the backbone tokenizer alone can silently restore its serving default
    # (for Nemotron 3.5, EOS) even when SALM trained with <unk> as PAD.
    if tok.pad_token_id is not None:
        config["pad_token_id"] = tok.pad_token_id
    if tok.eos_token_id is not None:
        config["eos_token_id"] = [tok.eos_token_id]

    # 4. Minimal generation_config.json (token termination/padding only;
    #    sampling params belong on
    #    the server, not baked into the checkpoint).
    gen_cfg = {"eos_token_id": [tok.eos_token_id]}
    if tok.pad_token_id is not None:
        gen_cfg["pad_token_id"] = tok.pad_token_id

    # Build every tokenizer-side artifact in a sibling staging directory. A
    # caught validation/serialization ``ValueError`` must leave the original
    # HF checkpoint untouched instead of advertising a half-prepared vLLM
    # model. Once staging succeeds, publish each file atomically and write the
    # authoritative root config last.
    with TemporaryDirectory(prefix=f".{output_dir.name}-vllm-", dir=output_dir.parent) as staging:
        staging_dir = Path(staging)
        artifacts_dir = staging_dir / "artifacts"
        artifacts_dir.mkdir()
        tok.save_pretrained(str(artifacts_dir))
        # Newer transformers splits long chat_template into a separate
        # ``chat_template.jinja`` file; inline it back and drop the file.
        tok_cfg_path = artifacts_dir / "tokenizer_config.json"
        tok_cfg = json.loads(tok_cfg_path.read_text())
        jinja_file = artifacts_dir / "chat_template.jinja"
        if jinja_file.exists():
            jinja_from_file = jinja_file.read_text()
            if jinja_from_file.strip():
                tok_cfg["chat_template"] = jinja_from_file
            jinja_file.unlink()
        # Normalize to dict form; transformers writes a list which HF loaders reject.
        tok_cfg["extra_special_tokens"] = {"audio_token": audio_token}
        # Some NeMo containers save a proprietary ``TokenizersBackend`` class
        # unknown to HF; the underlying tokenizer.json is standard, so force
        # the universal base class.
        tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")
        (artifacts_dir / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2) + "\n")

        staged_config_path = staging_dir / "config.json"
        staged_config_path.write_text(json.dumps(config, indent=2) + "\n")
        staged_artifacts = list(artifacts_dir.iterdir())
        if any(not path.is_file() for path in staged_artifacts):
            raise ValueError("Tokenizer export unexpectedly produced a directory; refusing a partial publication.")
        if any(path.name == "config.json" for path in staged_artifacts):
            raise ValueError(
                "Tokenizer export unexpectedly produced config.json; refusing to replace the model config early."
            )

        # Back up every live publication target so an I/O error during the
        # multi-file commit can restore the exact pre-call checkpoint. The
        # staged root config is replaced atomically and strictly last, so an
        # interrupted first-time conversion never advertises incomplete
        # tokenizer artifacts as a vLLM model.
        live_jinja_path = output_dir / "chat_template.jinja"
        publication_targets = [output_dir / path.name for path in staged_artifacts]
        publication_targets.extend([live_jinja_path, config_path])
        backups_dir = staging_dir / "backups"
        backups_dir.mkdir()
        original_backups = {}
        for target in publication_targets:
            if target.exists():
                backup = backups_dir / target.name
                copy2(target, backup)
                original_backups[target] = backup
            else:
                original_backups[target] = None

        try:
            for staged_path in staged_artifacts:
                staged_path.replace(output_dir / staged_path.name)
            live_jinja_path.unlink(missing_ok=True)
            staged_config_path.replace(config_path)
        except BaseException:
            for target, backup in original_backups.items():
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    copy2(backup, target)
            raise


def _try_prepare_for_vllm(output_dir: str, model_cfg: dict) -> None:
    """Run vLLM prep; on ``ValueError``, warn and keep the HF-only output.

    Backward compat for callers that never needed vLLM (e.g., NeMo SALM).
    """
    from nemo.utils import logging as LOG

    try:
        prepare_for_vllm(output_dir, model_cfg)
    except ValueError as e:
        LOG.warning(
            "Checkpoint saved as HF-only; vLLM prep skipped: %s. "
            "The checkpoint is still loadable by NeMo SALM and plain HF, but "
            "is NOT vLLM-ready until prep succeeds.",
            e,
        )


def _uses_automodel_parallel(strategy_cfg: dict) -> bool:
    """Check if the strategy config targets AutomodelParallelStrategy."""
    target = strategy_cfg.get("_target_", "")
    return "AutomodelParallelStrategy" in target


@hydra_runner(config_name="HfExportConfig", schema=HfExportConfig)
def main(cfg: HfExportConfig) -> None:
    """
    Read PyTorch Lightning checkpoint and export the model to HuggingFace Hub format.
    The resulting model can be then initialized via ModelClass.from_pretrained(path).

    Also supports distributed checkpoints for models trained with FSDP2/TP
    via AutomodelParallelStrategy.  Parallelism sizes (tp_size, pp_size, etc.)
    are read automatically from the ``trainer.strategy`` section of the
    experiment config (``ckpt_config``).

    When the checkpoint is a distributed checkpoint (a directory), launch this
    script via ``torchrun`` with the same number of GPUs used for training.

    Examples:
        # Single-file checkpoint — original SALM (HF Transformers backend):
        python to_hf.py \\
            class_path=nemo.collections.speechlm2.models.SALM \\
            ckpt_path=/path/to/checkpoint.ckpt \\
            ckpt_config=/path/to/config.yaml \\
            output_dir=/path/to/hf_output

        # Single-file checkpoint — SALMAutomodel (NeMo Automodel backend):
        python to_hf.py \\
            class_path=nemo.collections.speechlm2.models.SALMAutomodel \\
            ckpt_path=/path/to/checkpoint.ckpt \\
            ckpt_config=/path/to/config.yaml \\
            output_dir=/path/to/hf_output

        # Distributed checkpoint (parallelism read from config automatically):
        torchrun --nproc-per-node=8 to_hf.py \\
            class_path=nemo.collections.speechlm2.models.SALMAutomodel \\
            ckpt_path=/path/to/distributed_ckpt_dir \\
            ckpt_config=/path/to/config.yaml \\
            output_dir=/path/to/hf_output
    """
    if not Path(cfg.ckpt_path).exists():
        raise RuntimeError(f"No such file or directory: {cfg.ckpt_path}")

    full_cfg = OmegaConf.to_container(OmegaConf.load(cfg.ckpt_config), resolve=True)
    model_cfg = full_cfg["model"]
    audio_token_estimator = full_cfg.get("data", {}).get("train_ds", {}).get("audio_token_estimator")
    if audio_token_estimator is not None:
        # The vLLM prompt processor must reserve exactly as many audio
        # placeholders as the checkpoint's encoder emits.
        model_cfg["audio_token_estimator"] = audio_token_estimator
    model_cfg["torch_dtype"] = _canonical_torch_dtype_name(cfg.dtype)
    cls = import_class_by_path(cfg.class_path)

    strategy_cfg = full_cfg.get("trainer", {}).get("strategy", {})

    _is_torchrun = "RANK" in os.environ
    if _is_torchrun and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    is_distributed = (
        _is_torchrun
        and Path(cfg.ckpt_path).is_dir()
        and _uses_automodel_parallel(strategy_cfg)
        and dist.get_world_size() > 1
    )

    if is_distributed:
        strategy = setup_distributed_from_config(strategy_cfg)

        # Don't call configure_model() inside __init__ — we set the distributed setup first.
        model_cfg["init_configure_model"] = False
        model_cfg["pretrained_weights"] = False
        model = cls(model_cfg)
        model.configure_model(distributed_setup=strategy.distributed_setup)

        load_checkpoint(model, cfg.ckpt_path)

        # Consolidate DTensors to regular tensors and save on rank 0.
        consolidated = consolidate_state_dict(model)
        if dist.get_rank() == 0:
            save_hf_checkpoint(model, consolidated, cfg)
            _try_prepare_for_vllm(cfg.output_dir, model_cfg)

        dist.barrier()
        dist.destroy_process_group()
    else:
        model_cfg["init_configure_model"] = True
        model_cfg["pretrained_weights"] = False
        model = cls(model_cfg)
        load_checkpoint(model, cfg.ckpt_path)
        model = model.to(str_to_dtype(cfg.dtype))
        model.save_pretrained(cfg.output_dir, config=_hf_export_config(model, cfg.dtype))
        save_llm_backbone_config(model, cfg.output_dir)
        _try_prepare_for_vllm(cfg.output_dir, model_cfg)


if __name__ == "__main__":
    main()
