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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from nemo.core.config import hydra_runner
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


def setup_distributed_from_config(strategy_cfg: dict) -> Any:
    """Initialize torch.distributed and create a device mesh from a Hydra strategy config.

    Instantiates the strategy from the trainer config dict (as found in the
    experiment YAML), initializes the process group, resolves automodel
    configs, and calls :meth:`strategy.create_device_mesh`.

    Returns:
        An :class:`AutomodelParallelStrategy` with device_mesh ready.
    """
    import hydra

    from nemo.utils.trainer_utils import _resolve_automodel_configs

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    strategy = hydra.utils.instantiate(strategy_cfg)
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


def save_hf_checkpoint(model: torch.nn.Module, state_dict: dict, cfg: HfExportConfig) -> None:
    """Save a consolidated state dict and model config in HuggingFace Hub format."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dtype = getattr(torch, cfg.dtype)
    state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}

    save_file(state_dict, output_dir / "model.safetensors")

    config = OmegaConf.to_container(model.cfg) if isinstance(model.cfg, DictConfig) else model.cfg
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


_HYBRID_ARCHITECTURES = {"NemotronHForCausalLM", "NemotronHybridForCausalLM"}


def _detect_vllm_architecture(model_cfg: dict) -> str:
    """Determine the vLLM plugin model class from the pretrained LLM backbone.

    Raises:
        ValueError: if the HF config can't be loaded or has no 'architectures'.
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

    if set(archs) & _HYBRID_ARCHITECTURES:
        return "NeMoSpeechLMHybridForConditionalGeneration"
    return "NeMoSpeechLMForConditionalGeneration"


def prepare_for_vllm(output_dir: str, model_cfg: dict) -> None:
    """Patch a saved checkpoint to be vLLM-ready.

    Adds tokenizer (with audio token and chat template), patches config.json
    with model_type/architectures, and writes generation_config.json.

    Args:
        output_dir: Path to the HuggingFace checkpoint directory.
        model_cfg: Model config dict (from experiment YAML).

    Raises:
        ValueError: If ``pretrained_llm`` or ``audio_locator_tag`` is missing.
    """
    from transformers import AutoTokenizer

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
    arch = _detect_vllm_architecture(model_cfg)
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "nemo_speechlm"
    config["architectures"] = [arch]
    config["audio_locator_tag"] = audio_token
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    # 2. Save tokenizer (backbone chat_template carries over via save_pretrained)
    existing = [
        f.name
        for f in output_dir.iterdir()
        if f.name in ("tokenizer_config.json", "tokenizer.json", "generation_config.json")
    ]
    if existing:
        LOG.info("Overwriting existing files in %s: %s", output_dir, existing)
    tok = AutoTokenizer.from_pretrained(pretrained_llm, trust_remote_code=True)
    if audio_token not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [audio_token]})
    tok.save_pretrained(str(output_dir))
    # Newer transformers splits long chat_template into a separate
    # ``chat_template.jinja`` file; inline it back and drop the file.
    tok_cfg_path = output_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text())
    jinja_file = output_dir / "chat_template.jinja"
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

    # 4. Minimal generation_config.json (EOS only; sampling params belong on
    #    the server, not baked into the checkpoint).
    gen_cfg = {"eos_token_id": [tok.eos_token_id]}
    (output_dir / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2) + "\n")


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
    model_cfg["torch_dtype"] = cfg.dtype
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

        # Don't call configure_model() inside __init__ — we set device_mesh first.
        model_cfg["init_configure_model"] = False
        model = cls(model_cfg)
        model.configure_model(
            device_mesh=strategy.device_mesh,
            distributed_config=strategy.distributed_config,
            moe_config=strategy.moe_config,
            moe_mesh=strategy.moe_mesh,
        )
        model_cfg["pretrained_weights"] = False

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
        model = cls(model_cfg)
        load_checkpoint(model, cfg.ckpt_path)
        model = model.to(getattr(torch, cfg.dtype))
        model_cfg["pretrained_weights"] = False
        model.save_pretrained(cfg.output_dir)
        _try_prepare_for_vllm(cfg.output_dir, model_cfg)


if __name__ == "__main__":
    main()
