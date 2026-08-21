# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""DFlash draft training for audio-conditioned SALM targets."""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Sequence

import torch
from lightning import LightningModule
from nemo_automodel.components.speculative.dflash.core import (
    DFlashTrainerModule,
    NoValidAnchorsError,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3 import (
    Qwen3DFlashDraftModel,
    build_target_layer_ids,
)
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo.collections.speechlm2.models.salm import (
    replace_placeholders_and_build_targets,
)
from nemo.collections.speechlm2.parts.cp_helpers import (
    encode_audio_with_cp_distribution,
    get_perception_fsdp_group,
)
from nemo.core.classes.common import safe_instantiate

_DRAFT_CONFIG_MANAGED_KEYS = {
    "architectures",
    "block_size",
    "dflash_config",
    "layer_types",
    "max_window_layers",
    "num_hidden_layers",
    "num_target_layers",
}


def _all_ranks_agree(local_condition: bool, device: torch.device) -> bool:
    """Return whether every distributed rank reports a true condition."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_condition
    flag = torch.tensor([int(local_condition)], device=device, dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _max_rank_value(local_value: int, device: torch.device) -> int:
    """Return the maximum integer reported by any distributed rank."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_value
    value = torch.tensor([local_value], device=device, dtype=torch.int32)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return int(value.item())


def _all_ranks_report_same_value(local_value: int, device: torch.device) -> bool:
    """Return whether every distributed rank reports the same integer."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return True
    extrema = torch.tensor([local_value, -local_value], device=device, dtype=torch.int32)
    torch.distributed.all_reduce(extrema, op=torch.distributed.ReduceOp.MIN)
    return int(extrema[0].item()) == -int(extrema[1].item())


def _preprocessing_signature(batch: dict[str, torch.Tensor]) -> int:
    """Describe rank-local branches that may enter distributed perception."""
    audio_lens = batch.get("audio_lens")
    has_audio = audio_lens is not None and audio_lens.numel() > 0
    has_speaker_targets = batch.get("spk_targets") is not None
    return int(has_audio) | (int(has_speaker_targets) << 1)


def _synchronize_ep_group_before_target_forward(moe_mesh) -> None:
    """Keep rank-local audio preprocessing skew out of DeepEP's timeout."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if moe_mesh is None or "ep" not in moe_mesh.mesh_dim_names:
        return
    ep_mesh = moe_mesh["ep"]
    if ep_mesh.size() > 1:
        torch.distributed.barrier(group=ep_mesh.get_group())


def _has_valid_dflash_anchors(loss_mask: torch.Tensor, block_size: int) -> bool:
    """Return whether this rank can form at least one DFlash anchor."""
    max_anchor = max(loss_mask.shape[1] - block_size, 0)
    return bool((loss_mask[:, : max_anchor + 1] > 0.5).any().item())


def _validate_dflash_parallelism(mesh_context) -> None:
    """Reject parallel layouts whose sequence shards the non-TP DFlash draft cannot consume."""
    unsupported = []
    for name in ("tp", "pp", "cp"):
        size = int(getattr(mesh_context, f"{name}_size", 1))
        if size > 1:
            unsupported.append(f"{name}_size={size}")
    if unsupported:
        raise NotImplementedError(
            "SALM DFlash currently requires tp_size=pp_size=cp_size=1; got " + ", ".join(unsupported)
        )


def _build_draft_config(
    target_config,
    dflash_config: dict,
    block_size: int,
    mask_token_id: int,
) -> tuple[Qwen3Config, list[int]]:
    """Create the Qwen3-shaped DFlash draft config for a SALM target."""
    num_target_layers = int(target_config.num_hidden_layers)
    draft_layers = int(dflash_config.get("draft_num_hidden_layers", 2))
    target_layer_ids = list(
        dflash_config.get("target_layer_ids") or build_target_layer_ids(num_target_layers, draft_layers)
    )
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError("dflash.target_layer_ids must be unique")
    if not target_layer_ids or min(target_layer_ids) < 0 or max(target_layer_ids) >= num_target_layers:
        raise ValueError(f"dflash.target_layer_ids must be within [0, {num_target_layers})")

    architecture = dict(dflash_config.get("draft_model_config") or {})
    managed = sorted(_DRAFT_CONFIG_MANAGED_KEYS.intersection(architecture))
    if managed:
        raise ValueError(f"dflash.draft_model_config cannot override managed keys: {', '.join(managed)}")

    draft_dict = target_config.to_dict()
    draft_dict.update(architecture)
    draft_dict.update(
        {
            "architectures": ["Qwen3DFlashDraftModel"],
            "num_hidden_layers": draft_layers,
            "layer_types": ["full_attention"] * draft_layers,
            "max_window_layers": draft_layers,
            "num_target_layers": num_target_layers,
            "block_size": block_size,
            "dflash_config": {
                "mask_token_id": mask_token_id,
                "target_layer_ids": target_layer_ids,
            },
        }
    )
    draft_config = Qwen3Config.from_dict(draft_dict)
    if draft_config.hidden_size != target_config.hidden_size:
        raise ValueError(
            "The DFlash draft hidden_size must match the frozen target because its embeddings and LM head are reused "
            f"({draft_config.hidden_size} != {target_config.hidden_size})."
        )
    return draft_config, target_layer_ids


def _expand_ids_with_audio(
    input_ids: torch.Tensor,
    replacements: Sequence[torch.Tensor],
    padding_id: int,
    placeholder_id: int,
    mask_token_id: int,
) -> torch.Tensor:
    """Expand audio placeholders to mask-token runs matching fused embeddings."""
    rows = []
    replacement_idx = 0
    for row in input_ids:
        non_padding = (row != padding_id).nonzero(as_tuple=False)
        first_non_padding = int(non_padding[0]) if non_padding.numel() else row.numel()
        row = row[first_non_padding:]
        pieces = []
        for token in row:
            if int(token) == placeholder_id:
                length = replacements[replacement_idx].shape[0]
                replacement_idx += 1
                pieces.append(torch.full((length,), mask_token_id, dtype=row.dtype, device=row.device))
            else:
                pieces.append(token.view(1))
        rows.append(torch.cat(pieces) if pieces else row)
    if replacement_idx != len(replacements):
        raise ValueError(f"Used {replacement_idx} of {len(replacements)} audio replacements")

    max_len = max(row.numel() for row in rows)
    expanded = torch.full((len(rows), max_len), padding_id, dtype=input_ids.dtype, device=input_ids.device)
    for index, row in enumerate(rows):
        expanded[index, -row.numel() :] = row
    return expanded


def _get_consolidated_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Gather an FSDP2 draft into a Hugging Face-saveable rank-zero state dict."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return model.state_dict()

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


class SALMDFlashModule(LightningModule):
    """Train a Qwen3-style DFlash draft from a frozen ``SALMAutomodel`` target."""

    _CHECKPOINT_STATE_PREFIX = "draft_model."
    _REBUILT_STATE_PREFIXES = ("target.", "trainer_module.")

    def __init__(self, target_model: nn.Module, cfg: dict):
        super().__init__()
        self.target = target_model
        self.cfg = cfg
        self.dflash_config = cfg.get("dflash", cfg)
        self.block_size = int(self.dflash_config.get("block_size", 8))
        mask_token_id = self.dflash_config.get("mask_token_id")
        if mask_token_id is None:
            raise ValueError("dflash.mask_token_id must identify an unused token in the target vocabulary")
        self.mask_token_id = int(mask_token_id)
        self.attention_backend = str(self.dflash_config.get("attention_backend", "flex_attention"))
        self.output_dir = self.dflash_config.get("output_dir")
        self.learning_rate = float(self.dflash_config.get("lr", 6e-4))
        self.draft_model = None
        self.trainer_module = None
        self.target_layer_ids = None
        self._draft_dp_group = None
        self._draft_dp_size = 1
        self._partial_val_metrics = defaultdict(list)
        self.register_state_dict_post_hook(self._keep_draft_checkpoint_state)

    def train(self, mode: bool = True):
        """Set the draft's mode while keeping the frozen target in evaluation mode."""
        super().train(mode)
        self.target.eval()
        return self

    @staticmethod
    def _keep_draft_checkpoint_state(module, state_dict, prefix, local_metadata) -> None:
        """Exclude the rebuilt frozen target from resumable checkpoints."""
        draft_prefix = f"{prefix}{module._CHECKPOINT_STATE_PREFIX}"
        for key in tuple(state_dict):
            if not key.startswith(draft_prefix):
                del state_dict[key]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a draft-only checkpoint while allowing the target to be rebuilt."""
        incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [key for key in incompatible.missing_keys if not key.startswith(self._REBUILT_STATE_PREFIXES)]
        if strict and (missing or incompatible.unexpected_keys):
            raise RuntimeError(
                f"Error loading {type(self).__name__}: missing={missing}, unexpected={incompatible.unexpected_keys}"
            )
        return type(incompatible)(missing, incompatible.unexpected_keys)

    def configure_model(self) -> None:
        """Build and shard the frozen SALM target before creating the draft."""
        if self.draft_model is not None:
            return

        strategy = self.trainer.strategy
        distributed_setup = getattr(strategy, "distributed_setup", None)
        if distributed_setup is not None:
            mesh_context = distributed_setup.mesh_context
            _validate_dflash_parallelism(mesh_context)
        self.target._trainer = self.trainer
        self.target.configure_model(
            distributed_setup=distributed_setup,
            activation_checkpointing_perception=getattr(strategy, "activation_checkpointing_perception", False),
        )
        self.target.eval()
        self.target.requires_grad_(False)

        target_config = self.target.llm.config
        vocab_size = int(target_config.vocab_size)
        if not 0 <= self.mask_token_id < vocab_size:
            raise ValueError(f"dflash.mask_token_id={self.mask_token_id} is outside [0, {vocab_size})")
        draft_config, self.target_layer_ids = _build_draft_config(
            target_config, self.dflash_config, self.block_size, self.mask_token_id
        )
        draft_config._attn_implementation = self.attention_backend
        dtype = next(self.target.llm.parameters()).dtype
        self.draft_model = Qwen3DFlashDraftModel(draft_config).to(self.target.device, dtype=dtype)
        if self.dflash_config.get("activation_checkpointing", True):
            self.draft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.trainer_module = DFlashTrainerModule(
            draft_model=self.draft_model,
            target_lm_head=self.target.llm.get_output_embeddings(),
            target_embed_tokens=self.target.llm.get_input_embeddings(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=self.attention_backend,
            num_anchors=int(self.dflash_config.get("num_anchors", 512)),
            max_total_anchors=int(self.dflash_config.get("max_total_anchors", 512)),
            loss_decay_gamma=self.dflash_config.get("loss_decay_gamma", 4.0),
            loss_type=str(self.dflash_config.get("loss_type", "dflash")),
            prefix_weight_base=float(self.dflash_config.get("prefix_weight_base", 0.9)),
            use_fused_linear_ce=bool(self.dflash_config.get("use_fused_linear_ce", True)),
            linear_ce_chunk_size=int(self.dflash_config.get("linear_ce_chunk_size", 256)),
        )

        device_mesh = self.device_mesh
        dim_names = device_mesh.mesh_dim_names
        if "dp_replicate" in dim_names and "dp_shard_cp" in dim_names:
            draft_fsdp_mesh = device_mesh["dp_replicate", "dp_shard_cp"]
        elif "dp_shard_cp" in dim_names:
            draft_fsdp_mesh = device_mesh["dp_shard_cp"]
        else:
            draft_fsdp_mesh = device_mesh["dp"]
        self._draft_dp_size = int(draft_fsdp_mesh.size())
        self._draft_dp_group = draft_fsdp_mesh.get_group() if self._draft_dp_size > 1 else None
        if draft_fsdp_mesh.size() > 1:
            from torch.distributed.fsdp import fully_shard

            self.draft_model = fully_shard(self.draft_model, mesh=draft_fsdp_mesh)

        if any(parameter.requires_grad for parameter in self.target.parameters()):
            raise RuntimeError("The DFlash SALM target must be fully frozen")

    @property
    def device(self):
        return next(self.draft_model.parameters()).device

    def _audio_embeddings(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        spk_targets = batch.get("spk_targets")
        if self.target._uses_parallel_expert_encoder() and spk_targets is None:
            embeddings, lengths = self.target.perception(
                input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
            )
            return [embedding[:length] for embedding, length in zip(embeddings, lengths)]

        device_mesh = getattr(self.target, "_device_mesh", None)
        return encode_audio_with_cp_distribution(
            self.target.perception,
            batch["audios"],
            batch["audio_lens"],
            chunk_size_seconds=self.target.cfg.get("encoder_chunk_size_seconds"),
            sampling_rate=self.target.sampling_rate,
            cp_mesh=None,
            spk_targets=spk_targets,
            fsdp_sync_group=get_perception_fsdp_group(device_mesh),
        )

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.target.cfg.get("packed_sequences", False):
            raise NotImplementedError("SALM DFlash currently requires model.packed_sequences=false")

        input_ids = batch["input_ids"]
        audio_embeddings = self._audio_embeddings(batch)
        text_ids = torch.where(input_ids == self.target.audio_locator_tag_id, 0, input_ids)
        text_embeddings = self.target._embed_tokens(text_ids)
        target_ids = input_ids.where(batch["loss_mask"], -100)
        input_embeddings, target_ids, attention_mask = replace_placeholders_and_build_targets(
            input_ids=input_ids,
            embeds=text_embeddings,
            padding_id=self.target.text_pad_id,
            placeholder_id=self.target.audio_locator_tag_id,
            replacements=audio_embeddings,
            target_ids=target_ids,
        )
        expanded_ids = _expand_ids_with_audio(
            input_ids,
            audio_embeddings,
            self.target.text_pad_id,
            self.target.audio_locator_tag_id,
            self.mask_token_id,
        )
        return {
            # DFlash consumes the full, unshifted token stream. Its block builder
            # gathers a token and its supervision mask at the same sequence index;
            # applying causal-LM input/label shifting here would offset response
            # boundaries and remove the final token from draft supervision.
            "input_ids": expanded_ids,
            "input_embeddings": input_embeddings,
            "attention_mask": attention_mask,
            "loss_mask": target_ids.ne(-100),
        }

    @torch.no_grad()
    def _target_hidden_states(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the frozen audio-conditioned target and concatenate configured layers."""
        if hasattr(self.target.llm, "model") and hasattr(self.target.llm.model, "layers"):
            layer_container = self.target.llm.model.layers
        elif hasattr(self.target.llm, "layers"):
            layer_container = self.target.llm.layers
        elif hasattr(self.target.llm, "transformer") and hasattr(self.target.llm.transformer, "h"):
            layer_container = self.target.llm.transformer.h
        else:
            raise ValueError("Unsupported SALM target structure for DFlash hidden-state capture")
        if isinstance(layer_container, nn.ModuleDict):
            layers = [layer_container[str(index)] for index in range(len(layer_container))]
        else:
            layers = list(layer_container)

        captured = {}
        handles = []

        def make_hook(layer_id: int):
            def hook(_module, _args, output):
                captured[layer_id] = output[0] if isinstance(output, tuple) else output

            return hook

        for layer_id in self.target_layer_ids:
            handles.append(layers[layer_id].register_forward_hook(make_hook(layer_id)))

        forward_kwargs = {
            "inputs_embeds": inputs["input_embeddings"],
            "attention_mask": inputs["attention_mask"],
        }
        forward_parameters = inspect.signature(type(self.target.llm).forward).parameters
        accepts_extra_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in forward_parameters.values()
        )
        for name, value in {
            "output_hidden_states": False,
            "use_cache": False,
            "return_dict": True,
            "compute_logits": False,
        }.items():
            if accepts_extra_kwargs or name in forward_parameters:
                forward_kwargs[name] = value
        try:
            self.target.llm(**forward_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != len(self.target_layer_ids):
            raise RuntimeError(f"Expected {len(self.target_layer_ids)} captured target layers, got {sorted(captured)}")
        return torch.cat([captured[layer_id] for layer_id in self.target_layer_ids], dim=-1)

    def _run_batch(self, batch: dict[str, torch.Tensor]):
        inputs = self._prepare_batch(batch)
        if not _all_ranks_agree(
            _has_valid_dflash_anchors(inputs["loss_mask"], self.block_size),
            inputs["loss_mask"].device,
        ):
            raise NoValidAnchorsError("At least one rank has no valid DFlash anchors")
        _synchronize_ep_group_before_target_forward(getattr(self.trainer.strategy, "moe_mesh", None))
        hidden_states = self._target_hidden_states(inputs)
        return self.trainer_module(
            input_ids=inputs["input_ids"],
            hidden_states=hidden_states,
            loss_mask=inputs["loss_mask"],
        )

    def _globally_normalized_loss(self, metrics) -> torch.Tensor:
        """Weight a local DFlash mean by the global draft-DP loss denominator.

        FSDP averages gradients across its process group. Multiplying the local
        weighted-loss numerator by ``dp_size / global_weight`` therefore yields
        the true global weighted mean even when ranks sample different numbers of
        valid anchors or supervised block positions.
        """
        local_weight = metrics.loss_weight.to(device=metrics.loss.device, dtype=metrics.loss.dtype)
        if not (self._draft_dp_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized()):
            return metrics.loss

        global_weight = local_weight.detach().clone()
        torch.distributed.all_reduce(
            global_weight,
            op=torch.distributed.ReduceOp.SUM,
            group=self._draft_dp_group,
        )
        return metrics.loss * local_weight * self._draft_dp_size / global_weight.clamp_min(1.0e-6)

    def training_step(self, batch, batch_idx):
        batches = list(batch.values()) if isinstance(batch, dict) and "input_ids" not in batch else [batch]
        losses = []
        skip_counts = defaultdict(int)
        num_batches = _max_rank_value(len(batches), self.device)
        for dataset_index in range(num_batches):
            dataset_batch = batches[dataset_index] if dataset_index < len(batches) else None
            if not _all_ranks_agree(dataset_batch is not None, self.device):
                skip_counts["missing_batch"] += 1
                continue
            assert dataset_batch is not None
            signature = _preprocessing_signature(dataset_batch)
            if not _all_ranks_report_same_value(signature, self.device):
                skip_counts["preprocessing_signature"] += 1
                continue
            try:
                metrics = self._run_batch(dataset_batch)
            except NoValidAnchorsError:
                skip_counts["no_valid_anchors"] += 1
                continue
            losses.append(self._globally_normalized_loss(metrics))
            self.log("train/dflash_loss", metrics.loss, on_step=True, prog_bar=True)
            self.log("train/dflash_accuracy", metrics.accuracy, on_step=True)
            self.log("train/accept_len", metrics.accept_len, on_step=True)
        if not losses:
            # Every rank takes the same synchronized skip branches above. Lightning
            # rejects ``None`` from ``training_step`` under distributed automatic
            # optimization, so return a standalone differentiable zero. It has no
            # graph edge to optimizer-owned draft parameters: backward is valid, all
            # draft gradients stay ``None``, and AdamW performs no parameter update.
            self.log("train/dflash_skipped_step", 1.0, on_step=True)
            for reason, count in skip_counts.items():
                self.log(f"train/dflash_skip/{reason}", float(count), on_step=True)
            return torch.zeros((), device=self.device, requires_grad=True)
        self.log("train/dflash_skipped_step", 0.0, on_step=True)
        return torch.stack(losses).mean()

    def on_validation_epoch_start(self) -> None:
        self._partial_val_metrics.clear()

    def validation_step(self, batch, batch_idx) -> None:
        batches = (
            list(batch.items()) if isinstance(batch, dict) and "input_ids" not in batch else [("validation", batch)]
        )
        num_batches = _max_rank_value(len(batches), self.device)
        for dataset_index in range(num_batches):
            dataset_name, dataset_batch = batches[dataset_index] if dataset_index < len(batches) else ("missing", None)
            if not _all_ranks_agree(dataset_batch is not None, self.device):
                continue
            assert dataset_batch is not None
            signature = _preprocessing_signature(dataset_batch)
            if not _all_ranks_report_same_value(signature, self.device):
                continue
            try:
                metrics = self._run_batch(dataset_batch)
            except NoValidAnchorsError:
                continue
            # Counts can exceed float32's exact-integer range over a long epoch
            # with 512 anchors. Accumulate all additive validation statistics in
            # float64 so a single stacked all-reduce remains exact for counts.
            metric_dtype = torch.float64
            metric_device = metrics.loss.device
            self._partial_val_metrics[dataset_name].append(
                torch.stack(
                    [
                        metrics.loss.detach().to(dtype=metric_dtype) * metrics.loss_weight.to(metric_dtype),
                        metrics.loss_weight.to(dtype=metric_dtype, device=metric_device),
                        metrics.correct_tokens.to(dtype=metric_dtype, device=metric_device),
                        metrics.valid_tokens.to(dtype=metric_dtype, device=metric_device),
                        metrics.accept_len_sum.to(dtype=metric_dtype, device=metric_device),
                        metrics.valid_blocks.to(dtype=metric_dtype, device=metric_device),
                    ]
                )
            )

    def on_validation_epoch_end(self) -> None:
        all_sums = []
        for dataset_name, partial_metrics in self._partial_val_metrics.items():
            if not partial_metrics:
                continue
            metric_sums = torch.stack(partial_metrics).sum(dim=0)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(metric_sums, op=torch.distributed.ReduceOp.SUM)
            all_sums.append(metric_sums)
            self._log_validation_metrics(metric_sums, suffix=f"/{dataset_name}")
        if all_sums:
            self._log_validation_metrics(torch.stack(all_sums).sum(dim=0))
        self._partial_val_metrics.clear()

    def _log_validation_metrics(self, metric_sums: torch.Tensor, suffix: str = "") -> None:
        loss_sum, loss_weight, correct, valid, accept_sum, valid_blocks = metric_sums
        accuracy = correct / valid.clamp_min(1)
        self.log(
            f"val/dflash_loss{suffix}",
            loss_sum / loss_weight.clamp_min(1),
            on_epoch=True,
        )
        self.log(f"val/dflash_accuracy{suffix}", accuracy, on_epoch=True)
        self.log(
            f"val/accept_len{suffix}",
            accept_sum / valid_blocks.clamp_min(1),
            on_epoch=True,
        )
        if not suffix:
            # Preserve the existing SALM recipe's ModelCheckpoint monitor without
            # changing non-DFlash logging or requiring a DFlash-only exp_manager.
            self.log("val_acc", accuracy, on_epoch=True)

    def configure_optimizers(self):
        optimizer_config = self.dflash_config.get("optimizer")
        if optimizer_config is None:
            return torch.optim.AdamW(self.draft_model.parameters(), lr=self.learning_rate)
        optimizer = safe_instantiate(
            optimizer_config,
            params=self.draft_model.parameters(),
            _convert_="all",
        )
        scheduler_config = self.dflash_config.get("lr_scheduler")
        if scheduler_config is None:
            return optimizer
        scheduler = safe_instantiate(scheduler_config, optimizer=optimizer, _convert_="all")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_end(self) -> None:
        if not self.output_dir:
            return
        state_dict = _get_consolidated_model_state_dict(self.draft_model)
        if self.trainer.is_global_zero:
            self.draft_model.save_pretrained(self.output_dir, state_dict=state_dict)
