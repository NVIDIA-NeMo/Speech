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
"""Streaming STT SpeechLM with a NeMo Automodel LLM backend.

:class:`StreamingSTTModelAutomodel` is to
:class:`~nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModel`
what :class:`~nemo.collections.speechlm2.models.salm_automodel.SALMAutomodel` is
to :class:`~nemo.collections.speechlm2.models.salm.SALM`: the training and
inference recipe is unchanged, only the LLM backend and the parallelism story
differ.

The differences relative to the HuggingFace-backed parent class:

* **Deferred construction.** ``__init__`` only builds the tokenizer (and the
  optional forced aligner); the LLM and the perception module are created in
  :meth:`configure_model`, which the Lightning strategy calls *after* the device
  mesh exists. This lets Automodel shard the LLM while loading it, so each rank
  only ever materializes its own shard.
* **Parallelism.** FSDP2 / HSDP (and EP for MoE backbones) via
  :class:`~nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy`
  instead of DDP. The perception module and the optional aux chunk classifier
  are sharded with ``fully_shard`` on the same mesh Automodel uses for the LLM.
  Tensor and context parallelism are rejected at ``on_fit_start`` — see
  :meth:`_validate_parallelism_compatibility`.
* **Embeddings stay inside the LLM.** The parent moves ``embed_tokens`` to the
  top level to dodge FSDP/TP conflicts; Automodel needs it in place for its own
  parallelization plan and state-dict adapters, so here it is accessed through
  the ``embed_tokens`` property and embedded via ``F.embedding`` on the
  all-gathered weight (see :meth:`_embed_tokens`).
* **Vocabulary growth.** ``resize_token_embeddings`` cannot run on a sharded
  embedding table, so added special tokens (``<blank>`` / ``write`` /
  ``end_of_audio``) reuse the spare rows most checkpoints already have
  (``config.vocab_size > len(tokenizer)``); see :meth:`_sync_llm_vocab_size`.
* **LoRA** is applied by Automodel's own PEFT implementation (config keys
  ``dim`` / ``alpha`` / ``dropout`` / ``target_modules``), not HuggingFace PEFT.

Everything else — the streaming data recipe, the losses, the fixed / dynamic /
offline generation paths — is inherited unchanged from ``StreamingSTTModel``.
"""

from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning.fabric.plugins.precision.utils import _DtypeContextManager
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Partial
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataset
from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel, StreamingSTTModelConfig
from nemo.collections.speechlm2.parts.automodel_lora import (
    LORA_PARAM_PATTERN,
    ensure_lora_trainable,
    make_peft_config,
    maybe_install_lora,
)
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_automodel_llm,
    maybe_load_pretrained_models,
    setup_perception,
)
from nemo.collections.speechlm2.parts.utils import freeze_module, to_dataclass, unfreeze_module
from nemo.utils import logging

# Fields of ``transformers.modeling_outputs.CausalLMOutputWithPast``; used to
# normalize LLM outputs that come back as a plain dict.
_CAUSAL_LM_OUTPUT_FIELDS = ("loss", "logits", "past_key_values", "hidden_states", "attentions")

# HuggingFace-PEFT LoRA keys. Automodel's ``PeftConfig`` uses different names
# (``dim`` / ``alpha`` / ``dropout``), so a config copy-pasted from the HF-backed
# recipe would fail deep inside Automodel with an unhelpful ``TypeError``.
_HF_PEFT_LORA_KEYS = ("r", "lora_alpha", "lora_dropout", "task_type", "peft_type", "modules_to_save")


class StreamingSTTModelAutomodel(StreamingSTTModel):
    """Streaming STT SpeechLM whose LLM is loaded and parallelized by NeMo Automodel.

    Args:
        cfg: Model configuration as a plain Python dict (required for
            hyperparameter serialization in PTL checkpoints). Accepts every key
            of :class:`~nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModelConfig`
            plus the Automodel-specific ones: ``lora`` (Automodel ``PeftConfig``
            format), ``automodel_backend``, ``sdpa_method``, ``compile``,
            ``aux_loss_coeff``, ``train_gate``, ``moe_metrics``,
            ``trust_remote_code``, ``init_from_checkpoint`` and
            ``init_configure_model``.
        forced_aligner: Optional forced aligner for online alignment.
        data_cfg: Training dataset config (required with ``forced_aligner``).
        val_data_cfg: Optional validation dataset config; falls back to ``data_cfg``.
        dataset_cls: Dataset class used when online forced alignment is enabled.
    """

    def __init__(
        self,
        cfg: dict,
        forced_aligner=None,
        data_cfg: Optional[DictConfig] = None,
        val_data_cfg: Optional[DictConfig] = None,
        dataset_cls=StreamingSTTDataset,
    ) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to StreamingSTTModelAutomodel as a Python dict to support hyperparameter "
            f"serialization in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        # NOTE: StreamingSTTModel.__init__ is deliberately bypassed — it builds
        # the LLM and the perception module eagerly, whereas the Automodel path
        # must defer both to configure_model() so they can be created directly
        # on the device mesh. The pieces of the parent __init__ that do not
        # depend on the LLM are reused through the extracted helpers below.
        # ``super(StreamingSTTModel, self)`` (rather than a direct
        # ``LightningModule.__init__(self)`` call) is deliberate: referencing
        # ``super`` gives this frame the ``__class__`` cell that Lightning's
        # ``save_hyperparameters()`` needs to discover the init args — without
        # it ``self.hparams`` comes out empty and checkpoints lose ``cfg``.
        super(StreamingSTTModel, self).__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.core_cfg: StreamingSTTModelConfig = to_dataclass(StreamingSTTModelConfig, cfg)
        self._normalize_chunk_size()

        self.tokenizer = AutoTokenizer(self.core_cfg.pretrained_llm, use_fast=True)
        self.llm = None  # populated by configure_model()
        self.perception = None  # populated by configure_model()

        self._use_fsdp = False
        self._configured_with_mesh = False
        self._device_mesh = None
        self._current_batch_idx = 0

        self._register_special_tokens()
        self._validate_lora_cfg()
        self._setup_forced_aligner(forced_aligner, data_cfg, val_data_cfg, dataset_cls)

        if self.cfg.get("init_configure_model", False):
            self.configure_model()

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_lora_cfg(self) -> None:
        """Fail fast on a HuggingFace-PEFT-style ``lora`` block.

        The parent class installs LoRA through HuggingFace PEFT
        (``r`` / ``lora_alpha`` / ``lora_dropout``); Automodel has its own
        implementation with different key names. Silently forwarding the HF keys
        raises a ``TypeError`` several frames deep inside Automodel, so translate
        it into an actionable message here.
        """
        if "lora" not in self.cfg or not self.cfg.lora:
            return
        offending = [k for k in _HF_PEFT_LORA_KEYS if k in self.cfg.lora]
        if offending:
            raise ValueError(
                f"model.lora contains HuggingFace-PEFT keys {offending}, but "
                f"{type(self).__name__} installs LoRA with NeMo Automodel's PEFT implementation. "
                "Use the Automodel key names instead, e.g.:\n"
                "  lora:\n"
                "    dim: 128           # was `r`\n"
                "    alpha: 256         # was `lora_alpha`\n"
                "    dropout: 0.01      # was `lora_dropout`\n"
                "    target_modules: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"]"
            )

    def _unfreeze_lora_params(self) -> int:
        """Re-enable gradients on the LoRA adapters after the LLM-body freeze.

        On the ``device_mesh`` path Automodel installs LoRA *inside*
        ``from_pretrained``, i.e. before :meth:`_apply_freeze_config` runs, so
        ``freeze_module(self.llm.model)`` also freezes ``lora_A`` / ``lora_B``.
        ``ensure_lora_trainable`` only appends the ``prevent_freeze_params``
        regex, and ``optim_setup.freeze_and_subset`` skips parameters that
        already have ``requires_grad=False`` *before* consulting that regex — so
        without this the adapters would silently never train (the only hint being
        an "UNMATCHED freeze-preventing pattern" warning).

        Returns the number of adapter parameters re-enabled.
        """
        import re

        pattern = re.compile(LORA_PARAM_PATTERN)
        n = 0
        for name, param in self.llm.named_parameters():
            if pattern.match(name) and not param.requires_grad:
                param.requires_grad_(True)
                n += 1
        if n:
            logging.info(f"Re-enabled gradients on {n} LoRA parameters frozen by the LLM-body freeze.")
        return n

    # ------------------------------------------------------------------
    # LLM-backend indirection hooks (see StreamingSTTModel for the contract)
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Infer the device from the LLM's parameters.

        ``LightningModule.device`` is set by the Trainer and defaults to CPU
        during standalone inference (no Trainer). Query the actual parameter
        storage instead so ``.to(self.device)`` works for both regular and
        ``DTensor`` (FSDP2 / TP) parameters.
        """
        if self.llm is not None:
            p = next(self.llm.parameters(), None)
            if p is not None:
                return p._local_tensor.device if isinstance(p, DTensor) else p.device
        return super().device

    @property
    def embed_tokens(self):
        """The LLM's embedding layer (kept inside the LLM, unlike the parent class)."""
        if self.llm is None:
            return None
        return self.llm.model.embed_tokens

    def _embed_tokens(self, input_ids: Tensor) -> Tensor:
        """Embed token IDs using the LLM's embedding table.

        Uses ``F.embedding`` instead of calling the ``nn.Embedding`` module to
        avoid triggering FSDP2 pre-forward hooks (which lazily initialize the
        child before the root LLM module, causing a ``RuntimeError``).

        When the weight is a sharded ``DTensor`` (FSDP2) it is ``full_tensor()``-ed
        first to all-gather the complete embedding table — the same operation
        FSDP2 performs inside the LLM's forward pass.

        ``grad_placements`` matters whenever the embedding table is trainable
        (``freeze_embed_tokens: false``, which the streaming recipe needs so the
        new ``<blank>`` token can be learned). The lookup happens *outside* the
        LLM's FSDP2 forward, so FSDP2 never reduce-scatters this parameter's
        gradient. Without declaring the gradient ``Partial``, ``full_tensor()``
        defaults to a ``Replicate`` backward and the redistribute back to
        ``Shard(0)`` is a pure local chunk — each rank would keep only its own
        micro-batch contribution. ``Partial(reduce_op="avg")`` makes that
        redistribute an averaging reduce-scatter, matching what FSDP2 does for
        every other parameter.
        """
        weight = self.embed_tokens.weight
        if isinstance(weight, DTensor):
            if weight.requires_grad:
                grad_placements = [Partial(reduce_op="avg")] * weight.device_mesh.ndim
                weight = weight.full_tensor(grad_placements=grad_placements)
            else:
                weight = weight.full_tensor()
        return F.embedding(input_ids, weight)

    @property
    def _embed_ref_tensor(self) -> Tensor:
        """Dtype/device reference for ``Tensor.type_as``.

        ``type_as`` on a ``DTensor`` would try to convert to the DTensor *type*;
        the local shard carries the same dtype and device, so use it instead.
        """
        weight = self.embed_tokens.weight
        return weight.to_local() if isinstance(weight, DTensor) else weight

    def _llm_forward(self, **kwargs):
        """Run the LLM and normalize the output into a HuggingFace ``ModelOutput``.

        Automodel's own model implementations (e.g. the MoE backbones) do not
        always honor ``return_dict=True`` — some return the logits tensor
        directly. Wrap those so the inherited training and streaming-inference
        code can keep using ``out.logits`` / ``out.past_key_values``.
        """
        out = self.llm(**kwargs)

        if isinstance(out, ModelOutput):
            normalized = out
        elif isinstance(out, torch.Tensor):
            normalized = CausalLMOutputWithPast(logits=out)
        elif isinstance(out, dict):
            normalized = CausalLMOutputWithPast(**{k: v for k, v in out.items() if k in _CAUSAL_LM_OUTPUT_FIELDS})
        elif isinstance(out, (tuple, list)) and out:
            normalized = CausalLMOutputWithPast(logits=out[0])
        else:
            raise RuntimeError(f"Unsupported LLM output type: {type(out)!r}; expected a ModelOutput, dict or Tensor.")

        if kwargs.get("use_cache", False) and normalized.past_key_values is None:
            raise RuntimeError(
                f"The Automodel LLM {type(self.llm).__name__!r} did not return a KV cache "
                "(`past_key_values`) with `use_cache=True`. Streaming inference requires "
                "incremental decoding support from the backbone; use the HuggingFace-backed "
                "StreamingSTTModel for inference with this LLM."
            )
        if kwargs.get("output_hidden_states", False) and normalized.hidden_states is None:
            raise RuntimeError(
                f"The Automodel LLM {type(self.llm).__name__!r} did not return hidden states with "
                "`output_hidden_states=True`, which the aux chunk-boundary classifier "
                "(`use_chunk_classifier=true`) needs."
            )
        return normalized

    def _move_embedding_ctx(self):
        """No-op: ``embed_tokens`` already lives inside the LLM here."""
        return nullcontext()

    # ------------------------------------------------------------------
    # Vocabulary sizing
    # ------------------------------------------------------------------

    def _resize_llm_embeddings(self) -> None:
        """Defer the embedding resize — the LLM does not exist during ``__init__``."""
        if self.llm is None:
            return
        self._sync_llm_vocab_size()

    def _sync_llm_vocab_size(self) -> None:
        """Ensure the LLM embedding table covers every tokenizer ID.

        Unlike the parent class this never *shrinks* the table: most checkpoints
        ship with spare rows (``config.vocab_size > len(tokenizer)``, e.g. 151936
        vs 151669 for Qwen3), which the newly added special tokens can occupy
        without touching the parameter shapes. That matters because a sharded
        (``DTensor``) embedding cannot be resized in place at all — growing one
        raises with an actionable message instead of failing obscurely later.
        """
        target = len(self.tokenizer.tokenizer)
        embed = self.embed_tokens
        current = int(embed.weight.shape[0])
        if current >= target:
            if current > target:
                logging.info(
                    f"LLM embedding table has {current} rows for a tokenizer of {target} tokens "
                    f"({current - target} spare rows) — added special tokens reuse the spare rows, no resize needed."
                )
            return

        if isinstance(embed.weight, DTensor):
            raise RuntimeError(
                f"The tokenizer has {target} tokens but the sharded LLM embedding table only has {current} rows, "
                "and a DTensor-sharded embedding cannot be resized. Extend the checkpoint's vocabulary offline "
                "(so that config.vocab_size >= len(tokenizer)), or reduce the number of added special tokens "
                "(`blank_token` / `write_token` / `end_of_audio_token`)."
            )
        self.llm.resize_token_embeddings(target)
        logging.info(f"Resized the LLM embedding table from {current} to {target} rows.")

    # ------------------------------------------------------------------
    # Aux chunk-boundary classifier
    # ------------------------------------------------------------------

    def _build_chunk_classifier(self) -> None:
        """Build the aux backbone + head, skipping the warm-start when the LLM is sharded.

        The parent's warm start copies the last K LLM layers into the aux
        backbone. Under FSDP2 those weights are ``DTensor`` shards, which cannot
        be loaded into a plain (unsharded) module, so fall back to random init.
        """
        llm_param = next(self.llm.parameters(), None)
        if isinstance(llm_param, DTensor) and self.core_cfg.chunk_classifier_init_from_llm:
            logging.warning(
                "chunk_classifier_init_from_llm=True is not supported when the LLM is sharded "
                "(DTensor parameters); falling back to random init for the aux chunk classifier."
            )
            init_from_llm = self.core_cfg.chunk_classifier_init_from_llm
            self.core_cfg.chunk_classifier_init_from_llm = False
            try:
                super()._build_chunk_classifier()
            finally:
                self.core_cfg.chunk_classifier_init_from_llm = init_from_llm
            return
        super()._build_chunk_classifier()

    # ------------------------------------------------------------------
    # Model construction (called by the Lightning strategy with the device mesh)
    # ------------------------------------------------------------------

    def configure_model(
        self,
        device_mesh=None,
        distributed_config=None,
        moe_config=None,
        moe_mesh=None,
        activation_checkpointing_llm: Optional[bool] = None,
        activation_checkpointing_perception: Optional[bool] = None,
    ) -> None:
        """Build (and parallelize) the LLM, the perception module and the aux head.

        Called by Lightning after ``setup_environment()``, so ``device_mesh`` is
        available and Automodel can shard the LLM while loading its weights.
        Safe to call twice — the second call is a no-op.

        TODO(rebase onto newer NeMo): this signature mirrors the SALMAutomodel of
        the vendored NeMo. Upstream (NVIDIA/NeMo main) has since collapsed the
        four parallelism arguments into one object::

            def configure_model(self, distributed_setup=None, activation_checkpointing_perception=None):
                device_mesh     = distributed_setup.mesh_context.device_mesh
                self._moe_mesh  = distributed_setup.mesh_context.moe_mesh
                automodel_kwargs["distributed_setup"] = distributed_setup

        Do NOT port that early: today's ``AutomodelParallelStrategy`` exposes
        ``distributed_config`` / ``moe_config`` / ``moe_mesh`` /
        ``activation_checkpointing_llm`` and has no ``distributed_setup``, and the
        pinned ``nemo_automodel``'s ``from_pretrained`` takes ``device_mesh`` /
        ``moe_mesh`` / ``tp_plan`` but not ``distributed_setup`` — so the new-style
        hook would silently take the no-mesh branch and train unsharded, without
        gradient synchronization. Migrate all of it together (strategy +
        nemo_automodel + SALMAutomodel + this class), and at that point also:

        * move :meth:`training_step` to upstream's ``training_step(dataloader_iter)``
          / ``_training_step_batch`` split — it is where ``_forward_default_dtype``
          is captured for :meth:`backward`;
        * fold upstream's ``NotImplementedError`` for ``pp_size > 1`` into
          :meth:`_validate_parallelism_compatibility`, which already rejects TP/CP.
        """
        resolved_mesh = device_mesh if device_mesh is not None else self.device_mesh
        if self.llm is not None:
            # Already built. Refuse the case where the first build had no mesh
            # (``init_configure_model=true``, or ``from_pretrained`` — see
            # parts/hf_hub.py) but one exists now: returning silently would leave
            # every rank with a full unsharded replica and, since
            # ModelParallelStrategy installs no DDP wrapper, no gradient
            # synchronization at all.
            if resolved_mesh is not None and not self._configured_with_mesh:
                raise RuntimeError(
                    "configure_model() already ran without a device mesh (model.init_configure_model=true "
                    "or a from_pretrained() restore), but a device mesh is available now. The existing model "
                    "would train unsharded and WITHOUT gradient synchronization. Remove `init_configure_model` "
                    "from the model config for distributed training — the strategy calls configure_model() "
                    "with the device mesh itself."
                )
            return

        # Use the provided device_mesh, or fall back to the LightningModule property.
        device_mesh = resolved_mesh
        if device_mesh is not None:
            self._device_mesh = device_mesh

        dtype = self._resolve_dtype()

        # Fall back to trainer.strategy for configs (Lightning training path).
        if distributed_config is None and self._trainer is not None:
            distributed_config = getattr(self._trainer.strategy, "distributed_config", None)
        if moe_mesh is None and self._trainer is not None:
            moe_mesh = getattr(self._trainer.strategy, "moe_mesh", None)
        if moe_config is None and self._trainer is not None:
            moe_config = getattr(self._trainer.strategy, "moe_config", None)
        if activation_checkpointing_llm is None and self._trainer is not None:
            activation_checkpointing_llm = getattr(self._trainer.strategy, "activation_checkpointing_llm", None)
        if activation_checkpointing_llm is None:
            activation_checkpointing_llm = False
        if activation_checkpointing_perception is None and self._trainer is not None:
            activation_checkpointing_perception = getattr(
                self._trainer.strategy, "activation_checkpointing_perception", None
            )
        if activation_checkpointing_perception is None:
            activation_checkpointing_perception = False

        automodel_kwargs = self._build_automodel_kwargs(
            device_mesh=device_mesh,
            distributed_config=distributed_config,
            moe_config=moe_config,
            moe_mesh=moe_mesh,
            activation_checkpointing_llm=activation_checkpointing_llm,
        )

        # When LoRA is configured and we have a device_mesh, pass peft_config
        # through Automodel so LoRA is applied before FSDP2 sharding (which
        # handles meta-device init correctly).
        peft_config = make_peft_config(self.cfg.lora) if "lora" in self.cfg else None
        if peft_config is not None and device_mesh is not None:
            automodel_kwargs["peft_config"] = peft_config

        # --- LLM ---
        self.llm = load_pretrained_automodel_llm(
            self.core_cfg.pretrained_llm,
            pretrained_weights=self.core_cfg.load_llm_weights,
            dtype=dtype,
            trust_remote_code=self.cfg.get("trust_remote_code", False),
            **automodel_kwargs,
        )
        # Special tokens were registered on the tokenizer in __init__, when the
        # LLM did not exist yet — reconcile the vocabulary now.
        self._sync_llm_vocab_size()

        # --- Speech encoder (perception module) ---
        self.perception = setup_perception(
            cfg=self.cfg,
            output_dim=self.llm.config.hidden_size,
            pretrained_asr=self.core_cfg.pretrained_asr,
            pretrained_weights=self.core_cfg.load_asr_weights,
            audio_pad_to=self.core_cfg.audio_pad_to,
            att_context_size=self.core_cfg.att_context_size,
        )

        # --- Aux chunk-boundary classifier (only built when enabled) ---
        if self.core_cfg.use_chunk_classifier:
            assert self.core_cfg.chunk_size == 0, (
                "use_chunk_classifier=True requires dynamic chunking "
                f"(chunk_size=0), got chunk_size={self.core_cfg.chunk_size}"
            )
            self._build_chunk_classifier()
            # Aux training/eval reads self._user_footer_first_id (the BCE positive
            # label). It's normally set lazily by _ensure_inference_cache, but
            # training runs before any inference call — so prime the cache now.
            self._ensure_inference_cache()

        # Activation checkpointing on the perception encoder layers. Must run
        # BEFORE FSDP2 wrapping (same as the LLM path inside Automodel) so that
        # checkpoint_wrapper sees the pristine layer objects.
        self.perception.set_activation_checkpointing(activation_checkpointing_perception)

        # Module-level freezing FIRST — everything that marks parameters trainable
        # (LoRA adapters, MoE gates) must run *after* it, because
        # ``freeze_module(self.llm.model)`` recurses over the whole LLM body and
        # ``optim_setup.freeze_and_subset`` skips ``requires_grad=False`` params
        # before it ever consults ``prevent_freeze_params``.
        self._apply_freeze_config()

        # --- LoRA ---
        # With a device_mesh, LoRA was already applied inside Automodel's
        # from_pretrained (before sharding); otherwise apply it now.
        if peft_config is not None:
            if device_mesh is None:
                maybe_install_lora(self)
            else:
                # Still need the prevent_freeze_params pattern for configure_optimizers.
                ensure_lora_trainable(self)
            # In the device_mesh path the adapters existed before the freeze above
            # and were frozen along with the LLM body; re-enable them.
            self._unfreeze_lora_params()
            # Automodel's LoRA does not freeze the base model, but keep the
            # explicit lm_head switch honored (parity with the parent class).
            if (lm_head := self._lm_head_module) is not None:
                if self.core_cfg.freeze_llm_head:
                    freeze_module(lm_head)
                else:
                    unfreeze_module(lm_head)

        # MoE options last: ``train_gate`` unfreezes router weights that live
        # inside ``llm.model`` and would be re-frozen by ``_apply_freeze_config``.
        self.setup_moe_options()

        if device_mesh is None:
            maybe_load_pretrained_models(self)
            logging.info("\n" + str(ModelSummary(self, max_depth=2)))
            return

        # Cast the perception module to the training dtype BEFORE FSDP2 wrapping.
        # The LLM is already in the target dtype (loaded with torch_dtype=dtype)
        # and FSDP2 requires a uniform parameter dtype.
        if dtype != torch.float32:
            self.perception.to(dtype=dtype)

        self._configured_with_mesh = True

        # Use the same FSDP mesh Automodel uses for the LLM so that gradient
        # clipping can torch.stack norms from all parameters.
        dim_names = device_mesh.mesh_dim_names
        if "dp_replicate" in dim_names and "dp_shard_cp" in dim_names:
            fsdp_mesh = device_mesh["dp_replicate", "dp_shard_cp"]
        elif "dp_shard_cp" in dim_names:
            fsdp_mesh = device_mesh["dp_shard_cp"]
        else:
            fsdp_mesh = device_mesh["dp"]

        if fsdp_mesh.size() > 1:
            self._use_fsdp = True
            self.perception = fully_shard(self.perception, mesh=fsdp_mesh)
            # Every parameter must belong to some FSDP2 group, otherwise its
            # gradients are never reduced across ranks and the replicas diverge.
            if self.core_cfg.use_chunk_classifier:
                self.chunk_classifier_backbone = fully_shard(self.chunk_classifier_backbone, mesh=fsdp_mesh)
                self.chunk_classifier_head = fully_shard(self.chunk_classifier_head, mesh=fsdp_mesh)

        # Optionally initialize weights from a previous training checkpoint
        # (fresh optimizer/scheduler). Must happen after FSDP wrapping so that
        # DCP loading can fill DTensor parameters with the correct shards.
        maybe_load_pretrained_models(self)
        logging.info("\n" + str(ModelSummary(self, max_depth=2)))

    def _resolve_dtype(self) -> torch.dtype:
        """Training dtype: trainer precision first, then ``model.dtype`` from the config."""
        if self._trainer is not None:
            precision = str(self._trainer.precision)
            if "bf16" in precision:
                return torch.bfloat16
            if "16" in precision:
                return torch.float16
            return torch.float32
        cfg_dtype = self.cfg.get("torch_dtype", None) or getattr(self.core_cfg, "dtype", None)
        if cfg_dtype is None:
            return torch.float32
        return getattr(torch, cfg_dtype) if isinstance(cfg_dtype, str) else cfg_dtype

    def _build_automodel_kwargs(
        self,
        device_mesh,
        distributed_config,
        moe_config,
        moe_mesh,
        activation_checkpointing_llm: bool,
    ) -> dict:
        """Assemble the kwargs forwarded to Automodel's ``from_pretrained`` / ``from_config``."""
        automodel_kwargs: dict[str, Any] = {}
        if device_mesh is not None:
            automodel_kwargs["device_mesh"] = device_mesh
            # Automodel's instantiate_infrastructure unconditionally calls
            # .to_dict() on these configs, so always provide defaults.
            if distributed_config is None:
                from nemo_automodel.components.distributed.config import FSDP2Config

                distributed_config = FSDP2Config()
            if moe_config is None:
                from nemo_automodel.components.moe.config import MoEParallelizerConfig

                moe_config = MoEParallelizerConfig()
            # Route the single LLM AC flag to both paths: the EP/MoE parallelizer
            # reads ``activation_checkpointing`` directly (MoEParallelizerConfig
            # has no such field), while FSDP2's AC wrapping reads the field on
            # FSDP2Config.
            if activation_checkpointing_llm:
                distributed_config.activation_checkpointing = True
            automodel_kwargs["distributed_config"] = distributed_config
            automodel_kwargs["moe_config"] = moe_config
            automodel_kwargs["activation_checkpointing"] = activation_checkpointing_llm
        if moe_mesh is not None:
            automodel_kwargs["moe_mesh"] = moe_mesh

        # torch.compile support.
        compile_cfg = self.cfg.get("compile", None)
        if compile_cfg is not None:
            from nemo_automodel.components.utils.compile_utils import CompileConfig

            automodel_kwargs["compile_config"] = CompileConfig(**dict(compile_cfg))

        # Backend dispatch — lets YAML pick attn/linear/rms_norm/MoE backends
        # (e.g. attn=sdpa to bypass TransformerEngine).
        backend_cfg = self.cfg.get("automodel_backend", None)
        if backend_cfg is not None:
            from nemo_automodel.components.models.common import BackendConfig

            automodel_kwargs["backend"] = BackendConfig(**OmegaConf.to_container(backend_cfg, resolve=True))

        # Pin the SDPA kernel used by attn=sdpa (e.g. ["flash_attention"]).
        sdpa_method = self.cfg.get("sdpa_method", None)
        if sdpa_method is not None:
            automodel_kwargs["sdpa_method"] = list(OmegaConf.to_container(sdpa_method, resolve=True))

        return automodel_kwargs

    # ------------------------------------------------------------------
    # Training loop integration
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Validate the parallelism configuration and configure the MoE aux-loss scaler."""
        self._validate_parallelism_compatibility()
        self._configure_moe_aux_loss_scaler()

    def _validate_parallelism_compatibility(self) -> None:
        """Reject parallelism dimensions the streaming recipe cannot support.

        * **Context parallelism** splits the sequence across ranks, which is
          incompatible with the interleaved audio/text layout and the causal
          chunk-by-chunk supervision this model is trained with.
        * **Tensor parallelism** makes the LLM's ``lm_head`` emit vocab-sharded
          ``DTensor`` logits, which the inherited loss cannot consume: the
          blank / non-blank breakdown indexes the per-token loss with a plain
          boolean mask (``per_token_loss[is_blank]``), and mixing a
          ``torch.Tensor`` mask with a ``DTensor`` raises at the first step.
          Supporting TP also requires padding the interleaved sequence to a
          multiple of ``tp_size`` (see ``SALMAutomodel.prepare_inputs``).

        Only FSDP2 (``dp_size`` / ``dp_replicate_size``) and expert parallelism
        (``ep_size``, MoE backbones) are supported.
        """
        device_mesh = getattr(self, "_device_mesh", None)
        if device_mesh is None:
            return
        names = device_mesh.mesh_dim_names or ()
        if "cp" in names and device_mesh["cp"].size() > 1:
            raise ValueError(
                f"{type(self).__name__} does not support context parallelism "
                f"(got cp_size={device_mesh['cp'].size()}). The interleaved audio/text sequence must stay "
                "on a single rank; use FSDP2 (dp_size) instead."
            )
        if "tp" in names and device_mesh["tp"].size() > 1:
            raise ValueError(
                f"{type(self).__name__} does not support tensor parallelism "
                f"(got tp_size={device_mesh['tp'].size()}). TP shards the lm_head output, and the streaming "
                "loss (blank / non-blank breakdown) indexes the per-token loss with a plain boolean mask, "
                "which is not DTensor-safe. Use FSDP2 (dp_size) instead."
            )

    def training_step(self, batch, batch_idx: int):
        # Recorded for _setup_moe_fsdp_sync(), which runs during backward().
        self._current_batch_idx = batch_idx
        # Lightning runs training_step inside precision_plugin.train_step_context()
        # -> forward_context(), so this is the default dtype the forward saw. See
        # backward() for why it has to be replayed there.
        self._forward_default_dtype = torch.get_default_dtype()
        ans = super().training_step(batch, batch_idx)
        self.maybe_log_moe_metrics(batch_idx)
        return ans

    def backward(self, *args, **kwargs):
        """Backward under the forward's default dtype, so AC recompute matches.

        ``precision: bf16-true`` resolves to Lightning's ``HalfPrecision``, whose
        ``forward_context()`` sets the *global default dtype* to bf16 — but only
        around the forward. Backward, and therefore any activation-checkpointing
        recompute, runs outside it at the fp32 default. Anything the LLM creates
        without an explicit dtype (notably the empty per-expert buffers a MoE layer
        builds when an expert receives zero tokens) is then bf16 when saved and
        fp32 when recomputed, and torch raises::

            CheckpointError: Recomputed values for the following tensors have
            different metadata than during the forward pass.
            saved: {'shape': [0, 1], 'dtype': torch.bfloat16}
            recomputed: {'shape': [0, 1], 'dtype': torch.float32}

        Replaying the captured default dtype makes recompute see exactly what the
        forward saw. Mirroring the recorded value (rather than forcing
        ``_resolve_dtype()``) keeps this correct for precision plugins that do NOT
        touch the default dtype, e.g. ``bf16-flash``/``FlashPrecision`` — there the
        captured value is already fp32 and this is a no-op.
        """
        self._setup_moe_fsdp_sync()
        # Escape hatch for bisecting this replay itself: model.replay_forward_dtype_in_backward=false
        # restores stock behaviour (backward at the fp32 default). Only do that with
        # activation_checkpointing_llm=false, or the CheckpointError comes back.
        dtype = getattr(self, "_forward_default_dtype", None)
        if not self.cfg.get("replay_forward_dtype_in_backward", True):
            dtype = None
        ctx = (
            _DtypeContextManager(dtype) if dtype is not None and dtype != torch.get_default_dtype() else nullcontext()
        )
        with ctx:
            super().backward(*args, **kwargs)

    def _setup_moe_fsdp_sync(self):
        """Configure MoE FSDP gradient sync for gradient accumulation.

        When ``accumulate_grad_batches > 1``, disables gradient all-reduce and
        resharding on intermediate backward passes and re-enables them on the
        final backward before ``optimizer.step()``. No-op when the LLM lacks
        Automodel's ``MoEFSDPSyncMixin`` or gradient accumulation is inactive.
        """
        if not self._use_fsdp or not hasattr(self.llm, 'prepare_for_grad_accumulation'):
            return
        acc = self.trainer.accumulate_grad_batches if self._trainer else 1
        if acc <= 1:
            return
        batch_idx = getattr(self, '_current_batch_idx', 0)
        is_final = (batch_idx + 1) % acc == 0 or (batch_idx + 1) == self.trainer.num_training_batches
        if is_final:
            self.llm.prepare_for_final_backward()
        else:
            self.llm.prepare_for_grad_accumulation()

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm=None):
        """Mesh-aware gradient clipping, logging the pre-clip gradient norm.

        When Automodel parallelizes the LLM, some parameters end up as DTensors
        on the ``(dp_replicate, dp_shard_cp)`` mesh while others may be on the
        flattened ``dp`` mesh. PyTorch's ``clip_grad_norm_`` requires all norms
        to share the same mesh for ``torch.stack``, so delegate to Automodel's
        implementation which groups parameters by ``(mesh_id, placements)``.

        The norm is logged as ``grad_norm`` because it is the earliest signal
        that training is diverging: a single ``inf``/``nan`` gradient makes the
        global norm non-finite, and clipping then scales *every* parameter by
        that value — so one bad backward silently NaNs the whole model (encoder
        included) while the forward that produced it looked perfectly healthy.
        """
        norm = None
        if not self._use_fsdp or gradient_clip_val is None or gradient_clip_val <= 0:
            params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            if params:
                norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(p.grad.detach().float(), 2) for p in params]), 2
                )
            self._log_grad_norm(norm)
            return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        from nemo_automodel.components.training.utils import _clip_grad_norm_impl

        params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if params:
            norm = _clip_grad_norm_impl(params, max_norm=gradient_clip_val)
        self._log_grad_norm(norm)

    def _log_grad_norm(self, norm) -> None:
        """Log the pre-clip gradient norm, and localize it when it is non-finite."""
        if norm is None:
            return
        norm = norm.full_tensor() if isinstance(norm, DTensor) else norm
        norm = norm.detach().float()
        if not torch.isfinite(norm):
            logging.warning(
                "Non-finite gradient norm (%s) at step %s — clipping scales EVERY parameter by it, so the "
                "whole model (perception included) is NaN from the next forward on. First offenders:\n%s",
                norm.item(),
                getattr(self, "_current_batch_idx", "?"),
                self._first_nonfinite_grads(),
            )
        self.log("grad_norm", norm, on_step=True, prog_bar=True)

    def _first_nonfinite_grads(self, limit: int = 8) -> str:
        """Names of the first parameters whose gradient is non-finite.

        Which module they live in is the whole diagnosis: gradients inside
        ``llm.*`` only means the LLM backward produced them, whereas
        ``perception.*`` being hit too (before any clipping has run) points at the
        loss or the interleaved embeddings instead. Only called on the non-finite
        path, so the full parameter scan costs nothing in a healthy step. Uses the
        local shard of a ``DTensor`` to avoid a collective inside a warning.
        """
        hits, scanned = [], 0
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            scanned += 1
            g = p.grad
            g = g.to_local() if isinstance(g, DTensor) else g
            if g.numel() and not torch.isfinite(g).all():
                finite = torch.isfinite(g)
                hits.append(f"    {name}: {int((~finite).sum())}/{g.numel()} non-finite")
                if len(hits) >= limit:
                    hits.append(f"    ... (stopping after {limit})")
                    break
        if not hits:
            return f"    (no non-finite parameter grads among {scanned} scanned — the norm itself overflowed)"
        return "\n".join(hits)

    # ------------------------------------------------------------------
    # MoE helpers (no-ops for dense LLMs)
    # ------------------------------------------------------------------

    def setup_moe_options(self):
        """Apply MoE config overrides and enable load-balance tracking.

        Must be called after ``self.llm`` is created. Safe no-op when the LLM
        has no ``Gate`` modules (dense backbone) or when Automodel's MoE
        components are unavailable.
        """
        aux_loss_coeff = self.cfg.get("aux_loss_coeff", 0.0)
        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        moe_requested = (
            aux_loss_coeff > 0
            or self.cfg.get("train_gate", False)
            or (moe_metrics_cfg is not None and moe_metrics_cfg.get("enabled", False))
        )
        try:
            from nemo_automodel.components.moe.layers import Gate
        except ImportError:
            if moe_requested:
                logging.warning(
                    "MoE options (aux_loss_coeff / train_gate / moe_metrics) were requested but "
                    "nemo_automodel's MoE components are not importable — ignoring them."
                )
            return

        if aux_loss_coeff > 0:
            for module in self.llm.modules():
                if isinstance(module, Gate):
                    module.aux_loss_coeff = aux_loss_coeff

        if self.cfg.get("train_gate", False):
            for module in self.llm.modules():
                if isinstance(module, Gate):
                    module.train_gate = True
                    module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)

        if moe_metrics_cfg is not None and moe_metrics_cfg.get("enabled", False):
            from nemo_automodel.components.moe.load_balance_metrics import enable_load_balance_tracking

            enable_load_balance_tracking(self.llm)

    def maybe_log_moe_metrics(self, step: int):
        """Collect and log MoE load-balance metrics.

        All ranks must call this method (the all-reduce inside
        ``collect_expert_loads`` is collective).
        """
        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if moe_metrics_cfg is None or not moe_metrics_cfg.get("enabled", False):
            return

        from nemo_automodel.components.moe.load_balance_metrics import (
            collect_expert_loads,
            compute_brief_metrics,
            compute_detailed_metrics,
        )

        layer_loads = collect_expert_loads(self.llm, dp_group=self._get_moe_dp_group())
        if not layer_loads:
            return

        top_k = moe_metrics_cfg.get("top_k_experts", 5)
        if moe_metrics_cfg.get("mode", "brief") == "detailed":
            detailed_every = moe_metrics_cfg.get("detailed_every_steps", None)
            if detailed_every is not None and step % detailed_every != 0:
                metrics = compute_brief_metrics(layer_loads, top_k=top_k)
            else:
                metrics = compute_detailed_metrics(layer_loads, top_k=top_k)
        else:
            metrics = compute_brief_metrics(layer_loads, top_k=top_k)

        self.log_dict(metrics, on_step=True)

    def _get_moe_dp_group(self):
        """Return the DP process group for MoE metrics all-reduce.

        Prefers the ``dp_cp`` submesh (includes context parallelism) and falls
        back to ``dp``. Both are flattened submeshes registered in
        ``device_mesh._flatten_mapping``, so they are resolved via
        ``get_flat_mesh``. Returns ``None`` when no device mesh is available.
        """
        device_mesh = getattr(self, "_device_mesh", None)
        if device_mesh is None:
            return None
        try:
            from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
        except ImportError:
            return None

        try:
            if "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
                return get_flat_mesh(device_mesh, "dp_cp").get_group()
            return get_flat_mesh(device_mesh, "dp").get_group()
        except KeyError:
            return None

    def _configure_moe_aux_loss_scaler(self) -> None:
        """Cancel FSDP's gradient averaging on MoE aux-loss grads.

        ``MoEAuxLossAutoScaler`` multiplies aux-loss-derived gradients by
        ``main_loss_backward_scale`` during backward; FSDP's all-reduce then
        divides every gradient by ``dp_group_size``. Setting the scaler to
        ``dp_group_size`` cancels that division out. No-op when
        ``nemo_automodel`` is unavailable.
        """
        try:
            from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
        except ImportError:
            return
        dp_group = self._get_moe_dp_group()
        dp_size = dp_group.size() if dp_group is not None else 1
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(float(dp_size))
