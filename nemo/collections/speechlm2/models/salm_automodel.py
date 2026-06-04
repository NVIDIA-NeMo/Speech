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
import warnings
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.models.salm import _resolve_audios_in_prompt, replace_placeholders_and_build_targets
from nemo.collections.speechlm2.parts.automodel_lora import ensure_lora_trainable, make_peft_config, maybe_install_lora
from nemo.collections.speechlm2.parts.encoder_chunking import encode_audio_with_optional_chunking
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.multispeaker import build_speaker_tokens, maybe_init_lss_loss
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_automodel_llm,
    maybe_load_pretrained_models,
    setup_speech_encoder,
    update_perception_output_dim,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType


class SALMAutomodel(LightningModule, HFHubMixin):
    def __init__(self, cfg) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to SALMAutomodel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.audio_locator_tag = self.cfg.audio_locator_tag

        tokenizer_src = self.cfg.get("tokenizer_path", None) or self.cfg.pretrained_llm
        self.tokenizer = AutoTokenizer(
            tokenizer_src, use_fast=True, trust_remote_code=self.cfg.get("trust_remote_code", False)
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        self.speaker_token_ids = build_speaker_tokens(self.cfg.get("speaker_tokens", None), self.tokenizer)
        self.lss_loss = maybe_init_lss_loss(self.cfg.get("lss_loss", None), self.speaker_token_ids)
        self.llm = None  # populated by configure_model
        self.perception = None  # populated by configure_model

        self._use_fsdp = False
        self._use_tp = False
        self._mtp_enabled = False  # set in configure_model when cfg.mtp.enabled

        if self.cfg.get("init_configure_model", False):
            self.configure_model()

    @property
    def device(self) -> torch.device:
        """Infer device from the LLM's parameters.

        ``LightningModule.device`` is set by the Trainer and defaults to CPU
        during standalone inference (no Trainer).  Override to query the actual
        parameter storage so that ``.to(self.device)`` works correctly for
        both regular and DTensor (FSDP2/distributed) parameters.
        """
        if self.llm is not None:
            p = next(self.llm.parameters(), None)
            if p is not None:
                return p._local_tensor.device if isinstance(p, DTensor) else p.device
        return super().device

    @property
    def embed_tokens(self):
        """Navigate to the LLM's embedding layer (kept inside the LLM)."""
        if self.llm is None:
            return None
        return self.llm.model.embed_tokens

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs using the LLM's embedding table.

        Uses ``F.embedding`` instead of calling the ``nn.Embedding`` module to
        avoid triggering FSDP2 pre-forward hooks (which lazily initialize the
        child before the root LLM module, causing a ``RuntimeError``).

        When the weight is a sharded ``DTensor`` (FSDP2), we ``full_tensor()``
        it first to all-gather the complete embedding table — the same operation
        FSDP2 performs inside the LLM's forward pass.
        """
        weight = self.embed_tokens.weight
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        return torch.nn.functional.embedding(input_ids, weight)

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "the text tokenizer has no <pad> or <unk> tokens available, using id 0 for padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def audio_locator_tag_id(self) -> int:
        return self.tokenizer.token_to_id(self.audio_locator_tag)

    @property
    def token_equivalent_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame/token at the output of ``self.perception``.
        """
        return self.perception.token_equivalent_duration

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor = None,
        cache=None,
        **llm_kwargs,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        ``llm_kwargs`` carries optional THD/packed-sequence metadata
        (``qkv_format``, ``cu_seqlens``, ``position_ids``, ``max_seqlen``);
        it is empty for the BSHD path.
        """
        # input_embeds: (B, T, H) for BSHD or (T_total, H) for THD packed
        # (the THD shape mirrors Automodel's _shard_thd_chunk_for_te output —
        # the model squeezes 3D inputs internally when qkv_format=="thd", so
        # passing 2D directly skips that hop)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
            **llm_kwargs,
        )
        if not isinstance(out, dict):
            # NeMo Automodel doesn't respect return_dict=True yet
            ans = {"logits": out}
        else:
            ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
            if cache is not None:
                ans["cache"] = out["past_key_values"]
            # MTP per-depth hidden states (present only when the LLM built an MTP head,
            # is in training mode, and received input_ids). Attribute access is None-safe;
            # out['mtp_per_depth_h'] would KeyError when the field is None.
            mtp_h = getattr(out, "mtp_per_depth_h", None)
            if mtp_h is not None:
                ans["mtp_per_depth_h"] = mtp_h
                ans["mtp_loss_scaling_factor"] = getattr(out, "mtp_loss_scaling_factor", None)
        return ans

    def _uses_parallel_expert_encoder(self) -> bool:
        """Whether the mounted perception encoder is a ``ParallelExpertEncoder``.

        The PE encoder performs its own context-preserving long-form online inference
        (``forward`` -> ``_forward_online``), so audio must be fed to it as a
        single long sequence when oracle speaker targets are absent.
        """
        from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder

        return isinstance(getattr(self.perception, "encoder", None), ParallelExpertEncoder)

    def _warn_parallel_expert_encoder_inference_compatibility(self, cp_size: int) -> None:
        if not self.cfg.get("pe_encoder_path", None):
            return

        unsupported = []
        if self.cfg.get("encoder_chunk_size_seconds", None) is not None:
            unsupported.append("encoder_chunk_size_seconds")
        if self.cfg.get("packed_sequences", False):
            unsupported.append("packed_sequences")
        if cp_size > 1:
            unsupported.append(f"cp_size={cp_size}")

        if unsupported:
            warnings.warn(
                "The ParallelExpertEncoder inference path is currently experimental and does not support "
                f"{', '.join(unsupported)}. It will be made to work with these options later."
                " This warning only applies when `spk_targets` are absent; training with ground-truth "
                "`spk_targets` still uses the regular encoder chunking path.",
                stacklevel=2,
            )

    def prepare_inputs(self, batch: dict):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations. With a
          ``ParallelExpertEncoder`` the full audio is encoded in a single
          perception forward (long-form streaming is handled inside the encoder).
          With an ordinary encoder, long source audio is optionally time-chunked
          and recombined via ``parts.encoder_chunking``.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.

        When ``batch["spk_targets"]`` is present, those RTTM-derived speaker
        targets are injected into a ``ParallelExpertEncoder``. Otherwise, the
        encoder runs its embedded Sortformer to predict diarization.
        """
        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        from nemo.collections.speechlm2.parts.cp_helpers import encode_audio_with_cp_distribution, get_cp_mesh

        spk_targets = batch.get("spk_targets", None)
        cp_mesh, cp_size, _ = get_cp_mesh(getattr(self, "_device_mesh", None))
        # Encoder path by (PEE, spk_targets):
        # PEE=true  & spk_targets=None  : Inference mode, uses recursive encoding in PEE, NO chunking/CP.
        # PEE=true  & spk_targets!=None : Training mode, ``spk_targets`` injected into PEE with chunking/CP.
        # PEE=false & spk_targets=None  : Training/Inference mode, plain encoder with chunking/CP.
        # PEE=false & spk_targets!=None : Training/Inference mode, plain encoder with chunking/CP and
        #                                 the provided ``spk_targets`` is ignored (no-op).
        if self._uses_parallel_expert_encoder() and spk_targets is None:
            self._warn_parallel_expert_encoder_inference_compatibility(cp_size)
            audio_embs, audio_emb_lens = self.perception(
                input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
            )
            audio_embs = [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]
        else:
            audio_embs = encode_audio_with_cp_distribution(
                self.perception,
                batch["audios"],
                batch["audio_lens"],
                chunk_size_seconds=self.cfg.get("encoder_chunk_size_seconds", None),
                sampling_rate=self.sampling_rate,
                cp_mesh=cp_mesh,
                spk_targets=spk_targets,
            )
        input_ids_to_embed = torch.where(batch["input_ids"] == self.audio_locator_tag_id, 0, batch["input_ids"])
        text_embs = self._embed_tokens(input_ids_to_embed)
        target_ids_full = batch["input_ids"].where(batch["loss_mask"], -100)  # CrossEntropyLoss().ignore_index

        # Packed-sequence (THD) path — used for both training and validation when enabled.
        # Generate stays on the BSHD path (it doesn't go through prepare_inputs).
        if self.cfg.get("packed_sequences", False):
            from nemo.collections.speechlm2.parts.packed_sequences import prepare_packed_llm_inputs

            return prepare_packed_llm_inputs(
                input_ids=batch["input_ids"],
                text_embs=text_embs,
                audio_embs=audio_embs,
                target_ids=target_ids_full,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                device_mesh=getattr(self, "_device_mesh", None),
            )

        # When MTP is on, also build the expanded token-id sequence (text=id, audio=pad)
        # aligned 1:1 with input_embs, so the LLM's MTP head can embed the future tokens.
        if self._mtp_enabled:
            input_embs, target_ids, attention_mask, mtp_input_ids = replace_placeholders_and_build_targets(
                input_ids=batch["input_ids"],
                embeds=text_embs,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                replacements=audio_embs,
                target_ids=target_ids_full,
                return_input_ids=True,
            )
        else:
            input_embs, target_ids, attention_mask = replace_placeholders_and_build_targets(
                input_ids=batch["input_ids"],
                embeds=text_embs,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                replacements=audio_embs,
                target_ids=target_ids_full,
            )
            mtp_input_ids = None
        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]
        # mtp_input_ids stays input-aligned (like input_embs): slice [:, :-1], NOT [:, 1:].
        if mtp_input_ids is not None:
            mtp_input_ids = mtp_input_ids[:, :-1]

        # BSHD path runs only when CP is inactive (the fit-start validator
        # rejects BSHD + CP > 1, see _validate_parallelism_compatibility).
        # Truncate the seq dim to be divisible by tp_size so sequence
        # parallelism doesn't reshape the input under us.
        if self._use_tp:
            tp_size = self.device_mesh["tp"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_size) != 0:
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]
                if mtp_input_ids is not None:
                    mtp_input_ids = mtp_input_ids[:, :-remainder]

        return {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "llm_kwargs": {"input_ids": mtp_input_ids} if mtp_input_ids is not None else {},
        }

    def on_fit_start(self) -> None:
        """Configure the MoE aux-loss backward scaler to cancel FSDP's gradient
        averaging (see ``_configure_moe_aux_loss_scaler``)."""
        self._validate_parallelism_compatibility()
        self._configure_moe_aux_loss_scaler()

    def _validate_parallelism_compatibility(self) -> None:
        """Raise on known-incompatible THD/CP/backend configurations.

        Delegates to :func:`nemo.collections.speechlm2.parts.parallel.validate_parallelism_compatibility`
        with the runtime-derived values from this model's config and device mesh.
        """
        import os

        from nemo.collections.speechlm2.parts.parallel import validate_parallelism_compatibility

        cp_size = 1
        device_mesh = getattr(self, "_device_mesh", None)
        if device_mesh is not None:
            names = device_mesh.mesh_dim_names or ()
            if "cp" in names:
                cp_size = device_mesh["cp"].size()

        attn_backend = self.cfg.get("automodel_backend", {}).get("attn", "te")
        nvte_fused_attn = os.environ.get("NVTE_FUSED_ATTN")
        device_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

        validate_parallelism_compatibility(
            packed_sequences=bool(self.cfg.get("packed_sequences", False)),
            cp_size=cp_size,
            attn_backend=attn_backend,
            nvte_fused_attn=nvte_fused_attn,
            device_capability=device_capability,
        )

    def training_step(self, batch: dict, batch_idx: int):
        self._current_batch_idx = batch_idx
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(
            inputs["input_embeds"],
            attention_mask=inputs["attention_mask"],
            **inputs.get("llm_kwargs", {}),
        )
        num_frames = (inputs["target_ids"] != -100).long().sum()

        # Match Automodel's training recipe: normalize CE by the *global* token count across
        # the DP group rather than each rank's local count. With variable-length speech batches
        # a local normalizer makes every rank contribute a differently-scaled gradient, and
        # FSDP's gradient averaging doesn't recover the true global mean. All-reduce the
        # labeled-token count and scale the per-rank loss by ``dp_size`` so that FSDP's
        # gradient averaging yields ``sum(rank_CE_sum) / num_frames_global``.
        dp_group = self._get_moe_dp_group()
        dp_size = dp_group.size() if dp_group is not None else 1
        if dp_group is not None and dist.is_available() and dist.is_initialized():
            num_frames_global = num_frames.clone()
            dist.all_reduce(num_frames_global, op=dist.ReduceOp.SUM, group=dp_group)
        else:
            num_frames_global = num_frames
        num_frames_global = num_frames_global.clamp(min=1)

        with loss_parallel():
            logits = forward_outputs["logits"]
            loss_sum = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # BSHD (B,T,V) or THD (1,T,V) -> (*, V)
                inputs["target_ids"].reshape(-1),  # BSHD (B,T) or THD (T,) -> (*,)
                reduction="sum",
                ignore_index=-100,
            )
            loss = loss_sum * dp_size / num_frames_global

        # Latent speaker supervision loss (auxiliary, optional).
        if self.lss_loss is not None and num_frames > 0:
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            loss = loss + self.lss_loss(log_probs=log_probs, labels=inputs["target_ids"])

        # Display the local per-token CE so logged values stay on the same scale as before
        # this fix. The gradient-carrying ``loss`` above is the globally-normalized quantity.
        with torch.no_grad():
            loss_display = loss_sum.detach() / num_frames.clamp(min=1)

        # Multi-Token Prediction auxiliary loss. Reuse Automodel's calculate_mtp_loss
        # (it rolls labels per depth, masks the wrapped tail, projects each depth's hidden
        # state through the shared lm_head via self._mtp_loss_fn, and sums the CE). We pass
        # num_label_tokens=num_frames_global so each depth is normalized by the same global
        # token count as the main loss, and multiply by dp_size to cancel FSDP's gradient
        # averaging (matching the main-loss scaling above). loss_parallel() keeps the
        # vocab-sharded CE correct under tensor parallelism.
        mtp_h = forward_outputs.get("mtp_per_depth_h", None)
        if mtp_h is not None:
            from nemo_automodel.components.loss.mtp import calculate_mtp_loss

            scaling = forward_outputs.get("mtp_loss_scaling_factor", None)
            if scaling is None:
                scaling = getattr(self, "_mtp_loss_scaling_factor", 0.1)
            with loss_parallel():
                mtp_loss = dp_size * calculate_mtp_loss(
                    self._mtp_loss_fn,
                    mtp_per_depth_h=mtp_h,
                    labels=inputs["target_ids"],
                    model=self.llm,
                    scaling_factor=scaling,
                    num_label_tokens=num_frames_global,
                )
            loss = loss + mtp_loss
            self.log("mtp_loss", mtp_loss.detach(), on_step=True, prog_bar=True)

        # Input embeds shape is (B, T, H) for BSHD or (T, H) for THD packed.
        input_embeds = inputs["input_embeds"]
        if input_embeds.dim() == 2:
            B, T = 1, input_embeds.shape[0]
        else:
            B, T = input_embeds.shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "num_frames_global": num_frames_global.to(torch.float32),
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": (batch["input_ids"] != self.text_pad_id).long().sum() / batch["input_ids"].numel(),
        }
        self.log("loss", loss_display, on_step=True, prog_bar=True)
        self.log_dict({k: v for k, v in ans.items() if k != "loss"}, on_step=True)
        self.maybe_log_moe_metrics(batch_idx)
        return ans

    def on_validation_epoch_start(self) -> None:
        self._partial_val_loss_sums = defaultdict(list)
        self._partial_val_corrects = defaultdict(list)
        self._partial_val_num_frames = defaultdict(list)
        self._partial_val_lss = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        accuracies = []
        reduction_group = self._get_moe_dp_group()
        for name, vals in self._partial_val_loss_sums.items():
            loss_sum = torch.stack(vals).sum()
            correct = torch.stack(self._partial_val_corrects[name]).sum().to(loss_sum.dtype)
            num_frames = torch.stack(self._partial_val_num_frames[name]).sum().to(loss_sum.dtype)
            metric_sums = self._reduce_validation_metric_sums(
                torch.stack([loss_sum, correct, num_frames]), reduction_group
            )
            num_frames = metric_sums[2].clamp(min=1)
            val_loss = metric_sums[0] / num_frames
            val_acc = metric_sums[1] / num_frames

            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)

            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)

        self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)
        self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        if getattr(self, "lss_loss", None) is not None:
            lss_vals = []
            for name, vals in self._partial_val_lss.items():
                val_lss = torch.stack(vals).mean()
                self.log(f"val_lss_{name}", val_lss, on_epoch=True, sync_dist=True)
                lss_vals.append(val_lss)
            if lss_vals:
                self.log("val_lss", torch.stack(lss_vals).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_loss_sums.clear()
        self._partial_val_corrects.clear()
        self._partial_val_num_frames.clear()
        self._partial_val_lss.clear()

    def _reduce_validation_metric_sums(self, metric_sums: Tensor, group) -> Tensor:
        if group is not None and dist.is_available() and dist.is_initialized():
            metric_sums = metric_sums.clone()
            dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM, group=group)
        return metric_sums

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(
                inputs["input_embeds"],
                attention_mask=inputs["attention_mask"],
                **inputs.get("llm_kwargs", {}),
            )
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                logits = forward_outputs["logits"]
                loss_sum = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    inputs["target_ids"].reshape(-1),
                    reduction="sum",
                    ignore_index=-100,
                )

            if self.lss_loss is not None and num_frames > 0:
                if isinstance(logits, DTensor):
                    logits = logits.full_tensor()
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                lss_val = self.lss_loss(log_probs=log_probs, labels=inputs["target_ids"])
                self._partial_val_lss[name].append(lss_val.detach())

            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = inputs["target_ids"].reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            correct = preds.eq(refs).sum()

            self._partial_val_loss_sums[name].append(loss_sum.detach())
            self._partial_val_corrects[name].append(correct.detach().to(loss_sum.dtype))
            self._partial_val_num_frames[name].append(num_frames.detach().to(loss_sum.dtype))

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        self._setup_moe_fsdp_sync()
        with loss_parallel():
            super().backward(*args, **kwargs)

    def _setup_moe_fsdp_sync(self):
        """Configure MoE FSDP gradient sync for gradient accumulation.

        When ``accumulate_grad_batches > 1``, disables gradient all-reduce and
        resharding on intermediate backward passes and re-enables them on the
        final backward before ``optimizer.step()``.  This avoids redundant
        communication during gradient accumulation.

        Delegates to the LLM's ``MoEFSDPSyncMixin`` methods.  No-op when the
        LLM lacks the mixin or gradient accumulation is not active.
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
        """Override Lightning's gradient clipping to handle mixed FSDP device meshes.

        When automodel parallelizes the LLM, some parameters end up as DTensors
        on the ``(dp_replicate, dp_shard_cp)`` mesh while others may be on the
        flattened ``dp`` mesh.  PyTorch's ``clip_grad_norm_`` requires all norms
        to share the same mesh for ``torch.stack``.  We delegate to automodel's
        mesh-aware ``_clip_grad_norm_impl`` which groups parameters by
        ``(mesh_id, placements)`` and combines per-group norms as plain tensors.
        """
        if not self._use_fsdp or gradient_clip_val is None or gradient_clip_val <= 0:
            return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        from nemo_automodel.components.training.utils import _clip_grad_norm_impl

        params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if params:
            _clip_grad_norm_impl(params, max_norm=gradient_clip_val)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        spk_targets: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        enable_thinking: bool | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate LLM answers given text or mixed text+audio prompts.

        Example 1. High-level API using ``prompts`` to provide both text and audio::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [
            ...             {
            ...                 "role": "user",
            ...                 "content": f"Transcribe the following: {model.audio_locator_tag}",
            ...                 "audio": ["path/to/audio.wav"],
            ...             }
            ...         ]
            ...    ],
            ...    max_new_tokens=128,
            ... )

        You may also include a ``transformers.GenerationConfig`` object to customize decoding strategy::

            >>> answer_ids = model.generate(..., generation_config=GenerationConfig(do_sample=True, num_beams=5))

        Example 2. Lower-level API, using ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}],
            ...        [{"role": "user", "content": f"Transcribe the following in Polish: {model.audio_locator_tag}"}],
            ...    ],
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Example 3. Lower-level API, using pre-tokenized and pre-formatted ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=prompts,  # torch.Tensor, int64, of shape (batch, num_tokens)
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Inputs:
            prompts: batch of prompts Tensor or as list[dict] each in the following format
                [
                  # batch example id 0
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}]
                  # batch example id 1
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following in Polish: {model.audio_locator_tag}"}]
                ]
                "role" is LLM-specific, you can pass multiple turns as well.
                If ``prompts`` is a Tensor, we assume it was already formatted in the relevant chat template
                and tokenized with the model's tokenizer.
            audios: Optional. Time-domain audio signal zero-padded batch of shape (B, T).
                The number of audios must correspond to the number of occurrences of <audio_locator_tag> in prompts.
                Each prompt can have multiple audios.
            audio_lens: Optional. Length of each audio example.
            spk_targets: Optional ``(B, T, n_spk)`` speaker-activity tensor (e.g. oracle / RTTM-derived
                diarization) injected into the perception encoder. Only effective when the mounted
                encoder is a ``ParallelExpertEncoder`` (i.e. ``model.pe_encoder_path`` was set); it
                overrides the encoder's embedded Sortformer prediction for this call. When ``None``
                (default), the encoder runs its embedded Sortformer as usual.
            generation_config: Optional HuggingFace GenerationConfig object.
            enable_thinking: Optional prompt-formatter hint forwarded to ``encode_dialog``.
                Relevant for prompt formats that support thinking/reasoning mode.
            generation_kwargs: Keyword arguments passed directly to the underlying LLM's ``generate`` method.
        """
        # Encode prompt dicts into int token ids.
        if isinstance(prompts, torch.Tensor):
            tokens = prompts.to(self.device)
        else:
            if (
                maybe_audio := _resolve_audios_in_prompt(prompts, sampling_rate=self.sampling_rate, device=self.device)
            ) is not None:
                assert (
                    audios is None and audio_lens is None
                ), "Audios cannot be provided via ``prompts`` and ``audios``/``audio_lens`` arguments simultaneously."
                audios, audio_lens = maybe_audio
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            formatter_kwargs = {}
            if enable_thinking is not None:
                formatter_kwargs["enable_thinking"] = enable_thinking
            tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt, **formatter_kwargs)["input_ids"] for prompt in prompts],
                padding_value=self.text_pad_id,
            ).to(self.device)
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                pad_token_id=self.text_pad_id,
            )
        if audios is not None:
            # Audio + text input for generation.
            # Prepare token embeddings and audio embeddings.
            tokens_to_embed = tokens.where(tokens != self.audio_locator_tag_id, 0)
            token_embeds = self._embed_tokens(tokens_to_embed)
            if self._uses_parallel_expert_encoder() and spk_targets is None:
                # This is only used for inference when ``spk_targets`` is None.
                # PEE needs to produce ``spk_targets`` itself through recursive encoding.
                self._warn_parallel_expert_encoder_inference_compatibility(cp_size=1)
                audio_embeds, audio_embed_lens = self.perception(input_signal=audios, input_signal_length=audio_lens)
                audio_embeds = [emb[:emblen] for emb, emblen in zip(audio_embeds, audio_embed_lens)]
            else:
                audio_embeds = encode_audio_with_optional_chunking(
                    self.perception,
                    audios,
                    audio_lens,
                    chunk_size_seconds=self.cfg.get("encoder_chunk_size_seconds", None),
                    sampling_rate=self.sampling_rate,
                    spk_targets=spk_targets,
                )
            # Insert audio embeddings into relevant positions in text embeddings.
            input_embeds, _, attention_mask = replace_placeholders_and_build_targets(
                input_ids=tokens,
                embeds=token_embeds,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                replacements=audio_embeds,
                target_ids=None,
            )
            answer_tokens = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
                generation_config=generation_config,
            )
        else:
            # Text-only generation — embed_tokens stays in LLM, HF generate uses it natively.
            attention_mask = tokens != self.text_pad_id
            answer_tokens = self.llm.generate(
                input_ids=tokens,
                attention_mask=attention_mask,
                **generation_kwargs,
                generation_config=generation_config,
            )
        return answer_tokens

    def setup_moe_options(self):
        """Apply MoE config overrides and enable load balance tracking.

        Must be called after ``self.llm`` is created.  Iterates over all Gate
        modules in the LLM and overrides their settings.  Also enables
        load balance tracking when ``moe_metrics.enabled`` is set.

        Safe no-op when the LLM has no Gate modules (non-MoE backbone).
        """
        from nemo_automodel.components.moe.layers import Gate

        aux_loss_coeff = self.cfg.get("aux_loss_coeff", 0.0)
        if aux_loss_coeff > 0:
            for module in self.llm.modules():
                if isinstance(module, Gate):
                    module.aux_loss_coeff = aux_loss_coeff

        train_gate = self.cfg.get("train_gate", False)
        if train_gate:
            for module in self.llm.modules():
                if isinstance(module, Gate):
                    module.train_gate = True
                    module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)

        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if moe_metrics_cfg is not None and moe_metrics_cfg.get("enabled", False):
            from nemo_automodel.components.moe.load_balance_metrics import enable_load_balance_tracking

            enable_load_balance_tracking(self.llm)

    def maybe_log_moe_metrics(self, step: int):
        """Collect and log MoE load balance metrics.

        All ranks must call this method (the all-reduce inside
        ``collect_expert_loads`` is collective).  Metrics are logged via
        Lightning's ``self.log_dict`` which respects ``log_every_n_steps``.

        Args:
            step: Current ``batch_idx``, used to decide brief vs detailed mode.
        """
        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if moe_metrics_cfg is None or not moe_metrics_cfg.get("enabled", False):
            return

        from nemo_automodel.components.moe.load_balance_metrics import (
            collect_expert_loads,
            compute_brief_metrics,
            compute_detailed_metrics,
        )

        dp_group = self._get_moe_dp_group()
        layer_loads = collect_expert_loads(self.llm, dp_group=dp_group)
        if not layer_loads:
            return

        mode = moe_metrics_cfg.get("mode", "brief")
        top_k = moe_metrics_cfg.get("top_k_experts", 5)

        if mode == "detailed":
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

        Mirrors Automodel's ``_get_dp_group(include_cp=True)`` pattern: prefers
        the ``dp_cp`` submesh (includes context parallelism) for the broadest
        reduction, falling back to ``dp``. ``dp`` and ``dp_cp`` are flattened
        submeshes registered in ``device_mesh._flatten_mapping`` — they are not
        in ``mesh_dim_names`` of the root mesh, so resolve them via
        ``get_flat_mesh`` (same helper Automodel's ``base_recipe`` uses).

        Returns ``None`` when no device mesh is available (e.g. DDP training),
        causing ``collect_expert_loads`` to skip all-reduce (rank-local view).
        """
        device_mesh = getattr(self, "_device_mesh", None)
        if device_mesh is None:
            return None
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        try:
            if "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
                return get_flat_mesh(device_mesh, "dp_cp").get_group()
            return get_flat_mesh(device_mesh, "dp").get_group()
        except KeyError:
            return None

    def _configure_moe_aux_loss_scaler(self) -> None:
        """Cancel FSDP's gradient averaging on MoE aux-loss grads.

        ``MoEAuxLossAutoScaler`` multiplies aux-loss-derived gradients by
        ``main_loss_backward_scale`` during backward. FSDP's all-reduce then
        divides every gradient by ``dp_group_size``. Setting the scaler to
        ``dp_group_size`` (non-PP case) cancels that division out, matching the
        intent in ``nemo_automodel/recipes/llm/train_ft.py`` — otherwise the
        aux-loss contribution to the gradient would be under-scaled by a factor
        of ``dp_group_size``.

        No-op when ``nemo_automodel`` isn't available (non-MoE builds).
        """
        try:
            from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
        except ImportError:
            return
        dp_group = self._get_moe_dp_group()
        dp_size = dp_group.size() if dp_group is not None else 1
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(float(dp_size))

    def configure_optimizers(self):
        return configure_optimizers(self)

    def _build_and_attach_mtp_head(self, mtp_cfg, dtype) -> None:
        """Construct the NemotronV3 MTP head and attach it as ``self.llm.mtp``.

        The base 30B checkpoint has no MTP head, and neither the config-override nor the
        ``config=`` from_pretrained path can inject the pattern for this model (its
        ``__init__`` swallows ``**kwargs``). So we set the pattern on the live HF config and
        build the head with the same Automodel factory the model would use internally, then
        attach it. The head's weights are fresh (not in the checkpoint).

        NOTE: this head is NOT FSDP/EP-sharded. That's fine for single-node runs and the
        VERIFY-3 smoke test, but real multi-rank training should also shard/replicate it so
        its gradients sync across the data-parallel group (TODO before scaling out).
        """
        from nemo_automodel.components.models.nemotron_v3.mtp import (
            build_mtp_config_from_hf,
            build_nemotron_v3_mtp,
        )

        llm = self.llm
        cfg_obj = llm.config
        cfg_obj.mtp_hybrid_override_pattern = str(mtp_cfg.get("hybrid_override_pattern", "*"))
        mtp_config = build_mtp_config_from_hf(
            cfg_obj,
            loss_scaling_factor=self._mtp_loss_scaling_factor,
            num_nextn_predict_layers=int(mtp_cfg.get("num_nextn_predict_layers", 1)),
            use_repeated_layer=bool(mtp_cfg.get("use_repeated_layer", False)),
        )
        llm.mtp_config = mtp_config
        mtp = build_nemotron_v3_mtp(
            cfg_obj,
            mtp_config=mtp_config,
            backend=llm.backend,
            moe_config=llm.model.moe_config,
            dtype=dtype,
        )
        device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        llm.mtp = mtp.to(device=device, dtype=dtype)
        warnings.warn(
            f"MTP head attached (unsharded): num_layers={mtp_config.num_layers} "
            f"pattern={cfg_obj.mtp_hybrid_override_pattern!r}"
        )

    def configure_model(
        self,
        device_mesh=None,
        distributed_config=None,
        moe_config=None,
        moe_mesh=None,
        activation_checkpointing_llm: bool | None = None,
        activation_checkpointing_perception: bool | None = None,
    ) -> None:
        # Use provided device_mesh, or fall back to LightningModule property
        if device_mesh is not None:
            self._device_mesh = device_mesh
        else:
            device_mesh = self.device_mesh

        # Derive dtype from trainer precision (e.g. "bf16-flash" -> bfloat16).
        dtype = torch.float32
        if self._trainer is not None:
            precision = str(self._trainer.precision)
            if "bf16" in precision:
                dtype = torch.bfloat16
            elif "16" in precision:
                dtype = torch.float16
        elif hasattr(self.cfg, 'torch_dtype') and self.cfg.torch_dtype is not None:
            td = self.cfg.torch_dtype
            dtype = getattr(torch, td) if isinstance(td, str) else td

        # Fall back to trainer.strategy for configs (Lightning training path)
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

        automodel_kwargs = {}
        if device_mesh is not None:
            # Automodel infrastructure expects typed config objects. Latest
            # Automodel routes all distributed settings through DistributedSetup,
            # while older Automodel accepted these pieces as separate kwargs.
            if distributed_config is None:
                from nemo_automodel.components.distributed.config import FSDP2Config

                distributed_config = FSDP2Config()
            if moe_config is None:
                try:
                    from nemo_automodel.components.distributed.config import MoEParallelizerConfig
                except ImportError:
                    from nemo_automodel.components.moe.config import MoEParallelizerConfig

                moe_config = MoEParallelizerConfig()

            try:
                from nemo_automodel.components.distributed.config import DistributedSetup
                from nemo_automodel.components.distributed.mesh import MeshContext
            except ImportError:
                DistributedSetup = None
                MeshContext = None

            if DistributedSetup is not None and MeshContext is not None:
                if activation_checkpointing_llm:
                    distributed_config.activation_checkpointing = activation_checkpointing_llm
                automodel_kwargs["distributed_setup"] = DistributedSetup(
                    mesh_context=MeshContext.from_meshes(device_mesh, moe_mesh),
                    strategy_config=distributed_config,
                    moe_parallel_config=moe_config,
                    activation_checkpointing=activation_checkpointing_llm,
                )
            else:
                automodel_kwargs["device_mesh"] = device_mesh
                # Older Automodel used separate kwargs and only understood a bool
                # strategy flag, while the EP/MoE path consumed the original value.
                if activation_checkpointing_llm:
                    distributed_config.activation_checkpointing = True
                automodel_kwargs["distributed_config"] = distributed_config
                automodel_kwargs["moe_config"] = moe_config
                automodel_kwargs["activation_checkpointing"] = activation_checkpointing_llm
        if moe_mesh is not None and "distributed_setup" not in automodel_kwargs:
            automodel_kwargs["moe_mesh"] = moe_mesh

        # When LoRA is configured and we have a device_mesh, pass peft_config
        # through automodel so LoRA is applied before FSDP2 sharding (handles
        # meta-device init correctly).
        peft_config = make_peft_config(self.cfg.lora) if "lora" in self.cfg else None
        if peft_config is not None and device_mesh is not None:
            automodel_kwargs["peft_config"] = peft_config

        # Pass compile_config through to automodel for torch.compile support.
        compile_cfg = self.cfg.get("compile", None)
        if compile_cfg is not None:
            from nemo_automodel.components.utils.compile_utils import CompileConfig

            compile_dict = dict(compile_cfg)
            automodel_kwargs["compile_config"] = CompileConfig(**compile_dict)

        pretrained_weights = self.cfg.get("pretrained_weights", True)
        pretrained_llm_weights = self.cfg.get("pretrained_llm_weights", pretrained_weights)
        pretrained_asr_weights = self.cfg.get("pretrained_asr_weights", pretrained_weights)
        # Pass backend through to automodel — lets YAML pick attn/linear/rms_norm/MoE
        # dispatcher backends (e.g. set attn=sdpa to bypass TransformerEngine).
        backend_cfg = self.cfg.get("automodel_backend", None)
        if backend_cfg is not None:
            from nemo_automodel.components.models.common import BackendConfig

            automodel_kwargs["backend"] = BackendConfig(**OmegaConf.to_container(backend_cfg, resolve=True))

        # Pin the SDPA kernel used by attn=sdpa (e.g. [flash_attention] to force FA2
        # and error out if unavailable). Accepts strings; resolved by automodel.
        sdpa_method = self.cfg.get("sdpa_method", None)
        if sdpa_method is not None:
            automodel_kwargs["sdpa_method"] = list(OmegaConf.to_container(sdpa_method, resolve=True))

        # Multi-Token Prediction (MTP): we ADD an auxiliary head to a base model that ships
        # without one. We cannot inject the head's settings through from_pretrained:
        #   * mtp_hybrid_override_pattern is read ONLY from the HF config, but the base
        #     config class doesn't declare it, so HF drops it as an unknown kwarg;
        #   * passing config= collides — NemotronHForCausalLM.__init__ has **kwargs, so
        #     Automodel's _filter_kwargs_for_init can't strip the positional ``config``.
        # So we load the LLM normally and then construct + attach the head ourselves.
        mtp_cfg = self.cfg.get("mtp", None)
        self._mtp_enabled = bool(mtp_cfg is not None and mtp_cfg.get("enabled", False))
        if self._mtp_enabled:
            if self.cfg.get("packed_sequences", False):
                raise NotImplementedError("MTP is not yet supported with packed_sequences=true (THD path).")
            self._mtp_loss_scaling_factor = float(mtp_cfg.get("loss_scaling_factor", 0.1))
            # Same per-token loss class the Automodel recipe uses for MTP; reduction="sum"
            # so calculate_mtp_loss can normalize by our global token count.
            from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

            self._mtp_loss_fn = MaskedCrossEntropy(reduction="sum")

        self.llm = load_pretrained_automodel_llm(
            self.cfg.pretrained_llm,
            pretrained_weights=pretrained_llm_weights,
            dtype=dtype,
            trust_remote_code=self.cfg.get("trust_remote_code", False),
            **automodel_kwargs,
        )

        # Build + attach the MTP head ourselves (the base checkpoint has none).
        if self._mtp_enabled and getattr(self.llm, "mtp", None) is None:
            self._build_and_attach_mtp_head(mtp_cfg, dtype)
        if self._mtp_enabled and getattr(self.llm, "mtp", None) is None:
            raise RuntimeError("MTP enabled but self.llm.mtp is still None after _build_and_attach_mtp_head.")

        # Apply MoE options (aux_loss_coeff override, load balance tracking)
        self.setup_moe_options()

        # Create perception module (must happen after LLM so output_dim matches)
        setup_speech_encoder(self, pretrained_weights=pretrained_asr_weights)

        # Fix projection dim for pretrained_weights=False (config output_dim may not match LLM)
        update_perception_output_dim(self)

        # Activation checkpointing on perception encoder layers. Must run BEFORE
        # FSDP2 wrapping (see LLM path in automodel) so checkpoint_wrapper sees
        # the pristine layer objects and fully_shard indexes the final structure.
        self.perception.set_activation_checkpointing(activation_checkpointing_perception)

        # Apply LoRA adapters to the LLM.
        # When device_mesh is set, LoRA was already applied inside automodel's
        # from_pretrained (before sharding).  Otherwise, apply it now.
        if peft_config is not None and device_mesh is None:
            maybe_install_lora(self)
        elif peft_config is not None:
            # LoRA was applied by automodel; still need to ensure the
            # prevent_freeze_params pattern is set for configure_optimizers.
            ensure_lora_trainable(self)

        if device_mesh is None:
            maybe_load_pretrained_models(self)
            return

        # Cast perception to training dtype BEFORE FSDP2 wrapping.
        # The LLM is already in the target dtype (loaded via torch_dtype=dtype).
        # FSDP2 requires uniform parameter dtype, so we cast all parameters.
        if dtype != torch.float32:
            self.perception.to(dtype=dtype)

        if device_mesh["tp"].size() > 1:
            self._use_tp = True

        # Use the same FSDP mesh as automodel uses for the LLM so that
        # gradient clipping can torch.stack norms from all parameters.
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

        # Enable MoE FSDP gradient accumulation optimization.
        # The MoEFSDPSyncMixin on the LLM defers gradient sync/resharding on
        # intermediate backward passes — _setup_moe_fsdp_sync() drives it.
        # TODO(pzelasko): causes issue in torch's FSDP backward, investigate later:
        # AttributeError: 'FSDPParam' object has no attribute '_unsharded_param'. Did you mean: 'unsharded_param'?
        # if self._use_fsdp and hasattr(self.llm, 'prepare_for_grad_accumulation'):
        #     self.llm.backend.enable_fsdp_optimizations = True

        # Optionally initialize weights from a previous training checkpoint
        # (fresh optimizer/scheduler). Must happen after FSDP wrapping so that
        # DCP loading can fill DTensor parameters with correct shards.
        maybe_load_pretrained_models(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        embed_tokens = self.embed_tokens
        vocab_size = embed_tokens.num_embeddings if embed_tokens is not None else self.tokenizer.vocab_size
        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": vocab_size,
                    "excluded_token_ids": [self.audio_locator_tag_id],
                    "excluded_token_replacement_id": self.text_pad_id,
                    "forced_token_ids": {0: self.audio_locator_tag_id},
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
            ],
        }
