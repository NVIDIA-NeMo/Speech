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

        self.tokenizer = AutoTokenizer(
            self.cfg.pretrained_llm, use_fast=True, trust_remote_code=self.cfg.get("trust_remote_code", False)
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        self.llm = None  # populated by configure_model
        self.perception = None  # populated by configure_model

        self._use_fsdp = False
        self._use_tp = False

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
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        """
        # input_embeds and out: (B, T, H)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        if not isinstance(out, dict):
            # NeMo Automodel doesn't respect return_dict=True yet
            ans = {"logits": out}
        else:
            ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
            if cache is not None:
                ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Optionally chunk long source audio for the encoder and recombine the encoded chunks.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.
        """
        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        audio_embs = encode_audio_with_optional_chunking(
            self.perception,
            batch["audios"],
            batch["audio_lens"],
            chunk_size_seconds=self.cfg.get("encoder_chunk_size_seconds", None),
            sampling_rate=self.sampling_rate,
        )
        input_ids_to_embed = torch.where(batch["input_ids"] == self.audio_locator_tag_id, 0, batch["input_ids"])
        text_embs = self._embed_tokens(input_ids_to_embed)
        input_embs, target_ids, attention_mask = replace_placeholders_and_build_targets(
            input_ids=batch["input_ids"],
            embeds=text_embs,
            padding_id=self.text_pad_id,
            placeholder_id=self.audio_locator_tag_id,
            replacements=audio_embs,
            target_ids=batch["input_ids"].where(batch["loss_mask"], -100),  # CrossEntropyLoss().ignore_index
        )
        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]

        # Combine target audio and text into a single tensor to slice them together.
        # It will also help us truncate the sequence lengths to be divisible by TP world size,
        # when TP is enabled.
        # Input ids: (B, T, K+1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tp"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence length shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]

        return {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

    def on_fit_start(self) -> None:
        """Configure the MoE aux-loss backward scaler to cancel FSDP's gradient
        averaging (see ``_configure_moe_aux_loss_scaler``)."""
        self._configure_moe_aux_loss_scaler()

    def training_step(self, batch: dict, batch_idx: int):
        self._current_batch_idx = batch_idx
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
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
            loss_sum = torch.nn.functional.cross_entropy(
                forward_outputs["logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                inputs["target_ids"].flatten(0, 1),
                reduction="sum",
                ignore_index=-100,
            )
            loss = loss_sum * dp_size / num_frames_global

        # Display the local per-token CE so logged values stay on the same scale as before
        # this fix. The gradient-carrying ``loss`` above is the globally-normalized quantity.
        with torch.no_grad():
            loss_display = loss_sum.detach() / num_frames.clamp(min=1)

        B, T = inputs["input_embeds"].shape[:2]
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
        self._partial_val_losses = defaultdict(list)
        self._partial_accuracies = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_losses.clear()
        self._partial_accuracies.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["logits"].flatten(0, 1),
                        inputs["target_ids"].flatten(0, 1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_frames
                )

            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = inputs["target_ids"].reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            accuracy = preds.eq(refs).float().mean()

            self._partial_accuracies[name].append(accuracy)
            self._partial_val_losses[name].append(loss)

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
            audio_embeds = encode_audio_with_optional_chunking(
                self.perception,
                audios,
                audio_lens,
                chunk_size_seconds=self.cfg.get("encoder_chunk_size_seconds", None),
                sampling_rate=self.sampling_rate,
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
            automodel_kwargs["device_mesh"] = device_mesh
            # automodel's instantiate_infrastructure unconditionally calls
            # .to_dict() on these configs, so we must always provide defaults.
            if distributed_config is None:
                from nemo_automodel.components.distributed.config import FSDP2Config

                distributed_config = FSDP2Config()
            if moe_config is None:
                from nemo_automodel.components.moe.config import MoEParallelizerConfig

                moe_config = MoEParallelizerConfig()
            # Route the single LLM AC flag to both paths: the EP/MoE parallelizer
            # reads ``activation_checkpointing`` directly (MoEParallelizerConfig
            # has no such field), while FSDP2's AC wrapping reads the field on
            # FSDP2Config. Forcing both keeps behavior identical regardless of
            # whether ep_size is 1 (FSDP2 path) or > 1 (EP path).
            if activation_checkpointing_llm:
                distributed_config.activation_checkpointing = True
            automodel_kwargs["distributed_config"] = distributed_config
            automodel_kwargs["moe_config"] = moe_config
            automodel_kwargs["activation_checkpointing"] = activation_checkpointing_llm
        if moe_mesh is not None:
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

        self.llm = load_pretrained_automodel_llm(
            self.cfg.pretrained_llm,
            pretrained_weights=self.cfg.pretrained_weights,
            dtype=dtype,
            trust_remote_code=self.cfg.get("trust_remote_code", False),
            **automodel_kwargs,
        )

        # Apply MoE options (aux_loss_coeff override, load balance tracking)
        self.setup_moe_options()

        # Create perception module (must happen after LLM so output_dim matches)
        setup_speech_encoder(self, pretrained_weights=self.cfg.pretrained_weights)

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
        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.text_vocab_size,
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
            ],
        }
