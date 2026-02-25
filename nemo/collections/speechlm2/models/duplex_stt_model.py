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
import copy
import os
import re

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.label_prep import maybe_prepend_prompt_tokens, prepare_text_and_asr_labels
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.empty_text import EmptyTextMetric
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.turn_taking import TurnTakingMetrics
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    maybe_load_pretrained_models,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.collections.speechlm2.streaming.duplex_stt_inference import DuplexSTTStreamingInference
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


def maybe_rename_llm_kwargs_for_nemotron(kwargs: dict, model_cfg) -> dict:
    """This is required because Nemotron models have a different signature than other HF models."""
    if 'Nemotron' not in model_cfg.pretrained_llm:
        return kwargs
    cache = kwargs.pop("past_key_values")
    if cache is not None:
        cache_key = model_cfg.get("cache_key", "past_key_values")
        kwargs[cache_key] = cache
    return kwargs


class DuplexSTTModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()

        self.cfg = DictConfig(cfg)
        self.source_sample_rate = self.cfg.source_sample_rate
        self.validation_save_path = os.path.join(self.cfg.validation_save_path, "validation_logs")

        self.predict_user_text = self.cfg.get("predict_user_text", False)

        # Load LLM first
        llm = load_pretrained_hf(
            self.cfg.pretrained_llm,
            pretrained_weights=self.cfg.pretrained_weights,
            trust_remote_code=self.cfg.get("trust_remote_code", True),
        ).train()

        # Initialize tokenizer with optional special tokens from config
        self.tokenizer = AutoTokenizer(
            self.cfg.pretrained_llm,
            use_fast=True,
            bos_token=self.cfg.get("bos_token", None),
            eos_token=self.cfg.get("eos_token", None),
            pad_token=self.cfg.get("pad_token", None),
        )

        # Extract LLM components with configurable attribute names
        llm_attr_name = self.cfg.get("llm_attr_name", "model")
        self.llm = getattr(llm, llm_attr_name)
        self.lm_head = llm.lm_head

        # Extract embedding layer with configurable attribute name
        embed_tokens_attr_name = self.cfg.get("embed_tokens_attr_name", "embed_tokens")
        self.embed_tokens = getattr(self.llm, embed_tokens_attr_name)
        delattr(self.llm, embed_tokens_attr_name)

        if self.predict_user_text:
            self.asr_head = copy.deepcopy(self.lm_head)
            self.embed_asr_tokens = copy.deepcopy(self.embed_tokens)

        maybe_install_lora(self)

        # Load the pretrained streaming ASR model
        setup_speech_encoder(self, pretrained_weights=self.cfg.pretrained_weights)

        maybe_load_pretrained_models(self)

        self._use_fsdp = False
        self._use_tp = False

        # Initialize streaming inference engine
        self.streaming_inference = DuplexSTTStreamingInference(self)

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not generating text.

        DuplexSTTModel Input/Output Format:
        - Input: User audio (speech)
        - Output: Text tokens only

        Text pad ID is used for:
        1. Frames during user speech (where the model is listening)
        2. Frames after the model completes its text response

        Example:

            flow:              |---user audio---||---assistant text---||-user audio-|
            text channel:       0000000000000000  1xxxxxxx00000000002   0000000000000
            (model output)

        Where:
        - 0 indicates PAD ID (model not generating text)
        - 1 indicates BOS ID (beginning of assistant response)
        - 2 indicates EOS ID (end of assistant response)
        - x indicates text tokens corresponding to the assistant's response

        """
        return get_pad_id(self.tokenizer)

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
    ) -> dict[str, Tensor]:
        """
        Text prediction only (audio_loss_weight=0).
        """
        kwargs = dict(inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True)
        kwargs = maybe_rename_llm_kwargs_for_nemotron(kwargs, self.cfg)
        out = self.llm(**kwargs)

        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])

        asr_logits = None
        if self.predict_user_text:
            asr_in = out['last_hidden_state']
            asr_logits = self.asr_head(asr_in)  # (B, T, asr_vocab_size)

        if not self.training:
            if self.cfg.get("inference_pad_boost", None):
                text_logits[:, :, self.text_pad_id] += self.cfg.inference_pad_boost
            if self.cfg.get("inference_bos_boost", None):
                text_logits[:, :, self.text_bos_id] += self.cfg.inference_bos_boost
            if self.cfg.get("inference_eos_boost", None):
                text_logits[:, :, self.text_eos_id] += self.cfg.inference_eos_boost

        ans = {"text_logits": text_logits}
        if self.predict_user_text:
            ans["asr_logits"] = asr_logits

        if cache is not None:
            if 'Nemotron' in self.cfg.pretrained_llm:
                cache_key = self.cfg.get("cache_key", "cache_params")
                ans["cache"] = getattr(out, cache_key, out.get(cache_key))
            else:
                ans["cache"] = out["past_key_values"]

        return ans

    def _maybe_zero_out_scale_for_asr(
        self, loss_scale: torch.Tensor, text_labels: torch.Tensor, batch: dict
    ) -> torch.Tensor:
        """
        Zero out the loss scale after text_bos_id token for ASR datasets to not penalize the agent being silent in ASR training.
        """
        if batch['task'][0] == 'asr':
            for i in range(text_labels.shape[0]):
                bos_indices = (text_labels[i] == self.text_bos_id).nonzero(as_tuple=True)
                if bos_indices[0].numel() > 0:
                    bos_idx = bos_indices[0][0].item()
                    loss_scale[i, bos_idx + 1 :, :] = 0
        return loss_scale

    def prepare_inputs(self, batch: dict):

        # Speech encoder forward pass (audio is already augmented in the dataloader)
        source_encoded, source_encoded_lens, _ = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )

        source_encoded, source_encoded_lens, target_tokens = maybe_prepend_prompt_tokens(
            batch=batch,
            embed_fn=self.embed_tokens,
            source_encoded=source_encoded,
            source_encoded_lens=source_encoded_lens,
            text_pad_id=self.text_pad_id,
        )

        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        inputs = prepare_text_and_asr_labels(
            batch=batch,
            target_tokens=target_tokens,
            source_encoded=source_encoded,
            cfg=self.cfg,
            text_pad_id=self.text_pad_id,
            text_bos_id=self.text_bos_id,
            text_eos_id=self.text_eos_id,
            use_tp=self._use_tp,
            device_mesh=self.device_mesh if self._use_tp else None,
        )

        source_encoded = inputs["source_encoded"]
        text_inputs = inputs["text_inputs"]
        text_labels = inputs["text_labels"]
        target_token_lens = inputs["target_token_lens"]  # Use adjusted lengths from label_prep
        asr_inputs = None
        if self.predict_user_text:
            asr_inputs = inputs["asr_inputs"]
            asr_labels = inputs["asr_labels"]

        input_embeds = self.embed_tokens(text_inputs) * self.cfg.get("duplex_text_channel_weight", 1.0)
        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))
        if self.predict_user_text:
            asr_inputs_embeds = self.embed_asr_tokens(asr_inputs) * self.cfg.get("duplex_asr_text_weight", 1.0)
            input_embeds.add_(asr_inputs_embeds)

        seq_mask = torch.ones_like(text_labels.unsqueeze(-1), device=self.device, dtype=torch.bool)

        if self.cfg.get("mask_sequence_loss", True):
            for i in range(target_token_lens.size(0)):
                speech_end_idx = target_token_lens[i]
                seq_mask[i, speech_end_idx:, :] = 0

        loss_scale = seq_mask.clone().float()
        asr_loss_scale = seq_mask.clone().float()
        if self.cfg.get("token_loss_weight"):
            token_weights = self.cfg.token_loss_weight
            pad_weight = token_weights.get("pad", 1.0)
            bos_weight = token_weights.get("bos", 1.0)
            eos_weight = token_weights.get("eos", 1.0)
            text_weight = token_weights.get("text", 1.0)

            loss_scale = (
                torch.where(
                    text_labels.unsqueeze(-1) == self.text_pad_id,
                    pad_weight,
                    torch.where(
                        text_labels.unsqueeze(-1) == self.text_bos_id,
                        bos_weight,
                        torch.where(text_labels.unsqueeze(-1) == self.text_eos_id, eos_weight, text_weight),
                    ),
                )
                * seq_mask.float()
            )
            # Don't penalize the agent replies for ASR training
            loss_scale = self._maybe_zero_out_scale_for_asr(loss_scale, text_labels, batch)
            if self.predict_user_text:
                asr_loss_scale = (
                    torch.where(
                        asr_labels.unsqueeze(-1) == self.text_pad_id,
                        pad_weight,
                        torch.where(
                            asr_labels.unsqueeze(-1) == self.text_bos_id,
                            bos_weight,
                            torch.where(asr_labels.unsqueeze(-1) == self.text_eos_id, eos_weight, text_weight),
                        ),
                    )
                    * seq_mask.float()
                )

        ans = {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "text_labels": text_labels,
            "loss_scale": loss_scale,
            "seq_mask": seq_mask,
        }
        if self.predict_user_text:
            ans["asr_labels"] = asr_labels
            ans["asr_loss_scale"] = asr_loss_scale
        return ans

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        res = {
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0
            )
        }

        if batch["audio_data"] is not None:
            inputs = self.prepare_inputs(batch["audio_data"])

            forward_outputs = self(inputs["input_embeds"])

            num_frames = inputs["input_lens"].sum()

            with loss_parallel():
                text_logits = forward_outputs["text_logits"]
                asr_logits = None
                if self.predict_user_text:
                    asr_logits = forward_outputs["asr_logits"]

                if self.cfg.get("mask_sequence_loss", True):
                    text_logits = text_logits * inputs["seq_mask"][:, :, 0].unsqueeze(-1)

                text_loss = (
                    torch.nn.functional.cross_entropy(
                        text_logits.flatten(0, 1),
                        inputs["text_labels"].flatten(0, 1),
                        reduction="none",
                    )
                    * inputs["loss_scale"][:, :, 0].flatten(0, 1)
                ).sum(-1) / num_frames

                asr_loss = None
                if self.predict_user_text:
                    asr_loss = (
                        torch.nn.functional.cross_entropy(
                            asr_logits.flatten(0, 1),
                            inputs["asr_labels"].flatten(0, 1),
                            reduction="none",
                        )
                        * inputs["asr_loss_scale"][:, :, 0].flatten(0, 1)
                    ).sum(-1) / num_frames

                with torch.no_grad():
                    predicted_tokens = torch.argmax(text_logits, dim=-1)  # (B, T)
                    target_tokens = inputs["text_labels"]  # (B, T)
                    valid_mask = target_tokens != self.text_pad_id

                    correct_predictions = (predicted_tokens == target_tokens) & valid_mask

                    if valid_mask.sum() > 0:
                        token_accuracy = correct_predictions.sum().float() / valid_mask.sum().float()
                    else:
                        token_accuracy = torch.tensor(0.0, device=text_logits.device)

                loss = self.cfg.text_loss_weight * text_loss

                if self.predict_user_text:
                    loss = loss + self.cfg.get('asr_loss_weight', 1.0) * asr_loss

                B, T = inputs["input_embeds"].shape[:2]
                ans = {
                    "audio_loss": loss,
                    "audio_to_text_loss": text_loss,
                    "batch": B,
                    "length": T,
                    "token_accuracy": token_accuracy,
                }
                if self.predict_user_text:
                    ans["asr_loss"] = asr_loss

                res.update(ans)

        if batch["text_data"] is not None:
            text_input_ids = batch["text_data"]["text_tokens"][:, :-1]
            text_target = batch["text_data"]["text_tokens"][:, 1:]

            text_out = self.llm(
                inputs_embeds=self.embed_tokens(text_input_ids),
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            text_logits = self.lm_head(text_out['last_hidden_state'])  # (B, T, Vt)

            text_loss = torch.nn.functional.cross_entropy(
                text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                text_target.flatten(0, 1),
                ignore_index=self.text_pad_id,
            )
            res.update(
                {
                    "text_to_text_loss": text_loss,
                }
            )

        res["loss"] = (1.0 - self.cfg.get('text_to_text_loss_weight', 0.0)) * res.get(
            "audio_loss", 0.0
        ) + self.cfg.get('text_to_text_loss_weight', 0.0) * res.get("text_to_text_loss", 0.0)
        self.log_dict(res, on_step=True)

        return res

    def on_validation_epoch_start(self) -> None:
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.bleu = BLEU().reset()

        self.turn_taking_metrics = TurnTakingMetrics(
            eos_token_id=self.text_eos_id,
            bos_token_id=self.text_bos_id,
            tolerance=13,
            latency_multiplier=0.08,
        ).reset()

        if self.predict_user_text:
            self.src_bleu = BLEU().reset()
            self.src_wer = WER().reset()
            self.empty_user_text = EmptyTextMetric().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            if "qa" not in k and "mmsu" not in k:
                self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        acc_metrics = self.results_logger.compute_and_save()

        for name, result_dict in acc_metrics.items():
            if 'acc' in result_dict:
                self.log(f"{prefix}_{name}_acc", result_dict['acc'].to(self.device), on_epoch=True, sync_dist=True)

            if 'mcq_acc' in result_dict:
                self.log(
                    f"{prefix}_{name}_mcq_acc", result_dict['mcq_acc'].to(self.device), on_epoch=True, sync_dist=True
                )

        turn_taking_metrics = self.turn_taking_metrics.compute()
        for k, m in turn_taking_metrics.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        if self.predict_user_text:
            src_bleu = self.src_bleu.compute()
            for k, m in src_bleu.items():
                self.log(f"{prefix}_src_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
            src_wer = self.src_wer.compute()
            for k, m in src_wer.items():
                self.log(f"{prefix}_src_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
            empty_user_text = self.empty_user_text.compute()
            for k, m in empty_user_text.items():
                self.log(f"{prefix}_src_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue

            dataset_batch = dataset_batch["audio_data"]

            prompt_tokens = dataset_batch.get("prompt_tokens", None)
            prompt_token_lens = dataset_batch.get("prompt_token_lens", None)

            results = self.streaming_inference.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
                prompt_tokens=prompt_tokens,
                prompt_token_lens=prompt_token_lens,
            )

            # Strip timestamps for metrics
            text_clean = [re.sub(r"<[\|$].*?[\|$]>", "", s).strip() for s in results["text"]]

            # Agent text metrics
            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=text_clean)
            if "source_tokens" in dataset_batch and results["tokens_text"] is not None:
                self.turn_taking_metrics.update(
                    name=name, source_tokens=dataset_batch["source_tokens"], pred_tokens=results["tokens_text"]
                )

            # User text metrics
            if self.predict_user_text:
                self.src_bleu.update(name=name, refs=dataset_batch["source_texts"], hyps=results["src_text"])
                self.src_wer.update(name=name, refs=dataset_batch["source_texts"], hyps=results["src_text"])
                self.empty_user_text.update(name=name, hyps=results["src_text"])

            self.results_logger.update(
                name=name,
                refs=dataset_batch["target_texts"],
                hyps=results["text"],
                samples_id=dataset_batch['sample_id'],
                user_audio=dataset_batch["source_audio"],
                user_audio_sr=self.source_sample_rate,
                src_refs=dataset_batch["source_texts"],
                src_hyps=results["src_text"],
            )

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch = batch["audio_data"]

        prompt_tokens = batch.get("prompt_tokens", None)
        prompt_token_lens = batch.get("prompt_token_lens", None)

        prediction = self.streaming_inference.offline_inference(
            batch["source_audio"],
            batch["source_audio_lens"],
            input_pad_len=self.cfg.prediction.max_new_seconds * self.cfg.prediction.input_sample_rate,
            prompt_tokens=prompt_tokens,
            prompt_token_lens=prompt_token_lens,
        )
        prediction["sample_id"] = batch["sample_id"]
        return prediction

    def _get_bos_embedding(self) -> torch.Tensor:
        """Get BOS embedding for AR decoding."""
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    def _get_asr_bos_embedding(self) -> torch.Tensor:
        """Get ASR BOS embedding for AR decoding."""
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_asr_tokens(text_bos)
        return input_embeds

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),
                    desired_input_layouts=(Shard(1),),
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                }

                attn_layer = transformer_block.self_attn

                try:
                    config = self.llm.config

                    num_attention_heads = getattr(config, 'num_attention_heads', None)
                    num_key_value_heads = getattr(config, 'num_key_value_heads', None)
                    hidden_size = getattr(config, 'hidden_size', None)

                    if all([num_attention_heads, num_key_value_heads, hidden_size]):
                        for attr_name, val in [
                            ("num_attention_heads", num_attention_heads),
                            ("num_key_value_heads", num_key_value_heads),
                            ("hidden_size", hidden_size),
                        ]:
                            if val % tp_mesh.size() != 0:
                                logging.warning(
                                    f"config.{attr_name}={val} is not divisible by {tp_mesh.size()=}: "
                                    f"set a different tensor parallelism size to avoid errors."
                                )

                        if hasattr(attn_layer, 'num_heads'):
                            attn_layer.num_heads = num_attention_heads // tp_mesh.size()
                        elif hasattr(attn_layer, 'num_attention_heads'):
                            attn_layer.num_attention_heads = num_attention_heads // tp_mesh.size()

                        if hasattr(attn_layer, 'num_key_value_heads'):
                            attn_layer.num_key_value_heads = num_key_value_heads // tp_mesh.size()

                        if hasattr(attn_layer, 'hidden_size'):
                            attn_layer.hidden_size = hidden_size // tp_mesh.size()

                        logging.info(
                            f"Configured tensor parallel for attention: "
                            f"heads={num_attention_heads // tp_mesh.size()}, "
                            f"kv_heads={num_key_value_heads // tp_mesh.size()}, "
                            f"hidden_size={hidden_size // tp_mesh.size()}"
                        )
                    else:
                        raise AttributeError("Required config attributes not found")

                except Exception as e:
                    logging.warning(f"Failed to configure tensor parallel using config: {e}")
                    logging.warning("Falling back to attention layer attributes...")

                    try:
                        for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                            if hasattr(attn_layer, attr):
                                val = getattr(attn_layer, attr)
                                if val % tp_mesh.size() != 0:
                                    logging.warning(
                                        f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                                        f"set a different tensor parallelism size to avoid errors."
                                    )
                                setattr(attn_layer, attr, val // tp_mesh.size())
                    except Exception as fallback_e:
                        logging.warning(f"Both config and fallback methods failed: {fallback_e}")
                        logging.warning("Skipping tensor parallel configuration for this attention layer")

            for m in (self.lm_head,):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            if self.predict_user_text:
                self.asr_head = fully_shard(self.asr_head, **fsdp_config)
                self.embed_asr_tokens = fully_shard(self.embed_asr_tokens, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info("Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)
