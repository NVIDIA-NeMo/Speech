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
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from lhotse import fastcopy
from lhotse.serialization import SequentialJsonlWriter
from lightning import LightningModule
from omegaconf import DictConfig, open_dict
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import GenerationConfig

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.dataloader import tokenize_with_prompt
from nemo.collections.common.data.lhotse.text_adapters import TextTurn
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.models.salm import _resolve_audios_in_prompt, replace_placeholders_and_build_targets
from nemo.collections.speechlm2.modules.perception import AudioTranscriptionPerceptionModule
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging


class SALMWithAsrDecoder(LightningModule, HFHubMixin):
    def __init__(self, cfg) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to SALM as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.audio_locator_tag = self.cfg.audio_locator_tag

        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        self.llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights)
        if not hasattr(self.llm, "model") and hasattr(self.llm, "backbone"):
            type(self.llm).model = property(lambda self: self.backbone)
        if not hasattr(self.llm.model, "embed_tokens") and hasattr(self.llm.model, "embeddings"):
            self.llm.model.embed_tokens = self.llm.model.embeddings
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        setup_speech_encoder_with_asr(self, pretrained_weights=self.cfg.pretrained_weights)
        assert isinstance(self.perception, AudioTranscriptionPerceptionModule)

        # Load pretrained weights if provided
        if (init_from_path := self.cfg.get("init_from_path", None)) is not None:
            init_from_path = Path(init_from_path)
            assert init_from_path.is_dir(), "init_from_path must be a directory containing HF checkpoint"
            logging.warning(f"Loading pretrained weights from {str(init_from_path)}")
            from safetensors import safe_open

            tensors = {}
            with safe_open(init_from_path / "model.safetensors", framework="pt") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            missing_keys, unexpected_keys = self.load_state_dict(tensors, strict=False)
            logging.warning(f"Missing keys: {missing_keys}")
            logging.warning(f"Unexpected keys: {unexpected_keys}")

        maybe_install_lora(self)

        self._use_fsdp = False
        self._use_tp = False

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
        ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.
        """
        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        encoded, encoded_len = self.perception.forward_encoder(
            input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
        )
        asr_hyps = self.perception.transcribe_encoded(encoded=encoded, encoded_len=encoded_len)
        # During training, we randomly drop the transcript
        for hyp in asr_hyps:
            if self.training and random.random() < self.cfg.get("asr_transcript_drop_prob", 0.0):
                hyp.text = ""
        asr_tokens = [
            torch.as_tensor(self.tokenizer.text_to_ids(f">> {hyp.text} <<" if hyp.text else ">> <<"))
            for hyp in asr_hyps
        ]
        asr_tokens_len = [at.shape[0] for at in asr_tokens]
        asr_tokens = torch.cat(asr_tokens, dim=0).unsqueeze(0).to(self.device)
        transcript_embs = torch.split(self.embed_tokens(asr_tokens).squeeze(0), asr_tokens_len, dim=0)
        audio_embs, audio_emb_lens = self.perception(encoded=encoded, encoded_len=encoded_len)
        audio_embs = [
            torch.cat([aemb[:aemblen], temb], dim=0)
            for aemb, aemblen, temb in zip(audio_embs, audio_emb_lens, transcript_embs)
        ]
        input_ids_to_embed = torch.where(batch["input_ids"] == self.audio_locator_tag_id, 0, batch["input_ids"])
        text_embs = self.embed_tokens(input_ids_to_embed)
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
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence lenght shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]

        return {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
        num_frames = (inputs["target_ids"] != -100).long().sum()
        with loss_parallel():
            loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["target_ids"].flatten(0, 1),
                    reduction="sum",
                    ignore_index=-100,
                )
                / num_frames
            )

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": (batch["input_ids"] != self.text_pad_id).long().sum() / batch["input_ids"].numel(),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses = defaultdict(list)
        self._partial_accuracies = defaultdict(list)

        # collect generations per validation set (per-rank)
        self._val_generations = defaultdict(list)

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

        # Gather and write generations to a single file per dataset (rank 0 only)
        if self.cfg.get("val_save_path", None) is not None:
            dist = torch.distributed
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                gathered = [None for _ in range(world_size)]
                dist.all_gather_object(gathered, dict(self._val_generations))
                is_global_zero = dist.get_rank() == 0
            else:
                gathered = [dict(self._val_generations)]
                is_global_zero = True

            if is_global_zero:
                merged = defaultdict(list)
                for per_rank_dict in gathered:
                    for name, items in per_rank_dict.items():
                        merged[name].extend(items)

                val_save_path = Path(self.cfg.val_save_path) / f"{self.global_step:06d}"
                val_save_path.mkdir(parents=True, exist_ok=True)
                for name, items in merged.items():
                    out_path = val_save_path / f"{name}.jsonl"
                    with SequentialJsonlWriter(out_path) as writer:
                        for obj in items:
                            writer.write(obj)

        self._val_generations.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            try:
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

            except Exception as e:
                # Skip the dataset if there is an error, e.g., the dataset does not have answers
                logging.warning_once(f"Error in validation step for dataset {name}: {e}")

            # Run autoregressive generation and collect results (writing happens at epoch end)
            if self.cfg.get("val_save_path", None) is not None:
                convs_no_answer = [strip_response_if_any(conv) for conv in dataset_batch["conversations"]]
                convs_no_answer = [
                    tokenize_with_prompt(conv, self.tokenizer, self.cfg.prompt_format) for conv in convs_no_answer
                ]
                answer_ids = self.generate(
                    prompts=left_collate_vectors(
                        [c.input_ids for c in convs_no_answer], padding_value=self.text_pad_id
                    ).to(self.device),
                    audios=dataset_batch["audios"].to(self.device, non_blocking=True),
                    audio_lens=dataset_batch["audio_lens"].to(self.device, non_blocking=True),
                    generation_config=GenerationConfig(
                        max_new_tokens=128,
                        bos_token_id=self.text_bos_id,
                        eos_token_id=[self.text_eos_id],
                        pad_token_id=self.text_pad_id,
                        do_sample=False,
                        num_beams=1,  # greedy decoding
                    ),
                )
                answer_ids = answer_ids.cpu()
                answer_ids = [parse_hyp(ans, [self.text_eos_id]) for ans in answer_ids]
                batch_answers = [self.tokenizer.ids_to_text(ans) for ans in answer_ids]
                for conv, ans in zip(convs_no_answer, batch_answers):
                    conv.turns.append(TextTurn(role="assistant", value=ans))
                    for k, v in list(conv.custom.items()):
                        if isinstance(v, torch.Tensor):
                            del conv.custom[k]
                    self._val_generations[name].append(conv.to_dict())

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        generation_config: GenerationConfig = None,
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
            generation_kwargs: Keyword arguments passed directly to the underlying LLM's ``generate`` method.
        """
        # Encode prompt dicts into int token ids.
        if isinstance(prompts, torch.Tensor):
            tokens = prompts
        else:
            if (
                maybe_audio := _resolve_audios_in_prompt(prompts, sampling_rate=self.sampling_rate, device=self.device)
            ) is not None:
                assert (
                    audios is None and audio_lens is None
                ), "Audios cannot be provided via ``prompts`` and ``audios``/``audio_lens`` arguments simultaneously."
                audios, audio_lens = maybe_audio
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt)["input_ids"] for prompt in prompts],
                padding_value=self.text_pad_id,
            ).to(self.device)
        tokens_to_embed = tokens.where(tokens != self.audio_locator_tag_id, 0)
        token_embeds = self.embed_tokens(tokens_to_embed)

        if audios is not None:
            # Process audio when available
            # TODO: temporary workaround to perform batch_size=1 inference for audio encoder
            #   due to accuracy issues at bs>1
            # audio_embeds, audio_embed_lens = self.perception(audios, audio_lens)
            # audio_embeds = [audio_embeds[i, :elen] for i, elen in enumerate(audio_embed_lens)]

            encoded, encoded_len = self.perception.forward_encoder(input_signal=audios, input_signal_length=audio_lens)
            asr_hyps = self.perception.transcribe_encoded(encoded=encoded, encoded_len=encoded_len)
            asr_tokens = [
                torch.as_tensor(self.tokenizer.text_to_ids(f">> {hyp.text} <<" if hyp.text else ">> <<"))
                for hyp in asr_hyps
            ]
            asr_tokens_len = [at.shape[0] for at in asr_tokens]
            asr_tokens = torch.cat(asr_tokens, dim=0).unsqueeze(0).to(self.device)
            transcript_embs = torch.split(self.embed_tokens(asr_tokens).squeeze(0), asr_tokens_len, dim=0)
            audio_embeds, audio_embed_lens = self.perception(encoded=encoded, encoded_len=encoded_len)
            audio_embeds = [
                torch.cat([aemb[:aemblen], temb], dim=0)
                for aemb, aemblen, temb in zip(audio_embeds, audio_embed_lens, transcript_embs)
            ]
            # Insert audio embeddings into relevant positions in text embeddings.
            input_embeds, _, attention_mask = replace_placeholders_and_build_targets(
                input_ids=tokens,
                embeds=token_embeds,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                replacements=audio_embeds,
                target_ids=None,
            )
        else:
            # Text-only with embeddings - no audio placeholders to replace
            input_embeds = token_embeds
            attention_mask = tokens != self.text_pad_id

        generation_inputs = {"inputs_embeds": input_embeds, "attention_mask": attention_mask}
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                pad_token_id=self.text_pad_id,
            )
        # Generate the answers using HF Generate API.
        # Note: we need to put the text embedding layer back to the LLM for processing.
        with move_embedding(self):
            answer_tokens = self.llm.generate(
                **generation_inputs,
                **generation_kwargs,
                generation_config=generation_config,
            )
        return answer_tokens

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            # TODO: Distributing embeddings with TP in this setup is tricky
            #       because we're adding with the output of a non-parallelized
            #       speech encoder.
            # for m in (self.embed_tokens, self.embed_audio_tokens):
            #     parallelize_module(
            #         m,
            #         tp_mesh,
            #         ColwiseParallel(
            #             # input_layouts=Shard(1),
            #             # # Optional: Shard the output along the class dimension to compute the loss in parallel.
            #             # # See `loss_parallel` in `train.py`
            #             # output_layouts=Shard(1),
            #             # use_local_output=False,
            #         ),
            #     )

            # # Parallelize the first embedding and the last linear out projection
            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            # Parallelize each transformer block
            for transformer_block in llm.model.layers:
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
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                # Apply the plan for the current transformer block
                parallelize_module(transformer_block, tp_mesh, plan)

            parallelize_module(
                llm.lm_head,
                tp_mesh,
                ColwiseParallel(
                    input_layouts=Shard(1),
                    # Optional: Shard the output along the class dimension to compute the loss in parallel.
                    # See `loss_parallel` in `train.py`
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
            )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1  # Hybrid-sharding not supported
            self._use_fsdp = True
            fsdp_config = {"mesh": dp_mesh}
            for idx, layer in enumerate(llm.model.layers):
                llm.model.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            llm.lm_head = fully_shard(llm.lm_head, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            # self.perception.modality_adapter = fully_shard(self.perception.modality_adapter, **fsdp_config)
            # self.perception.asr.preprocessor = fully_shard(self.perception.asr.preprocessor **fsdp_config)
            # self.perception.asr.encoder = fully_shard(self.perception.asr.encoder, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            register_fsdp_forward_method(self.perception, "forward_encoder")
            register_fsdp_forward_method(self.perception, "transcribe_encoded")

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


def setup_speech_encoder_with_asr(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.
    """
    with open_dict(model.cfg):
        model.cfg.output_dim = model.llm.config.hidden_size
    model.perception = AudioTranscriptionPerceptionModule(model.cfg.perception, model.cfg.pretrained_asr).train()

    from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs

    WithOptionalCudaGraphs.disable_cuda_graphs_recursive(model.perception.asr, attribute_path="decoding.decoding")


def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    end = torch.isin(answer, torch.tensor(eos_tokens)).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]


def strip_response_if_any(
    conversation: NeMoMultimodalConversation,
) -> NeMoMultimodalConversation:
    turns = conversation.turns
    while turns[-1].role == "assistant":
        turns = turns[:-1]
    return fastcopy(conversation, turns=turns)
