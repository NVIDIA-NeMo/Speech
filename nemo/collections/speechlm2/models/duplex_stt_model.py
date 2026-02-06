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
import random
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import soundfile as sf
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn
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
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.common.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.parts.text_utils import tokens_to_str
from nemo.collections.speechlm2.parts.augmentation import DEFAULT_CODEC_SETTINGS, AudioAugmenter
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.label_prep import prepare_text_and_asr_labels
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.empty_text import EmptyTextMetric
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.turn_taking import TurnTakingMetrics
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


class DuplexSTTModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()

        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)
        self.predict_user_text = self.cfg.get("predict_user_text", False)

        # Load LLM first
        llm = load_pretrained_hf(
            self.cfg.pretrained_llm,
            pretrained_weights=self.cfg.pretrained_weights,
            trust_remote_code=self.cfg.get("trust_remote_code", True),
        ).train()

        # Handle different model types with all their specific configurations
        if 'Nemotron' in self.cfg.pretrained_llm:
            # ====== NEMOTRON-SPECIFIC HANDLING ======
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token = '<SPECIAL_12>'

            self.llm = getattr(llm, self.cfg.get("base_model_name", "backbone"))
            self.lm_head = llm.lm_head
            embed_tokens_name = self.cfg.get("embed_tokens_name", "embeddings")
            self.embed_tokens = getattr(self.llm, embed_tokens_name)

            delattr(self.llm, embed_tokens_name)
        elif 'Qwen2.5' in self.cfg.pretrained_llm:
            # ====== QWEN2.5-SPECIFIC HANDLING ======
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'

            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'

            self.llm = llm.model
            self.lm_head = llm.lm_head
            self.embed_tokens = self.llm.embed_tokens

            del self.llm.embed_tokens
        else:
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            self.llm = llm.model
            self.lm_head = llm.lm_head
            self.embed_tokens = self.llm.embed_tokens
            del self.llm.embed_tokens

        if self.predict_user_text:
            self.asr_head = copy.deepcopy(self.lm_head)
            self.embed_asr_tokens = copy.deepcopy(self.embed_tokens)
        self.user_bos_id = self.tokenizer.text_to_ids('^')[0]
        self.user_eos_id = self.tokenizer.text_to_ids('$')[0]

        maybe_install_lora(self)

        # Load the pretrained streaming ASR model
        setup_speech_encoder(self)

        if self.cfg.get("pretrained_perception_from_s2s", None):
            self.init_perception_from_another_s2s_checkpoint(self.cfg.pretrained_perception_from_s2s)

        if self.cfg.get("pretrained_s2s_model", None):
            logging.info(f"Loading pretrained s2s model from {self.cfg.pretrained_s2s_model}")
            if os.path.isdir(self.cfg.pretrained_s2s_model) and self.cfg.get("incremental_loading", False):
                # Hugging Face format
                import gc

                from safetensors import safe_open

                # Load tensors incrementally to avoid OOM
                model_state_dict = self.state_dict()
                loaded_keys = []
                missing_keys = []

                with safe_open(
                    os.path.join(self.cfg.pretrained_s2s_model, "model.safetensors"), framework="pt", device="cpu"
                ) as f:
                    available_keys = f.keys()
                    for key in available_keys:
                        if key in model_state_dict:
                            # Load tensor and copy to model parameter
                            tensor = f.get_tensor(key)
                            model_state_dict[key].copy_(tensor)
                            loaded_keys.append(key)
                            del tensor  # Free memory immediately
                        else:
                            missing_keys.append(key)

                        # Periodic garbage collection for very large models
                        if len(loaded_keys) % 100 == 0:
                            gc.collect()

                logging.info(f"Loaded {len(loaded_keys)} tensors from pretrained model")
                if missing_keys:
                    logging.warning(f"Keys in checkpoint but not in model: {len(missing_keys)} keys")

                del model_state_dict
                gc.collect()
            else:
                self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)

        self._use_fsdp = False
        self._use_tp = False

        # Initialize audio augmenter if any augmentation is enabled
        if (
            self.cfg.get('use_noise_aug', None)
            or self.cfg.get('use_room_ir_aug', None)
            or self.cfg.get('use_mic_ir_aug', None)
            or self.cfg.get('use_codec_aug', None)
        ):
            self.audio_augmenter = AudioAugmenter(sample_rate=self.source_sample_rate)

    def init_perception_from_another_s2s_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            elif os.path.isdir(checkpoint_path):
                logging.info(f"Loading from HuggingFace format directory: {checkpoint_path}")
                pretrained_model = self.__class__.from_pretrained(checkpoint_path)
                checkpoint_state = pretrained_model.state_dict()
                del pretrained_model
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = {
                k.replace("perception.", ""): v for k, v in checkpoint_state.items() if "perception." in k
            }
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.perception.state_dict())
            self.perception.load_state_dict(checkpoint_state, strict=True)

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            elif os.path.isdir(checkpoint_path):
                logging.info(f"Loading from HuggingFace format directory: {checkpoint_path}")
                pretrained_model = self.__class__.from_pretrained(checkpoint_path)
                checkpoint_state = pretrained_model.state_dict()
                del pretrained_model
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

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
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
        input_audio_tokens=None,
        seq_mask=None,
        target_text_tokens=None,
        modality_adapter_emb=None,
        speaker_encoder_emb=None,
    ) -> dict[str, Tensor]:
        """
        Text prediction only (audio_loss_weight=0).
        """
        # Handle different cache parameter names for different models
        if 'Nemotron' in self.cfg.pretrained_llm:
            kwargs = {
                "inputs_embeds": input_embeds,
                "return_dict": True,
                "use_cache": cache is not None,
            }
            if cache is not None:
                kwargs['use_cache'] = True
                kwargs[self.cfg.get("cache_key", "past_key_values")] = cache
            out = self.llm(**kwargs)
        else:
            out = self.llm(
                inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
            )

        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])

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

    def _is_noise_augmentation_dataset(self, formatter: str) -> bool:
        if self.cfg.get('force_use_noise_augmentation', False):
            return True
        return formatter != 's2s_duplex_overlap_as_s2s_duplex' and formatter != 'nemo_tarred_to_duplex'

    def _maybe_zero_out_scale_for_asr(
        self, loss_scale: torch.Tensor, text_labels: torch.Tensor, batch: dict
    ) -> torch.Tensor:
        """
        Zero out the loss scale after text_bos_id token for ASR datasets.
        """
        if batch['formatter'][0] == 'nemo_tarred_to_duplex':
            for i in range(text_labels.shape[0]):
                bos_indices = (text_labels[i] == self.text_bos_id).nonzero(as_tuple=True)
                if bos_indices[0].numel() > 0:
                    bos_idx = bos_indices[0][0].item()
                    loss_scale[i, bos_idx + 1 :, :] = 0
        return loss_scale

    def prepare_inputs(self, batch: dict):
        # Apply augmentations in order: noise -> room IR -> mic IR -> codec
        # Each augmentation has its own independent condition and flag

        # 1. Noise augmentation (controlled by use_noise_aug flag)
        if (
            self.cfg.get('use_noise_aug', None)
            and self.training
            and self._is_noise_augmentation_dataset(batch["formatter"][0])
        ):
            noise_prob = self.cfg.get('noise_prob', 0.99)
            noise_min_snr = self.cfg.get('noise_min_snr', 20)
            noise_max_snr = self.cfg.get('noise_max_snr', 50)
            noise_path = self.cfg.get('noise_aug_path', None)
            noise_path_name = "*"

            if noise_prob and random.random() < noise_prob and noise_path:
                batch["source_audio"] = self.audio_augmenter.add_noise_to_batch(
                    batch["source_audio"],
                    os.path.join(noise_path, noise_path_name),
                    snr_db=random.randint(noise_min_snr, noise_max_snr),
                    noise_prob_scale_user=self.cfg.get('noise_prob_scale_user', 0.3),
                    noise_prob_scale_user_min_snr=self.cfg.get('noise_prob_scale_user_min_snr', -15),
                    noise_prob_scale_user_max_snr=self.cfg.get('noise_prob_scale_user_max_snr', 24),
                    snr_measure_dur=self.cfg.get('snr_measure_dur', 0.0),
                    noise_resample=self.cfg.get('noise_resample', True),
                    noise_prob_low_pass=self.cfg.get('noise_prob_low_pass', 0.1),
                )

        # 2. Room impulse response augmentation
        if (
            self.cfg.get('use_room_ir_aug', None)
            and self.training
            and self._is_noise_augmentation_dataset(batch["formatter"][0])
        ):
            roomir_prob = self.cfg.get('roomir_prob', 0.0)
            roomir_path = self.cfg.get('roomir_aug_path', None)

            if roomir_prob > 0 and roomir_path and random.random() < roomir_prob:
                batch["source_audio"] = self.audio_augmenter.add_room_ir_to_batch(
                    batch["source_audio"],
                    batch["source_audio_lens"],
                    roomir_path,
                    use_loudness_norm=self.cfg.get('roomir_use_loudness_norm', True),
                )

        # 3. Microphone impulse response augmentation
        if (
            self.cfg.get('use_mic_ir_aug', None)
            and self.training
            and self._is_noise_augmentation_dataset(batch["formatter"][0])
        ):
            micir_prob = self.cfg.get('micir_prob', 0.0)
            micir_path = self.cfg.get('micir_aug_path', None)

            if micir_prob > 0 and micir_path and random.random() < micir_prob:
                batch["source_audio"] = self.audio_augmenter.add_mic_ir_to_batch(
                    batch["source_audio"],
                    batch["source_audio_lens"],
                    micir_path,
                    use_loudness_norm=self.cfg.get('micir_use_loudness_norm', True),
                )

        # 4. Codec augmentation
        if (
            self.cfg.get('use_codec_aug', None)
            and self.training
            and self._is_noise_augmentation_dataset(batch["formatter"][0])
        ):
            codec_prob = self.cfg.get('codec_prob', 0.0)
            codec_settings = self.cfg.get('codec_settings', None)

            if codec_prob > 0 and random.random() < codec_prob:
                # Use custom codec settings if provided, otherwise use defaults
                if codec_settings is None:
                    codec_settings = DEFAULT_CODEC_SETTINGS
                batch["source_audio"] = self.audio_augmenter.add_codec_to_batch(
                    batch["source_audio"],
                    batch["source_audio_lens"],
                    codec_settings,
                )

        source_encoded, source_encoded_lens, _ = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )

        target_tokens = batch["target_tokens"]

        if "prompt_tokens" in batch:
            prompt_embedded = self.embed_tokens(batch["prompt_tokens"])
            B, max_prompt_len, H = prompt_embedded.shape
            T_src = source_encoded.shape[1]
            T_tgt = target_tokens.shape[1]

            new_source_encoded = torch.zeros(
                B, max_prompt_len + T_src, H, dtype=source_encoded.dtype, device=source_encoded.device
            )
            new_target_tokens = torch.full(
                (B, max_prompt_len + T_tgt), self.text_pad_id, dtype=target_tokens.dtype, device=target_tokens.device
            )
            # If source_tokens are present (used by ASR head for user text prediction),
            # prepend PADs to align ASR labels with the prompt span as well.
            if "source_tokens" in batch:
                source_tokens = batch["source_tokens"]
                T_src_tok = source_tokens.shape[1]
                new_source_tokens = torch.full(
                    (B, max_prompt_len + T_src_tok),
                    self.text_pad_id,
                    dtype=source_tokens.dtype,
                    device=source_tokens.device,
                )

            # For each item, insert prompt and original data at correct offsets
            for i, prompt_len in enumerate(batch["prompt_token_lens"]):
                prompt_len = prompt_len.item()

                if prompt_len > 0:
                    new_source_encoded[i, :prompt_len, :] = prompt_embedded[i, :prompt_len, :]

                src_len = source_encoded_lens[i].item()
                new_source_encoded[i, prompt_len : prompt_len + src_len, :] = source_encoded[i, :src_len, :]

                tgt_len = batch["target_token_lens"][i].item()
                new_target_tokens[i, prompt_len : prompt_len + tgt_len] = target_tokens[i, :tgt_len]

                source_encoded_lens[i] = prompt_len + src_len
                batch["target_token_lens"][i] = prompt_len + tgt_len

                # If source_tokens exist, copy them after the prompt and update lengths
                if "source_tokens" in batch:
                    src_len = batch["source_token_lens"][i].item()
                    new_source_tokens[i, prompt_len : prompt_len + src_len] = source_tokens[i, :src_len]
                    batch["source_token_lens"][i] = prompt_len + src_len

            source_encoded = new_source_encoded
            target_tokens = new_target_tokens
            if "source_tokens" in batch:
                batch["source_tokens"] = new_source_tokens

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
            predict_user_text=self.predict_user_text,
            user_bos_id=self.user_bos_id,
            user_eos_id=self.user_eos_id,
            text_pad_id=self.text_pad_id,
            text_bos_id=self.text_bos_id,
            text_eos_id=self.text_eos_id,
            advance_text_channel_by=self.advance_text_channel_by,
            use_tp=self._use_tp,
            device_mesh=self.device_mesh if self._use_tp else None,
        )

        source_encoded = inputs["source_encoded"]
        text_inputs = inputs["text_inputs"]
        text_labels = inputs["text_labels"]
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
            for i in range(batch["target_token_lens"].size(0)):
                speech_end_idx = batch["target_token_lens"][i]
                seq_mask[i, speech_end_idx:, :] = 0

        loss_scale = seq_mask.clone().float()
        asr_loss_scale = seq_mask.clone().float()
        if self.cfg.get("token_loss_weight"):
            token_weights = self.cfg.token_loss_weight
            pad_weight = token_weights.get("pad", 1.0)
            bos_weight = token_weights.get("bos", 1.0)
            eos_weight = token_weights.get("eos", 1.0)
            text_weight = token_weights.get("text", 1.0)

            loss_scale = torch.where(
                text_labels.unsqueeze(-1) == self.text_pad_id,
                pad_weight,
                torch.where(
                    text_labels.unsqueeze(-1) == self.text_bos_id,
                    bos_weight,
                    torch.where(text_labels.unsqueeze(-1) == self.text_eos_id, eos_weight, text_weight),
                ),
            )
            # Don't penalize the agent replies for ASR training
            loss_scale = self._maybe_zero_out_scale_for_asr(loss_scale, text_labels, batch)
            if self.predict_user_text:
                asr_loss_scale = torch.where(
                    asr_labels.unsqueeze(-1) == self.text_pad_id,
                    pad_weight,
                    torch.where(
                        asr_labels.unsqueeze(-1) == self.user_bos_id,
                        bos_weight,
                        torch.where(asr_labels.unsqueeze(-1) == self.user_eos_id, eos_weight, text_weight),
                    ),
                )

        ans = {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": source_encoded_lens - 1,
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

    def on_train_epoch_start(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.bleu = BLEU().reset()

        self.turn_taking_metrics = TurnTakingMetrics(
            eos_token_id=self.tokenizer.text_to_ids('$')[0],
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

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
                prompt_tokens=prompt_tokens,
                prompt_token_lens=prompt_token_lens,
            )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])

            if "source_tokens" in dataset_batch and results["tokens_text"] is not None:
                self.turn_taking_metrics.update(
                    name=name, source_tokens=dataset_batch["source_tokens"], pred_tokens=results["tokens_text"]
                )

            pred_turns_list = self._split_agent_tokens_into_turns(results["tokens_text"])

            self.results_logger.update(
                name=name,
                refs=dataset_batch["target_texts"],
                hyps=results["text"],
                asr_hyps=None,
                samples_id=dataset_batch['sample_id'],
                pred_audio=None,
                user_audio=dataset_batch["source_audio"],
                user_audio_sr=self.source_sample_rate,
                src_refs=dataset_batch["source_texts"],
                src_hyps=results["src_text"],
                system_prompt=dataset_batch.get("system_prompt", None),
                source_turns=dataset_batch.get("source_turn_texts"),
                target_turns=dataset_batch.get("target_turn_texts"),
                pred_turns=pred_turns_list,
            )

            if self.cfg.get("eval_text_turn_taking", False):
                import re

                results["text"] = [re.sub(r"<\|.*?\|>", "", s).strip() for s in results["text"]]

            if self.predict_user_text:
                src_text_clean = [s.replace("^", " ").replace("$", " ") for s in results["src_text"]]
                self.src_bleu.update(name=name, refs=dataset_batch["source_texts"], hyps=src_text_clean)
                self.src_wer.update(name=name, refs=dataset_batch["source_texts"], hyps=src_text_clean)
                self.empty_user_text.update(name=name, hyps=results["src_text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def on_predict_epoch_start(self) -> None:
        return self.on_train_epoch_start()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch = batch["audio_data"]

        force_bos_positions = None
        force_bos_num_tokens_after_user_eos = self.cfg.prediction.get("force_bos_num_tokens_after_user_eos", None)
        if force_bos_num_tokens_after_user_eos is not None:
            force_bos_positions = []
            for cur_source_tokens in batch["source_tokens"]:
                tmp = torch.where(cur_source_tokens == self.text_eos_id)[0]
                if len(tmp) > 0:
                    force_bos_positions.append(tmp[0].item() + force_bos_num_tokens_after_user_eos)
                else:
                    force_bos_positions.append(None)

        prompt_tokens = batch.get("prompt_tokens", None)
        prompt_token_lens = batch.get("prompt_token_lens", None)

        prediction = self.offline_inference(
            batch["source_audio"],
            batch["source_audio_lens"],
            decode_audio=self.cfg.prediction.decode_audio,
            input_pad_len=self.cfg.prediction.max_new_seconds * self.cfg.prediction.input_sample_rate,
            force_bos_positions=force_bos_positions,
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

    def _split_agent_tokens_into_turns(self, tokens_text: torch.Tensor):
        """Split sequence of agent_tokens into turns as detected by text_bos_id and text_eos_id."""
        batch_size, seq_len = tokens_text.shape
        token_duration = 0.08

        turns_list = []

        for b in range(batch_size):
            current_tokens = tokens_text[b].cpu().numpy()

            in_turn = False
            current_turn_start = None
            current_turn_tokens = []
            batch_turns = []

            def _save_current_turn(turn_start, turn_tokens, end_token_idx, is_complete=True):
                if turn_start is None:
                    return

                start_time = turn_start * token_duration
                end_time = (end_token_idx + 1) * token_duration
                duration = end_time - start_time

                if len(turn_tokens) > 0:
                    turn_tokens_filtered = [t for t in turn_tokens if t != self.text_pad_id]
                    text = self.tokenizer.ids_to_text(turn_tokens_filtered)
                else:
                    text = ""

                turn = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "text": text,
                    "token_ids": turn_tokens.copy(),
                    "start_token_idx": turn_start,
                    "end_token_idx": end_token_idx,
                    "num_tokens": len(turn_tokens),
                    "is_complete": is_complete,
                }
                batch_turns.append(turn)

            for t in range(seq_len):
                token_id = current_tokens[t]

                if token_id == self.text_bos_id:
                    if in_turn and current_turn_start is not None:
                        logging.debug(
                            f"Batch {b}: Found BOS at position {t} while already in a turn "
                            f"(started at {current_turn_start}). Saving incomplete turn."
                        )
                        _save_current_turn(
                            current_turn_start, current_turn_tokens, end_token_idx=t - 1, is_complete=False
                        )

                    in_turn = True
                    current_turn_start = t
                    current_turn_tokens = []

                elif token_id == self.text_eos_id:
                    if in_turn:
                        _save_current_turn(current_turn_start, current_turn_tokens, end_token_idx=t, is_complete=True)

                        current_turn_start = None
                        current_turn_tokens = []
                        in_turn = False

                elif token_id == self.text_pad_id:
                    if in_turn:
                        current_turn_tokens.append(token_id)
                else:
                    if in_turn:
                        current_turn_tokens.append(token_id)

            if in_turn and current_turn_start is not None:
                logging.debug(
                    f"Batch {b}: Sequence ended while in a turn (started at {current_turn_start}). "
                    f"Saving incomplete turn."
                )
                _save_current_turn(
                    current_turn_start, current_turn_tokens, end_token_idx=seq_len - 1, is_complete=False
                )

            turns_list.append(batch_turns)

        return turns_list

    def _init_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        input_pad_len: int,
        force_bos_positions,
        prompt_tokens: torch.Tensor,
        prompt_token_lens: torch.Tensor,
    ):
        """Initialize inference resources and prepare inputs."""
        if self.cfg.get("custom_sample_inference", None):
            device = input_signal.device
            audio, sr = sf.read(self.cfg.custom_sample_inference)
            # sf.read returns (samples,) for mono or (samples, channels) for stereo
            # Convert to (channels, samples) format
            if audio.ndim == 1:
                audio = audio[None, :]  # Add channel dimension for mono
            else:
                audio = audio.T  # Transpose to (channels, samples) for stereo
            input_signal = torch.from_numpy(audio).float().to(device)[:1, :]
            input_signal = resample(input_signal, sr, self.source_sample_rate)
            input_signal_lens = torch.tensor([input_signal.size(-1)]).to(device)

        if force_bos_positions is not None:
            assert input_signal.shape[0] == len(
                force_bos_positions
            ), "force_bos_positions must have the same length as batch size"

        if input_pad_len > 0:
            input_signal = torch.nn.functional.pad(input_signal, (0, input_pad_len), mode='constant', value=0)
            input_signal_lens = input_signal_lens + input_pad_len

        source_encoded, lengths, _ = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )

        B, T_local, H = source_encoded.shape

        if prompt_tokens is not None and prompt_token_lens is not None:
            prompt_embedded = self.embed_tokens(prompt_tokens)
            B_prompt, max_prompt_len, H_prompt = prompt_embedded.shape

            assert B == B_prompt, f"Batch size mismatch: source={B}, prompt={B_prompt}"
            assert H == H_prompt, f"Hidden size mismatch: source={H}, prompt={H_prompt}"

            new_source_encoded = torch.zeros(
                B, max_prompt_len + T_local, H, dtype=source_encoded.dtype, device=source_encoded.device
            )

            for i, prompt_len in enumerate(prompt_token_lens):
                prompt_len = prompt_len.item()

                if prompt_len > 0:
                    new_source_encoded[i, :prompt_len, :] = prompt_embedded[i, :prompt_len, :]

                src_len = lengths[i].item()
                new_source_encoded[i, prompt_len : prompt_len + src_len, :] = source_encoded[i, :src_len, :]

                lengths[i] = prompt_len + src_len

            source_encoded = new_source_encoded
            T_local = source_encoded.shape[1]

        B, T_local, H = source_encoded.shape

        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame_source = source_encoded[:, T_local - 1 : T_local, :]
                pad_source = last_frame_source.repeat(1, T - T_local, 1)
                source_encoded = torch.cat([source_encoded, pad_source], dim=1)
        else:
            T = T_local

        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        use_cache = True
        if 'Nemotron' in self.cfg.pretrained_llm:
            cache = None
            use_cache = False
            logging.info("Using no-cache mode for Nemotron (full history each step)")
        else:
            cache = DynamicCache()
            use_cache = True

        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        if self.predict_user_text:
            gen_asr = torch.empty(B, T, device=self.device, dtype=torch.long)
        else:
            gen_asr = None

        if prompt_tokens is not None and prompt_token_lens is not None:
            for i, prompt_len in enumerate(prompt_token_lens):
                prompt_len = prompt_len.item()
                if prompt_len > 0:
                    gen_text[i, :prompt_len] = self.text_pad_id
                    if self.predict_user_text:
                        gen_asr[i, :prompt_len] = self.text_pad_id

        input_embeds[:, 0] += self._get_bos_embedding() * self.cfg.get("duplex_text_channel_weight", 1.0)
        if self.predict_user_text:
            input_embeds[:, 0] += self._get_asr_bos_embedding() * self.cfg.get("duplex_asr_text_weight", 1.0)

        start_gen_pos = 0
        if prompt_token_lens is not None:
            max_prompt_len = prompt_token_lens.max().item()
            start_gen_pos = max_prompt_len

        is_prompt_position_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        if prompt_token_lens is not None:
            for i, prompt_len in enumerate(prompt_token_lens):
                prompt_len_val = prompt_len.item()
                if prompt_len_val > 0:
                    is_prompt_position_mask[i, :prompt_len_val] = True

        return {
            "input_signal": input_signal,
            "input_signal_lens": input_signal_lens,
            "source_encoded": source_encoded,
            "lengths": lengths,
            "B": B,
            "T": T,
            "T_local": T_local,
            "input_embeds": input_embeds,
            "cache": cache,
            "use_cache": use_cache,
            "gen_text": gen_text,
            "gen_asr": gen_asr,
            "start_gen_pos": start_gen_pos,
            "is_prompt_position_mask": is_prompt_position_mask,
        }

    def _step_zero(self, inference_state):
        """Perform inference for the first step (position 0)."""
        ans = self(
            inference_state["input_embeds"][:, :1],
            cache=inference_state["cache"],
            input_audio_tokens=None,
            seq_mask=None,
            target_text_tokens=None,
            modality_adapter_emb=inference_state["source_encoded"][:, :1],
            speaker_encoder_emb=None,
        )

        if inference_state["start_gen_pos"] == 0:
            inference_state["gen_text"][:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
            if self.predict_user_text:
                inference_state["gen_asr"][:, 0] = ans["asr_logits"][:, -1].argmax(dim=-1)

        return ans, inference_state

    def _maybe_apply_forced_turn_taking(self, t, inference_state, is_prompt_position):
        """Apply forced turn-taking rules based on ASR channel tokens."""
        if not self.cfg.get("force_turn_taking", False):
            return

        threshold = self.cfg.get("force_turn_taking_threshold", 40)
        pad_window_steps = self.cfg.get("force_turn_taking_pad_window", 25)

        for batch_idx in range(inference_state["B"]):
            if is_prompt_position[batch_idx]:
                continue

            lookback_start = max(0, t - threshold)
            agent_text_window = inference_state["gen_text"][batch_idx, lookback_start:t]
            current_asr_token = inference_state["gen_asr"][batch_idx, t]

            # ASR EOS or ~1 sec of pad tokens → insert agent BOS if not present in window
            # Skip if we don't have enough tokens at the beginning
            if t < pad_window_steps:
                continue

            pad_lookback_start = t - pad_window_steps
            asr_recent_tokens = inference_state["gen_asr"][batch_idx, pad_lookback_start:t]
            has_pad_window = (asr_recent_tokens == self.text_pad_id).all() if len(asr_recent_tokens) > 0 else False

            # Require that the pad window starts after a non-pad token
            if has_pad_window and pad_lookback_start > 0:
                token_before_window = inference_state["gen_asr"][batch_idx, pad_lookback_start - 1]
                has_pad_window = token_before_window != self.text_pad_id
            elif has_pad_window and pad_lookback_start == 0:
                # If the pad window starts at position 0, it doesn't meet the requirement
                has_pad_window = False

            if current_asr_token == self.tokenizer.eos or has_pad_window:
                # User has finished talking or remains silent for a while
                if not (agent_text_window == self.text_bos_id).any():
                    inference_state["gen_text"][batch_idx, t] = self.text_bos_id
            elif current_asr_token == self.user_bos_id:
                # User has started talking but agent has not stopped yet
                if not (agent_text_window == self.text_eos_id).any():
                    inference_state["gen_text"][batch_idx, t] = self.text_eos_id

    def _step_inference(self, t, inference_state, ans, force_bos_positions):
        """Perform inference for one step t in the autoregressive loop."""
        last_emb = self.embed_tokens(inference_state["gen_text"][:, t - 1]) * self.cfg.get(
            "duplex_text_channel_weight", 1.0
        )
        if self.predict_user_text:
            last_asr_emb = self.embed_asr_tokens(inference_state["gen_asr"][:, t - 1]) * self.cfg.get(
                "duplex_asr_text_weight", 1.0
            )
            last_emb += last_asr_emb
        if force_bos_positions is not None:
            for batch_idx in range(last_emb.shape[0]):
                if (
                    force_bos_positions[batch_idx] == t
                    and not (inference_state["gen_text"][batch_idx, :t] == self.text_bos_id).any()
                ):
                    last_emb[batch_idx] = self.embed_tokens(
                        torch.full((1,), fill_value=self.text_bos_id, device=self.device)
                    ) * self.cfg.get("duplex_text_channel_weight", 1.0)

        inference_state["input_embeds"][:, t] += last_emb

        is_prompt_position = inference_state["is_prompt_position_mask"][:, t]

        if inference_state["use_cache"]:
            ans = self(
                inference_state["input_embeds"][:, t : t + 1],
                cache=ans["cache"],
                input_audio_tokens=None,
                seq_mask=None,
                target_text_tokens=None,
                modality_adapter_emb=inference_state["source_encoded"][:, t : t + 1],
                speaker_encoder_emb=None,
            )
            if not is_prompt_position.all():
                generated_tokens = ans["text_logits"][:, -1].argmax(dim=-1)
                inference_state["gen_text"][:, t] = torch.where(
                    is_prompt_position, inference_state["gen_text"][:, t], generated_tokens
                )
        else:
            ans = self(
                inference_state["input_embeds"][:, : t + 1],
                cache=None,
                input_audio_tokens=None,
                seq_mask=None,
                target_text_tokens=None,
                modality_adapter_emb=inference_state["source_encoded"][:, : t + 1],
                speaker_encoder_emb=None,
            )
            if not is_prompt_position.all():
                generated_tokens = ans["text_logits"][:, -1].argmax(dim=-1)
                inference_state["gen_text"][:, t] = torch.where(
                    is_prompt_position, inference_state["gen_text"][:, t], generated_tokens
                )

        if self.predict_user_text:
            if not is_prompt_position.all():
                generated_asr = ans["asr_logits"][:, -1].argmax(dim=-1)
                inference_state["gen_asr"][:, t] = torch.where(
                    is_prompt_position, inference_state["gen_asr"][:, t], generated_asr
                )
                self._maybe_apply_forced_turn_taking(t, inference_state, is_prompt_position)

        return ans

    def _post_inference(self, inference_state, prompt_token_lens):
        """Post-process inference results and prepare output."""
        gen_text = inference_state["gen_text"]
        gen_asr = inference_state["gen_asr"]
        lengths = inference_state["lengths"]
        T_local = inference_state["T_local"]
        T = inference_state["T"]
        B = inference_state["B"]

        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            if self.predict_user_text:
                gen_asr = gen_asr[:, :T_local]

        if self.predict_user_text:
            gen_text_src = gen_asr
            src_text_cleaned = [self.tokenizer.ids_to_text(gen_text_src[b]) for b in range(gen_text_src.shape[0])]
        else:
            gen_text_src = None
            src_text_cleaned = None

        if prompt_token_lens is not None:
            max_prompt_len = prompt_token_lens.max().item()
            if max_prompt_len > 0:
                current_T = gen_text.shape[1]
                gen_text_trimmed = torch.zeros(B, current_T - max_prompt_len, device=self.device, dtype=torch.long)
                if self.predict_user_text:
                    gen_asr_trimmed = torch.zeros(B, current_T - max_prompt_len, device=self.device, dtype=torch.long)
                lengths_trimmed = lengths.clone()

                for i, prompt_len in enumerate(prompt_token_lens):
                    prompt_len_val = prompt_len.item()
                    actual_len = lengths[i].item() - prompt_len_val
                    if actual_len > 0:
                        gen_text_trimmed[i, :actual_len] = gen_text[i, prompt_len_val : prompt_len_val + actual_len]
                        if self.predict_user_text:
                            gen_asr_trimmed[i, :actual_len] = gen_asr[i, prompt_len_val : prompt_len_val + actual_len]
                    lengths_trimmed[i] = actual_len

                gen_text = gen_text_trimmed
                if self.predict_user_text:
                    gen_asr = gen_asr_trimmed
                    gen_text_src = gen_asr
                lengths = lengths_trimmed

        ans = {
            "text": tokens_to_str(
                gen_text,
                lengths,
                tokenizer=self.tokenizer,
                pad_id=self.text_pad_id,
                eval_text_turn_taking=self.cfg.get("eval_text_turn_taking", True),
            ),
            "src_text": src_text_cleaned,
            "tokens_text_src": gen_text_src,
            "tokens_text": gen_text,
            "tokens_audio": None,
            "tokens_len": lengths,
            "source_audio": inference_state["input_signal"],
            "source_audio_len": inference_state["input_signal_lens"],
        }

        return ans

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
        input_pad_len: int = 0,
        force_bos_positions=None,
        prompt_tokens: torch.Tensor = None,
        prompt_token_lens: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction (text only).
        """
        inference_state = self._init_inference(
            input_signal, input_signal_lens, input_pad_len, force_bos_positions, prompt_tokens, prompt_token_lens
        )

        ans, inference_state = self._step_zero(inference_state)

        for t in range(1, inference_state["T"]):
            ans = self._step_inference(t, inference_state, ans, force_bos_positions)

        return self._post_inference(inference_state, prompt_token_lens)

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
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)
