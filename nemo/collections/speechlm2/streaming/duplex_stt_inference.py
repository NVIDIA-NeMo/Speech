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

import soundfile as sf
import torch
import torch.distributed as dist
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.parts.text_utils import tokens_to_str
from nemo.utils import logging


class DuplexSTTStreamingInference:
    """Streaming inference engine for DuplexSTT model."""

    def __init__(self, model):
        """Initialize the streaming inference engine with a reference to the parent model.

        Args:
            model: The DuplexSTTModel instance that owns this inference engine.
        """
        self.model = model

    def _init_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        input_pad_len: int,
        prompt_tokens: torch.Tensor,
        prompt_token_lens: torch.Tensor,
    ):
        """Initialize inference resources and prepare inputs."""
        if self.model.cfg.get("custom_sample_inference", None):
            device = input_signal.device
            audio, sr = sf.read(self.model.cfg.custom_sample_inference)
            # sf.read returns (samples,) for mono or (samples, channels) for stereo
            # Convert to (channels, samples) format
            if audio.ndim == 1:
                audio = audio[None, :]  # Add channel dimension for mono
            else:
                audio = audio.T  # Transpose to (channels, samples) for stereo
            input_signal = torch.from_numpy(audio).float().to(device)[:1, :]
            input_signal = resample(input_signal, sr, self.model.source_sample_rate)
            input_signal_lens = torch.tensor([input_signal.size(-1)]).to(device)

        if input_pad_len > 0:
            input_signal = torch.nn.functional.pad(input_signal, (0, input_pad_len), mode='constant', value=0)
            input_signal_lens = input_signal_lens + input_pad_len

        source_encoded, lengths, _ = self.model.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )

        B, T_local, H = source_encoded.shape

        if prompt_tokens is not None and prompt_token_lens is not None:
            prompt_embedded = self.model.embed_tokens(prompt_tokens)
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

        if self.model._use_fsdp:
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
        input_embeds *= self.model.cfg.get("duplex_user_channel_weight", 1.0)

        use_cache = True
        if 'Nemotron' in self.model.cfg.pretrained_llm:
            cache = None
            use_cache = False
            logging.info("Using no-cache mode for Nemotron (full history each step)")
        else:
            cache = DynamicCache()
            use_cache = True

        gen_text = torch.empty(B, T, device=self.model.device, dtype=torch.long)
        if self.model.predict_user_text:
            gen_asr = torch.empty(B, T, device=self.model.device, dtype=torch.long)
        else:
            gen_asr = None

        if prompt_tokens is not None and prompt_token_lens is not None:
            for i, prompt_len in enumerate(prompt_token_lens):
                prompt_len = prompt_len.item()
                if prompt_len > 0:
                    gen_text[i, :prompt_len] = self.model.text_pad_id
                    if self.model.predict_user_text:
                        gen_asr[i, :prompt_len] = self.model.text_pad_id

        input_embeds[:, 0] += self.model._get_bos_embedding() * self.model.cfg.get("duplex_text_channel_weight", 1.0)
        if self.model.predict_user_text:
            input_embeds[:, 0] += self.model._get_asr_bos_embedding() * self.model.cfg.get(
                "duplex_asr_text_weight", 1.0
            )

        start_gen_pos = 0
        if prompt_token_lens is not None:
            max_prompt_len = prompt_token_lens.max().item()
            start_gen_pos = max_prompt_len

        is_prompt_position_mask = torch.zeros(B, T, dtype=torch.bool, device=self.model.device)
        if prompt_token_lens is not None:
            for i, prompt_len in enumerate(prompt_token_lens):
                prompt_len_val = prompt_len.item()
                if prompt_len_val > 0:
                    is_prompt_position_mask[i, :prompt_len_val] = True

        return {
            "input_signal": input_signal,
            "input_signal_lens": input_signal_lens,
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
        ans = self.model(
            inference_state["input_embeds"][:, :1],
            cache=inference_state["cache"],
        )

        if inference_state["start_gen_pos"] == 0:
            inference_state["gen_text"][:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
            if self.model.predict_user_text:
                inference_state["gen_asr"][:, 0] = ans["asr_logits"][:, -1].argmax(dim=-1)

        return ans, inference_state

    def _maybe_apply_forced_turn_taking(self, t, inference_state, is_prompt_position):
        """Apply forced turn-taking rules based on ASR channel tokens."""
        if not self.model.cfg.get("force_turn_taking", False):
            return

        threshold = self.model.cfg.get("force_turn_taking_threshold", 40)
        pad_window_steps = self.model.cfg.get("force_turn_taking_pad_window", 25)

        # Backward compatibility: support old checkpoints trained with '^' and '$'
        # For old models, set model.user_bos_token="^" and model.user_eos_token="$" in config
        user_bos_token = self.model.cfg.get("user_bos_token", None)
        user_eos_token = self.model.cfg.get("user_eos_token", None)

        if user_bos_token is not None:
            legacy_user_bos_id = self.model.tokenizer.text_to_ids(user_bos_token)[0]
        else:
            legacy_user_bos_id = None

        if user_eos_token is not None:
            legacy_user_eos_id = self.model.tokenizer.text_to_ids(user_eos_token)[0]
        else:
            legacy_user_eos_id = None

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
            has_pad_window = (
                (asr_recent_tokens == self.model.text_pad_id).all() if len(asr_recent_tokens) > 0 else False
            )

            # Require that the pad window starts after a non-pad token
            if has_pad_window and pad_lookback_start > 0:
                token_before_window = inference_state["gen_asr"][batch_idx, pad_lookback_start - 1]
                has_pad_window = token_before_window != self.model.text_pad_id
            elif has_pad_window and pad_lookback_start == 0:
                # If the pad window starts at position 0, it doesn't meet the requirement
                has_pad_window = False

            # Check for user EOS: either tokenizer.eos (new) or legacy user_eos (old models)
            is_user_eos = current_asr_token == self.model.tokenizer.eos
            if legacy_user_eos_id is not None:
                is_user_eos = is_user_eos or (current_asr_token == legacy_user_eos_id)

            # Check for user BOS: either text_bos_id (new) or legacy user_bos (old models)
            is_user_bos = current_asr_token == self.model.text_bos_id
            if legacy_user_bos_id is not None:
                is_user_bos = is_user_bos or (current_asr_token == legacy_user_bos_id)

            if is_user_eos or has_pad_window:
                # User has finished talking or remains silent for a while
                if not (agent_text_window == self.model.text_bos_id).any():
                    inference_state["gen_text"][batch_idx, t] = self.model.text_bos_id
            elif is_user_bos:
                # User has started talking but agent has not stopped yet
                if not (agent_text_window == self.model.text_eos_id).any():
                    inference_state["gen_text"][batch_idx, t] = self.model.text_eos_id

    def _step_inference(self, t, inference_state, ans):
        """Perform inference for one step t in the autoregressive loop."""
        last_emb = self.model.embed_tokens(inference_state["gen_text"][:, t - 1]) * self.model.cfg.get(
            "duplex_text_channel_weight", 1.0
        )
        if self.model.predict_user_text:
            last_asr_emb = self.model.embed_asr_tokens(inference_state["gen_asr"][:, t - 1]) * self.model.cfg.get(
                "duplex_asr_text_weight", 1.0
            )
            last_emb += last_asr_emb

        inference_state["input_embeds"][:, t] += last_emb

        is_prompt_position = inference_state["is_prompt_position_mask"][:, t]

        if inference_state["use_cache"]:
            ans = self.model(
                inference_state["input_embeds"][:, t : t + 1],
                cache=ans["cache"],
            )
            if not is_prompt_position.all():
                generated_tokens = ans["text_logits"][:, -1].argmax(dim=-1)
                inference_state["gen_text"][:, t] = torch.where(
                    is_prompt_position, inference_state["gen_text"][:, t], generated_tokens
                )
        else:
            ans = self.model(
                inference_state["input_embeds"][:, : t + 1],
                cache=None,
            )
            if not is_prompt_position.all():
                generated_tokens = ans["text_logits"][:, -1].argmax(dim=-1)
                inference_state["gen_text"][:, t] = torch.where(
                    is_prompt_position, inference_state["gen_text"][:, t], generated_tokens
                )

        if self.model.predict_user_text:
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

        if self.model._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            if self.model.predict_user_text:
                gen_asr = gen_asr[:, :T_local]

        if self.model.predict_user_text:
            gen_text_src = gen_asr
            src_text_cleaned = [
                self.model.tokenizer.ids_to_text(gen_text_src[b]) for b in range(gen_text_src.shape[0])
            ]
        else:
            gen_text_src = None
            src_text_cleaned = None

        if prompt_token_lens is not None:
            max_prompt_len = prompt_token_lens.max().item()
            if max_prompt_len > 0:
                current_T = gen_text.shape[1]
                gen_text_trimmed = torch.zeros(
                    B, current_T - max_prompt_len, device=self.model.device, dtype=torch.long
                )
                gen_asr_trimmed = None
                if self.model.predict_user_text:
                    gen_asr_trimmed = torch.zeros(
                        B, current_T - max_prompt_len, device=self.model.device, dtype=torch.long
                    )
                lengths_trimmed = lengths.clone()

                for i, prompt_len in enumerate(prompt_token_lens):
                    prompt_len_val = prompt_len.item()
                    actual_len = lengths[i].item() - prompt_len_val
                    if actual_len > 0:
                        gen_text_trimmed[i, :actual_len] = gen_text[i, prompt_len_val : prompt_len_val + actual_len]
                        if self.model.predict_user_text:
                            gen_asr_trimmed[i, :actual_len] = gen_asr[i, prompt_len_val : prompt_len_val + actual_len]
                    lengths_trimmed[i] = actual_len

                gen_text = gen_text_trimmed
                if self.model.predict_user_text:
                    gen_asr = gen_asr_trimmed
                    gen_text_src = gen_asr
                lengths = lengths_trimmed

        ans = {
            "text": tokens_to_str(
                gen_text,
                lengths,
                tokenizer=self.model.tokenizer,
                pad_id=self.model.text_pad_id,
                eval_text_turn_taking=self.model.cfg.get("eval_text_turn_taking", True),
            ),
            "src_text": src_text_cleaned,
            "tokens_text_src": gen_text_src,
            "tokens_text": gen_text,
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
        input_pad_len: int = 0,
        prompt_tokens: torch.Tensor = None,
        prompt_token_lens: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction (text only).
        """
        inference_state = self._init_inference(
            input_signal, input_signal_lens, input_pad_len, prompt_tokens, prompt_token_lens
        )

        ans, inference_state = self._step_zero(inference_state)

        for t in range(1, inference_state["T"]):
            ans = self._step_inference(t, inference_state, ans)

        return self._post_inference(inference_state, prompt_token_lens)
