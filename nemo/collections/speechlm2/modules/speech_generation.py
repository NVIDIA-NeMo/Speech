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
import torch
from omegaconf import DictConfig
from torch import nn

from nemo.collections.tts.modules import transformer_2501
from nemo.core.classes.module import NeuralModule


class TransformerARSpeechDecoder(NeuralModule):
    def __init__(
        self,
        speech_decoder_parms: DictConfig,
        lantent_dim: int,
        num_audio_codebooks: int,
        num_audio_tokens_per_codebook: int,
    ):
        super().__init__()
        self.use_input_cache = False
        self.speech_decoder_parms = speech_decoder_parms
        self.lantent_dim = lantent_dim
        self.num_audio_codebooks = num_audio_codebooks
        self.num_audio_tokens_per_codebook = num_audio_tokens_per_codebook
        # optional configs
        self.cfg_unconditional_prob = self.speech_decoder_parms.pop("cfg_unconditional_prob", None)
        self.cfg_scale = self.speech_decoder_parms.pop("cfg_scale", 2.5)
        self.cond_on_prev_audio_tokens = self.speech_decoder_parms.pop("cond_on_prev_audio_tokens", True)
        self.detach_input = self.speech_decoder_parms.pop("detach_input", False)

        # projection to adapt llm embeddings into the same shape of speech decoder expected input
        if lantent_dim != self.speech_decoder_parms["d_model"]:
            self.input_proj = nn.Linear(lantent_dim, self.speech_decoder_parms["d_model"])
        else:
            self.input_proj = None

        # instanciate T5-TTS decoder to full compatibility and potentialy load pretrained model
        self.t5_decoder = transformer_2501.Transformer(**self.speech_decoder_parms)

        # projection to predict audio codes
        self.final_proj = nn.Linear(
            self.speech_decoder_parms["d_model"], num_audio_codebooks * num_audio_tokens_per_codebook
        )

        # create embeddings for encode input tokens
        if self.cond_on_prev_audio_tokens:
            audio_embeddings = []
            for _ in range(self.num_audio_codebooks):
                audio_embeddings.append(
                    nn.Embedding(num_audio_tokens_per_codebook, self.speech_decoder_parms["d_model"])
                )

            self.audio_embeddings = nn.ModuleList(audio_embeddings)

    def forward(self, hidden_states, speech_mask, input_audio_tokens=None, return_raw_logits=False):
        # Megatron LLM parallel training returns T, B, F so reshape it
        # T, B, F = hidden_states.size()
        hidden_states = hidden_states.transpose(0, 1).contiguous()  # .reshape(B, T, F) # from [T, B, F] to [B, T, F]
        # input cache needed due our transformer kv cache implementation expect the whole left context
        if self.use_input_cache:
            if self.cache["hidden_states"] is None:
                self.cache["hidden_states"] = hidden_states
            else:
                self.cache["hidden_states"] = torch.cat([self.cache["hidden_states"], hidden_states], dim=1)
                hidden_states = self.cache["hidden_states"]

            if self.cache["speech_mask"] is None:
                self.cache["speech_mask"] = speech_mask
            else:
                self.cache["speech_mask"] = torch.cat([self.cache["speech_mask"], speech_mask], dim=1)
                speech_mask = self.cache["speech_mask"]

            if self.cache["input_audio_tokens"] is None:
                self.cache["input_audio_tokens"] = input_audio_tokens
            else:
                self.cache["input_audio_tokens"] = torch.cat(
                    [self.cache["input_audio_tokens"], input_audio_tokens], dim=1
                )
                input_audio_tokens = self.cache["input_audio_tokens"]

        if self.detach_input:
            hidden_states = hidden_states.detach()

        # map hidden states to the shape of the
        if self.input_proj is not None:
            speech_decoder_input = self.input_proj(hidden_states)
        else:
            speech_decoder_input = hidden_states

        # workaround for inference, because during inference speech_mask will be None
        if speech_mask is None:
            speech_mask = torch.ones(
                (speech_decoder_input.size(0), speech_decoder_input.size(1)),
                device=speech_decoder_input.device,
                dtype=torch.bool,
            )

        if self.cfg_unconditional_prob:
            if self.training:
                # if training drop the "text" conditioning in a percentage of batch
                if torch.rand(1).item() < self.cfg_unconditional_prob:
                    # make the whole batch zeros to the unconditional model
                    # ToDo: move it to cache to need to just create a 1 frame tensor in inference
                    speech_decoder_input = torch.zeros_like(speech_decoder_input)
            else:
                # if inference or evaluation create a zero tensor for speech decoder input and concatenate it to compute unconditional logits
                speech_decoder_input_zeros = torch.zeros_like(speech_decoder_input)
                speech_decoder_input = torch.cat([speech_decoder_input, speech_decoder_input_zeros], dim=0)
                # duplicate mask to match the new shape
                speech_mask = torch.cat([speech_mask, speech_mask], dim=0)
                # if cond on prev tokens enabled, so duplicate the tokens to the new shape
                if self.cond_on_prev_audio_tokens:
                    input_audio_tokens = torch.cat([input_audio_tokens, input_audio_tokens], dim=0)

        if self.cond_on_prev_audio_tokens:
            if self.detach_input:
                input_audio_tokens = input_audio_tokens.detach()

            audio_tokens_embedded = self.embed_audio_tokens(
                input_audio_tokens.transpose(1, 2).contiguous()
            )  # (B, T', E)

            speech_decoder_input = speech_decoder_input + audio_tokens_embedded

        decoder_out = self.t5_decoder(x=speech_decoder_input, x_mask=speech_mask)['output']

        # if it is true we need to return just the last autoregressive step, it is valid because for 1 frame input we produce 1 frame ouput
        if self.use_input_cache:
            decoder_out = decoder_out[:, -1:, :]

        # get the logits of all codebooks
        all_code_logits = self.final_proj(decoder_out)

        # if using cfg and it is in inference or evaluation mix unconditional and coditional logits
        if self.cfg_unconditional_prob and not self.training:
            batch_size = all_code_logits.size(0) // 2
            cond_logits = all_code_logits[:batch_size]
            uncond_logits = all_code_logits[batch_size:]
            all_code_logits = (1 - self.cfg_scale) * uncond_logits + self.cfg_scale * cond_logits

        if return_raw_logits:
            return all_code_logits

        # convert the logits from the single projection to a list with logits separated by codebook
        all_codebook_logits = self.all_logits_to_each_codebooks_logits(all_code_logits)

        return all_codebook_logits, all_code_logits

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.num_audio_codebooks):
            si = idx * self.num_audio_tokens_per_codebook
            ei = si + self.num_audio_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            codebook_probs = torch.softmax(codebook_logits / temperature, dim=-1)  # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    def all_logits_to_each_codebooks_logits(self, logits):
        all_codebook_logits = []
        for idx in range(self.num_audio_codebooks):
            si = idx * self.num_audio_tokens_per_codebook
            ei = si + self.num_audio_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]  # (B, num_tokens_per_codebook)
            # B, T, F = codebook_logits.size()
            codebook_logits = codebook_logits.transpose(
                0, 1
            ).contiguous()  # .reshape(T, B, F) # transpose for compatibility with megatron format
            all_codebook_logits.append(codebook_logits)
        return all_codebook_logits

    def embed_audio_tokens(self, audio_tokens):
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(self.num_audio_codebooks):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        return audio_embedding

    def reset_input_and_kv_cache(self, use_cache):
        if use_cache:
            print("Enabling input and KV cache!")

        self.use_input_cache = use_cache
        self.cache = self._init_cache()
        self.t5_decoder.reset_cache(use_cache=use_cache)

    @staticmethod
    def _init_cache():
        return {
            'hidden_states': None,
            'speech_mask': None,
            'input_audio_tokens': None,
        }
