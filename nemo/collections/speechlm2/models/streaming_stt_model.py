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
import math
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributed.tensor.parallel import loss_parallel
from transformers import AutoModel, GenerationConfig

from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContext
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    AUDIO_TOKEN_IDX,
    IGNORE_INDEX,
    StreamingSTTBatch,
    StreamingSTTDataset,
    build_compact_turn_markers,
    decode_with_blank,
    parse_chat_template_ids,
)
from nemo.collections.speechlm2.parts.alignments import ForcedAligner
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding, setup_perception
from nemo.collections.speechlm2.parts.utils import freeze_module, to_dataclass, unfreeze_module
from nemo.utils import logging


def token_in_vocab(token: str, tokenizer: AutoTokenizer) -> bool:
    token_pieces = tokenizer.text_to_tokens(token)
    if len(token_pieces) == 1:
        return True
    else:
        return False


def interleave_embeddings(
    input_tokens: Tensor,
    audio_mask: Tensor,
    text_embeds: Tensor,
    audio_embs: Tensor,
    pad_id: int,
) -> dict[str, Tensor]:
    """
    Merge pre-computed text and audio embeddings into a single sequence,
    guided by ``audio_mask``.

    All operations are fully batched (no Python loops over batch items):

    1. ``cumsum`` on the audio mask gives a 0-based frame index per audio position.
    2. ``torch.gather`` selects the correct audio frame for each position.
    3. ``torch.where`` picks audio or text embeddings per position.

    Args:
        input_tokens: (B, L) token IDs — only used to derive the attention mask
            (non-``pad_id`` positions).
        audio_mask: (B, L) bool — True at ``AUDIO_TOKEN_IDX`` positions.
        text_embeds: (B, L, H) embeddings produced by the text embedding layer.
            Values at audio positions are unused and may be arbitrary.
        audio_embs: (B, T_enc, H) frame-level embeddings from the audio encoder.
            If there are more audio tokens than encoder frames (last-chunk
            ceiling), the tensor is zero-padded automatically.
        pad_id: text token ID used for padding — these positions get
            ``attention_mask = False``.

    Returns:
        dict with:
            ``input_embeds`` — (B, L, H) interleaved embeddings.
            ``attention_mask`` — (B, L) bool, False only at padding positions.
    """
    B, L = input_tokens.shape

    if not audio_mask.any():
        # Pure text — nothing to interleave.
        attention_mask = input_tokens != pad_id
        return {"input_embeds": text_embeds, "attention_mask": attention_mask}

    # Sequential 0-based frame index for each audio-token position.
    frame_indices = audio_mask.long().cumsum(dim=1) - 1  # (B, L)

    # Pad encoder output if the dataset produced more audio tokens than
    # the encoder returned (last chunk ceiling).
    max_frame_idx = frame_indices.max().item()
    T_enc = audio_embs.shape[1]
    if max_frame_idx >= T_enc:
        audio_embs = F.pad(audio_embs, (0, 0, 0, max_frame_idx - T_enc + 1))

    # Gather the correct audio frame for every position in L.
    H = audio_embs.shape[2]
    gather_idx = frame_indices.clamp(min=0).unsqueeze(-1).expand(B, L, H)
    audio_at_all_pos = torch.gather(audio_embs, dim=1, index=gather_idx)  # (B, L, H)

    # Merge: audio embeddings at audio positions, text embeddings elsewhere.
    embeds = torch.where(audio_mask.unsqueeze(-1), audio_at_all_pos, text_embeds)

    # Attend to every non-padding position.
    # pad_id is ≥ 0 and AUDIO_TOKEN_IDX is −100, so this is safe.
    attention_mask = input_tokens != pad_id  # (B, L)

    return {"input_embeds": embeds, "attention_mask": attention_mask}


@dataclass
class StreamingSTTModelConfig:
    pretrained_llm: str
    pretrained_asr: str
    load_llm_weights: bool
    blank_token: str
    load_asr_weights: bool
    freeze_speech_encoder: bool
    freeze_modality_adapter: bool
    freeze_modality_proj: bool
    freeze_llm_model: bool
    freeze_llm_head: bool
    freeze_embed_tokens: bool
    chunk_size: int
    audio_tag: str = "<audio>"
    att_context_size: Optional[List[int]] = None
    audio_pad_to: Optional[int] = None
    sample_rate: int = 16000
    frame_length_in_secs: float = 0.08
    blank_loss_weight: float = 1.0
    log_every_n_steps: int = 10
    dtype: str = "bfloat16"
    # --- Compact template ---
    # Compact template: use a write token to trigger text generation, and the EOS token
    # is automatically generated by the tokenizer.
    compact_template: bool = False
    write_token: str = "<|im_start|>"
    # --- Aux chunk-boundary classifier head ---
    # Master switch. Only valid in dynamic-chunking mode (chunk_size == 0).
    # When True, a small K-layer transformer head is built on top of the LLM's
    # last hidden state and trained with BCE at audio frame positions; the LM
    # head is no longer supervised at audio positions. When False (default),
    # the boundary decision falls back to the LM head signaling via the blank
    # / user_footer_first token (legacy behavior). The module is NOT built
    # unless this flag is True.
    use_chunk_classifier: bool = False
    chunk_classifier_loss_weight: float = 0.5
    chunk_classifier_num_layers: int = 2
    chunk_classifier_init_from_llm: bool = True
    chunk_classifier_threshold: float = 0.5
    chunk_classifier_use_at_inference: bool = False
    freeze_chunk_classifier: bool = False
    # Auto-balance the BCE: pos_weight = num_neg/num_pos per batch. Most audio
    # frames are "keep listening" (label=0); the few "emit" frames (label=1) get
    # drowned out without rebalancing.
    chunk_classifier_auto_balance: bool = True


@dataclass
class StreamingState:
    """Holds the KV cache and other state for B streaming audio sessions.

    All tensors have batch dimension B (B=1 for single-stream inference).
    The LLM cache ``past_key_values`` has shape ``(layers, (B, heads, seq, dim))``
    for K and V.  The perception cache has batch dim on axis 1.
    """

    cache: tuple | None = None  # HF past_key_values with batch dim B
    generated_tokens: list[list[int]] = field(default_factory=list)  # B lists of per-chunk token IDs
    seq_lens: list[int] = field(default_factory=list)  # per-stream sequence lengths
    audio_cache: CacheAwareContext | None = None  # perception cache with batch dim B
    audio_feature_buffer: BatchedCacheFeatureBufferer | None = None
    attention_mask: Optional[Tensor] = None  # (B, seq_len) mask for left-padded prefill
    # (B, seq_len, H) running buffer of LLM last hidden states; used by the aux
    # chunk-boundary classifier when chunk_classifier_use_at_inference is True.
    # None when disabled — keeps the field cheap to always carry on the state.
    aux_hidden_buffer: Optional[Tensor] = None
    batch_size: int = 1

    @property
    def seq_len(self) -> int:
        """Max seq_len across streams (= KV cache dimension)."""
        return max(self.seq_lens) if self.seq_lens else 0


class StreamingSTTModel(LightningModule, HFHubMixin):

    def __init__(
        self,
        cfg: dict,
        forced_aligner: Optional[ForcedAligner] = None,
        data_cfg: Optional[DictConfig] = None,
        dataset_cls=StreamingSTTDataset,
    ) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to StreamingSTTModel as a Python dict to support hyperparameter "
            f"serialization in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.core_cfg: StreamingSTTModelConfig = to_dataclass(StreamingSTTModelConfig, cfg)

        # --- LLM ---
        self.tokenizer = AutoTokenizer(self.core_cfg.pretrained_llm, use_fast=True)
        self.llm = load_pretrained_hf(
            self.core_cfg.pretrained_llm,
            pretrained_weights=self.core_cfg.load_llm_weights,
        )

        # Ensure <blank> token is in the vocabulary.
        # Unescape Python escape sequences (e.g. "\\n" → "\n") because Hydra/OmegaConf
        # loads YAML strings literally without interpreting backslash escapes.
        # An empty blank_token ("") disables the blank mechanism entirely
        # (fixed chunking only — see StreamingSTTDataset for the guard).
        self.blank_token = self.core_cfg.blank_token.encode().decode('unicode_escape')

        if self.blank_token == "":
            logging.info("blank_token is empty: blank mechanism disabled")
        elif not token_in_vocab(self.blank_token, self.tokenizer):
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.blank_token]})
            self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))
            logging.info(f"Added blank token `{self.blank_token}` to tokenizer: {self.blank_token_id}")
        else:
            logging.info(f"Blank token `{str(self.blank_token)}` already in tokenizer: {self.blank_token_id}")

        # Compact-template write_token: only register if missing (default <|im_start|>
        # is already in Qwen3's vocab → uses pretrained embedding).
        if self.core_cfg.compact_template:
            wt = self.core_cfg.write_token
            if not token_in_vocab(wt, self.tokenizer):
                self.tokenizer.add_special_tokens({"additional_special_tokens": [wt]})
                self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))
                logging.info(f"compact_template: added write_token `{wt}` to tokenizer (random init)")
            else:
                logging.info(f"compact_template: using existing vocab token `{wt}` as write_token")

        # Separate embedding layer to avoid FSDP/TP conflicts (same pattern as SALM)
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

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
        # Only valid in dynamic-chunking mode (chunk_size == 0). When disabled,
        # no module is built and existing runs / checkpoints see no behavior
        # change. The boundary decision falls back to the LM head's blank /
        # user_footer_first signal.
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
        elif self.core_cfg.chunk_classifier_use_at_inference:
            raise ValueError("chunk_classifier_use_at_inference=True requires use_chunk_classifier=True")

        self._apply_freeze_config()

        # --- LoRA ---
        if "lora" in self.cfg:
            # Install LoRA after freezing the LLM body to avoid freezing the LoRA weights
            maybe_install_lora(self)
            # huggingface PEFT library freezes the whole LLM, so we need to unfreeze the lm_head if needed
            if self.core_cfg.freeze_llm_head:
                freeze_module(self.llm.lm_head)
            else:
                unfreeze_module(self.llm.lm_head)

        if forced_aligner is not None:
            assert data_cfg is not None, "Dataset config is required for online forced alignment"
            assert dataset_cls is not None, "Dataset class is required for online forced alignment"
            self.forced_aligner = forced_aligner
            self.dataset = dataset_cls(cfg=data_cfg, tokenizer=self.tokenizer)
        else:
            self.forced_aligner = None
            self.dataset = None

        logging.info("\n" + str(ModelSummary(self, max_depth=2)))

    def _build_chunk_classifier(self) -> None:
        """Construct the aux backbone + linear head.

        The backbone reuses the LLM's architecture via ``AutoModel.from_config``
        (works for any modern HF decoder-only LLM — Llama/Qwen/Mistral/Phi/Gemma).
        ``embed_tokens`` is dropped since we always feed ``inputs_embeds``;
        same pattern as the main LLM at __init__.
        """
        K = max(int(self.core_cfg.chunk_classifier_num_layers), 1)
        aux_cfg = deepcopy(self.llm.config)
        aux_cfg.num_hidden_layers = K
        # Aux backbone is run as a full-sequence forward at both train and
        # inference time — no KV cache needed.
        aux_cfg.use_cache = False

        self.chunk_classifier_backbone = AutoModel.from_config(aux_cfg)
        # Drop V×H embedding table: we always feed inputs_embeds.
        # (Same trick as line 228-229 for the main LLM.)
        if hasattr(self.chunk_classifier_backbone, "embed_tokens"):
            del self.chunk_classifier_backbone.embed_tokens

        self.chunk_classifier_head = nn.Linear(aux_cfg.hidden_size, 1)
        nn.init.zeros_(self.chunk_classifier_head.bias)

        # Optional warm-start: copy the last K layers + final norm from the
        # main LLM. The aux backbone consumes the LLM's last hidden state, so
        # these layers operate on the right input distribution and converge
        # much faster than random init.
        if self.core_cfg.chunk_classifier_init_from_llm:
            try:
                src_layers = self.llm.model.layers[-K:]
                for i in range(K):
                    self.chunk_classifier_backbone.layers[i].load_state_dict(src_layers[i].state_dict())
                self.chunk_classifier_backbone.norm.load_state_dict(self.llm.model.norm.state_dict())
                logging.info(f"chunk_classifier: warm-started from last {K} LLM layers")
            except (AttributeError, KeyError) as e:
                logging.warning(
                    f"chunk_classifier_init_from_llm: warm-start failed ({e}); " "falling back to random init"
                )

    def _apply_freeze_config(self) -> None:
        if self.core_cfg.freeze_speech_encoder:
            freeze_module(self.perception.encoder)
        else:
            unfreeze_module(self.perception.encoder)

        if self.core_cfg.freeze_modality_adapter:
            freeze_module(self.perception.modality_adapter)
        else:
            unfreeze_module(self.perception.modality_adapter)

        if self.core_cfg.freeze_modality_proj:
            freeze_module(self.perception.proj)
        else:
            unfreeze_module(self.perception.proj)

        # Freeze the LLM body (lm_head and embed_tokens are handled separately)
        if self.core_cfg.freeze_llm_model:
            freeze_module(self.llm.model)
        else:
            unfreeze_module(self.llm.model)

        # lm_head is inside self.llm, so re-apply after the LLM-wide freeze
        if self.core_cfg.freeze_llm_head:
            freeze_module(self.llm.lm_head)
        else:
            unfreeze_module(self.llm.lm_head)

        # embed_tokens is a separate top-level module (moved out of llm)
        if self.core_cfg.freeze_embed_tokens:
            freeze_module(self.embed_tokens)
        else:
            unfreeze_module(self.embed_tokens)

        # Aux chunk-boundary classifier (backbone + linear head). Only present
        # when use_chunk_classifier is True.
        if self.core_cfg.use_chunk_classifier:
            if self.core_cfg.freeze_chunk_classifier:
                freeze_module(self.chunk_classifier_backbone)
                freeze_module(self.chunk_classifier_head)
            else:
                unfreeze_module(self.chunk_classifier_backbone)
                unfreeze_module(self.chunk_classifier_head)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return int(self.embed_tokens.num_embeddings)

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad_id
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "The text tokenizer has no <pad> or <unk> token; using id 0 for "
                "padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    @property
    def sample_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    @property
    def frame_duration(self) -> float:
        """Duration (in seconds) of one audio frame at the perception output."""
        return self.perception.token_equivalent_duration

    @property
    def blank_token_id(self) -> int:
        # Sentinel -1 when blank is disabled — guarantees `token == blank_token_id`
        # never matches a real vocab id, so stop-on-blank checks naturally no-op.
        if self.blank_token == "":
            return -1
        return self.tokenizer.text_to_ids(self.blank_token)[0]

    @property
    def has_blank(self) -> bool:
        return self.blank_token != ""

    # ------------------------------------------------------------------
    # Core: efficient audio-text embedding interleaving
    # ------------------------------------------------------------------

    def _build_input_embeds(
        self,
        input_tokens: Tensor,
        audios: Tensor,
        audio_lens: Tensor,
    ) -> dict[str, Tensor]:
        """
        Encode audio, embed text tokens, then interleave them.

        This is the high-level entry point used by ``training_step`` and
        ``_eval_step``.  The pure-tensor interleaving logic lives in
        :func:`interleave_embeddings` so it can be tested without a model.

        Args:
            input_tokens: (B, L) token IDs with ``AUDIO_TOKEN_IDX`` at audio
                positions and ``text_pad_id`` at left-padding positions.
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
        Returns:
            dict with keys ``input_embeds`` (B, L, H), ``attention_mask`` (B, L).
        """
        audio_mask = input_tokens == AUDIO_TOKEN_IDX  # (B, L)

        # --- text embeddings ---
        # Zero-out audio positions so embed_tokens gets valid indices.
        text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
        text_embeds = self.embed_tokens(text_tokens)  # (B, L, H)

        # --- audio embeddings ---
        audio_embs, _audio_emb_lens = self.perception(
            input_signal=audios,
            input_signal_length=audio_lens,
        )  # audio_embs: (B, T_enc, H)

        # --- interleave & build attention mask ---
        return interleave_embeddings(
            input_tokens=input_tokens,
            audio_mask=audio_mask,
            text_embeds=text_embeds,
            audio_embs=audio_embs,
            pad_id=self.text_pad_id,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor | None = None,
        cache=None,
        output_hidden_states: bool = False,
    ) -> dict[str, Tensor]:
        """
        Forward pass:  embeddings → LLM → logits.

        When ``output_hidden_states=True`` the dict also contains
        ``hidden_states`` (B, L, H) — the LLM's last-layer hidden state, used
        as input to the aux chunk-boundary classifier.
        """
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        ans = {"logits": out["logits"]}  # (B, L, V)
        if output_hidden_states:
            ans["hidden_states"] = out["hidden_states"][-1]  # (B, L, H)
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: StreamingSTTBatch, batch_idx: int):
        # Keep frozen modules in eval mode (disables dropout / batch-norm updates).
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        if self.forced_aligner is not None:
            alignments = self.forced_aligner.align(batch.audios, batch.audio_lens, batch.text)
            batch = self.dataset.get_batch_data(
                cuts=batch.cuts,
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                alignments=alignments,
                text=batch.text,
            )
            batch = move_data_to_device(batch, self.device)

        inputs = self._build_input_embeds(batch.input_tokens, batch.audios, batch.audio_lens)
        use_aux = self.core_cfg.use_chunk_classifier
        outputs = self.forward(
            inputs["input_embeds"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=use_aux,
        )

        target_ids = batch.target_tokens

        # When the aux chunk classifier is active, strip audio-frame positions
        # from the LM CE so the LM head is only supervised on text. The aux
        # head (below) handles the boundary decision via BCE. Use the input-axis
        # audio mask — NOT a target-value mask — so end-of-chunk blanks at
        # text positions (line 1696/1701 in inference) remain supervised.
        if use_aux:
            audio_mask = batch.input_tokens == AUDIO_TOKEN_IDX  # (B, L)
            target_ids = torch.where(audio_mask, torch.full_like(target_ids, IGNORE_INDEX), target_ids)
        else:
            audio_mask = None

        num_targets = (target_ids != IGNORE_INDEX).long().sum()

        if num_targets == 0:
            logging.warning("Batch %d: num_targets is 0 — skipping (returning zero loss).", batch_idx)
            return {"loss": torch.tensor(0.0, device=target_ids.device, requires_grad=True)}

        logits = outputs["logits"]

        # # Diagnose NaN sources (remove once stable).
        # if torch.isnan(inputs["input_embeds"]).any():
        #     logging.warning("Batch %d: NaN in input_embeds", batch_idx)
        # if torch.isnan(logits).any():
        #     logging.warning("Batch %d: NaN in logits", batch_idx)

        flat_logits = logits.flatten(0, 1)
        flat_targets = target_ids.flatten(0, 1)

        with loss_parallel():
            per_token_loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )

        # --- Blank vs non-blank loss breakdown ---
        blank_id = self.blank_token_id
        valid_mask = flat_targets != IGNORE_INDEX
        # When blank is disabled (blank_id=-1), is_blank is always False →
        # everything counts as non-blank, and blank_weight has no effect.
        is_blank = valid_mask & (flat_targets == blank_id)
        is_nonblank = valid_mask & (flat_targets != blank_id)
        num_blank = is_blank.sum()
        num_nonblank = is_nonblank.sum()

        # Apply blank loss weight (< 1.0 to down-weight easy blank predictions)
        blank_weight = self.core_cfg.blank_loss_weight
        if blank_weight != 1.0 and self.has_blank:
            effective_num_targets = num_blank * blank_weight + num_nonblank
            loss = (
                per_token_loss[is_nonblank].sum() + per_token_loss[is_blank].sum() * blank_weight
            ) / effective_num_targets
        else:
            loss = per_token_loss.sum() / num_targets

        with torch.no_grad():
            loss_blank = per_token_loss[is_blank].sum() / num_blank.clamp(min=1)
            loss_nonblank = per_token_loss[is_nonblank].sum() / num_nonblank.clamp(min=1)

        # --- Aux chunk-boundary classifier loss ---
        # BCE on the aux head's binary "ready to emit" prediction at audio frames.
        # Supervised positions: input is an audio frame AND original target was a
        # decision token (blank=keep listening, user_footer_first=emit). Using the
        # input-axis audio mask here exactly mirrors the supervision the LM head
        # used to provide at audio positions before §4(a) masked them out.
        cls_loss_log = torch.zeros((), device=loss.device)
        if use_aux and self.has_blank and audio_mask is not None and self._user_footer_first_id is not None:
            audio_mask_flat = audio_mask.flatten(0, 1)  # (B*L,)
            orig_targets_flat = batch.target_tokens.flatten(0, 1)  # pre-LM-CE-masking
            decision_mask = audio_mask_flat & (orig_targets_flat != IGNORE_INDEX)
            num_decisions = decision_mask.sum()
            if num_decisions > 0:
                aux_out = self.chunk_classifier_backbone(
                    inputs_embeds=outputs["hidden_states"],  # (B, L, H)
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                flat_aux = aux_out.last_hidden_state.flatten(0, 1)  # (B*L, H)
                cls_logits = self.chunk_classifier_head(flat_aux[decision_mask]).squeeze(-1)
                cls_targets = (orig_targets_flat[decision_mask] == self._user_footer_first_id).to(cls_logits.dtype)
                # Auto-balance: pos_weight = N_neg/N_pos. Skip when either class
                # is empty in this batch (pos_weight would zero out one side).
                num_pos = cls_targets.sum()
                num_neg = cls_targets.numel() - num_pos
                if self.core_cfg.chunk_classifier_auto_balance and num_pos > 0 and num_neg > 0:
                    pos_weight = (num_neg.float() / num_pos.float()).detach()
                else:
                    pos_weight = None
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, pos_weight=pos_weight)
                cls_w = self.core_cfg.chunk_classifier_loss_weight
                loss = loss + cls_w * cls_loss
                cls_loss_log = cls_loss.detach()
                cls_pos_ratio = num_pos.float() / num_decisions.clamp(min=1).float()
                self.log_dict(
                    {"loss_chunk_cls_pos_ratio": cls_pos_ratio},
                    on_step=True,
                )

        B, L = inputs["input_embeds"].shape[:2]
        self.log_dict(
            {
                "loss": loss,
                "loss_blank": loss_blank,
                "loss_nonblank": loss_nonblank,
                "loss_chunk_cls": cls_loss_log,
                "blank_ratio": num_blank.float() / num_targets,
                "learning_rate": torch.as_tensor(
                    self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0
                ),
                "batch_size": float(B),
                "sequence_length": float(L),
                "num_targets": num_targets.float(),
                "target_to_input_ratio": num_targets / (B * L),
            },
            on_step=True,
        )
        return {"loss": loss}

    def configure_optimizers(self):
        return configure_optimizers(self)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses: dict[str, list] = defaultdict(list)
        self._partial_accuracies: dict[str, list] = defaultdict(list)
        # Per-class TP/total counts for the aux chunk classifier. Aggregated
        # across the epoch so macro acc isn't biased by per-batch composition.
        self._partial_aux_pos_correct: dict[str, list] = defaultdict(list)
        self._partial_aux_pos_total: dict[str, list] = defaultdict(list)
        self._partial_aux_neg_correct: dict[str, list] = defaultdict(list)
        self._partial_aux_neg_total: dict[str, list] = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        if val_losses:
            self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        if accuracies:
            self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        # --- Aux chunk classifier: macro accuracy ---
        # Sum per-class counts across the epoch and compute pos/neg accuracy
        # once at the end. Macro acc = (pos_acc + neg_acc) / 2 — class-balanced.
        macro_accs = []
        for name in self._partial_aux_pos_total.keys():
            pos_correct = torch.stack(self._partial_aux_pos_correct[name]).sum()
            pos_total = torch.stack(self._partial_aux_pos_total[name]).sum()
            neg_correct = torch.stack(self._partial_aux_neg_correct[name]).sum()
            neg_total = torch.stack(self._partial_aux_neg_total[name]).sum()
            pos_acc = pos_correct.float() / pos_total.clamp(min=1).float()
            neg_acc = neg_correct.float() / neg_total.clamp(min=1).float()
            macro = (pos_acc + neg_acc) / 2
            self.log(f"val_aux_pos_acc_{name}", pos_acc, on_epoch=True, sync_dist=True)
            self.log(f"val_aux_neg_acc_{name}", neg_acc, on_epoch=True, sync_dist=True)
            self.log(f"val_aux_macro_acc_{name}", macro, on_epoch=True, sync_dist=True)
            macro_accs.append(macro)
        if macro_accs:
            self.log("val_aux_macro_acc", torch.stack(macro_accs).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_losses.clear()
        self._partial_accuracies.clear()
        self._partial_aux_pos_correct.clear()
        self._partial_aux_pos_total.clear()
        self._partial_aux_neg_correct.clear()
        self._partial_aux_neg_total.clear()

    def validation_step(self, batch, batch_idx: int):
        # Support multiple validation dataloaders ({name: batch} dict).
        if isinstance(batch, dict):
            for name, dataset_batch in batch.items():
                if dataset_batch is not None:
                    self._eval_step(dataset_batch, name, batch_idx)
        else:
            self._eval_step(batch, "val", batch_idx)

    def _eval_step(self, batch: StreamingSTTBatch, name: str, batch_idx: int = 0) -> None:
        if self.forced_aligner is not None:
            alignments = self.forced_aligner.align(batch.audios, batch.audio_lens, batch.text)
            batch = self.dataset.get_batch_data(
                cuts=batch.cuts,
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                alignments=alignments,
                text=batch.text,
            )
            batch = move_data_to_device(batch, self.device)

        inputs = self._build_input_embeds(batch.input_tokens, batch.audios, batch.audio_lens)
        aux_active = self.core_cfg.use_chunk_classifier and self.has_blank and self._user_footer_first_id is not None
        outputs = self.forward(
            inputs["input_embeds"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=aux_active,
        )

        target_ids = batch.target_tokens
        num_targets = (target_ids != IGNORE_INDEX).long().sum()

        with loss_parallel():
            loss = (
                F.cross_entropy(
                    outputs["logits"].flatten(0, 1),
                    target_ids.flatten(0, 1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
                / num_targets
            )

        preds = outputs["logits"].argmax(dim=-1).view(-1)
        refs = target_ids.reshape(-1)
        preds = preds[refs != IGNORE_INDEX]
        refs = refs[refs != IGNORE_INDEX]
        accuracy = preds.eq(refs).float().mean()

        self._partial_val_losses[name].append(loss)
        self._partial_accuracies[name].append(accuracy)

        # --- Aux chunk classifier: per-class correct/total counts ---
        if aux_active:
            audio_mask = batch.input_tokens == AUDIO_TOKEN_IDX  # (B, L)
            audio_mask_flat = audio_mask.flatten(0, 1)
            orig_targets_flat = batch.target_tokens.flatten(0, 1)
            decision_mask = audio_mask_flat & (orig_targets_flat != IGNORE_INDEX)
            if decision_mask.any():
                aux_out = self.chunk_classifier_backbone(
                    inputs_embeds=outputs["hidden_states"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                flat_aux = aux_out.last_hidden_state.flatten(0, 1)
                cls_logits = self.chunk_classifier_head(flat_aux[decision_mask]).squeeze(-1)
                cls_targets = orig_targets_flat[decision_mask] == self._user_footer_first_id
                # Use the configured threshold so val acc reflects what
                # _generate_dynamic_streaming will actually do at inference.
                thr = self.core_cfg.chunk_classifier_threshold
                cls_preds = torch.sigmoid(cls_logits) >= thr
                correct = cls_preds == cls_targets
                pos_mask = cls_targets
                neg_mask = ~cls_targets
                self._partial_aux_pos_correct[name].append((correct & pos_mask).sum().detach())
                self._partial_aux_pos_total[name].append(pos_mask.sum().detach())
                self._partial_aux_neg_correct[name].append((correct & neg_mask).sum().detach())
                self._partial_aux_neg_total[name].append(neg_mask.sum().detach())

        # Log decoded predictions vs references periodically (first sample in batch).
        if batch_idx % self.core_cfg.log_every_n_steps == 0:
            # Per-sample: decode only the first sample's non-IGNORE tokens.
            sample_target = batch.target_tokens[0]
            sample_logits = outputs["logits"][0]
            sample_preds = sample_logits.argmax(dim=-1)
            mask = sample_target != IGNORE_INDEX
            sample_ref_ids = sample_target[mask].tolist()
            sample_pred_ids = sample_preds[mask].tolist()

            ref_decoded = decode_with_blank(sample_ref_ids, self.blank_token, self.tokenizer)
            pred_decoded = decode_with_blank(sample_pred_ids, self.blank_token, self.tokenizer)
            ref_text = batch.text[0] if batch.text else ""
            logging.info(
                "[%s] batch %d\n  gt:         `%s`\n  ref_tokens: `%s`\n  pred:       `%s`",
                name,
                batch_idx,
                ref_text,
                ref_decoded,
                pred_decoded,
            )

    # ------------------------------------------------------------------
    # Test (delegates to validation logic)
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    # ------------------------------------------------------------------
    # Backward + OOMptimizer
    # ------------------------------------------------------------------

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    @property
    def oomptimizer_schema(self) -> dict:
        from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

        return {
            "cls": StreamingSTTBatch,
            "inputs": [
                {
                    "name": "input_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": int(self.text_vocab_size),
                },
                {"name": "input_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": int(self.text_vocab_size),
                },
                {"name": "target_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
            ],
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensure_inference_cache(self) -> None:
        """Lazily cache token templates and IDs needed for inference.

        Uses ``apply_chat_template(tokenize=False)`` on a 4-message dummy
        conversation and splits the text around a sentinel to isolate
        user-header, user-footer + assistant-header, and assistant-footer tokens.

        The 4-message pattern (two user+assistant pairs) ensures the *first*
        assistant turn is not the last — this prevents Qwen3-style chat
        templates from injecting ``<think>``/``</think>`` tags, which only
        appear on the final assistant turn.
        """
        if hasattr(self, '_inference_cache_ready'):
            return

        hf_tok = self.tokenizer.tokenizer
        chunk_size = self.core_cfg.chunk_size

        # --- Build turn template ---
        if self.core_cfg.compact_template:
            user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids = build_compact_turn_markers(
                hf_tok, self.core_cfg.write_token
            )
            logging.info(
                f"compact_template: user_header={user_header_ids}, "
                f"write+mid={user_footer_and_asst_header_ids}, footer={asst_footer_ids}"
            )
        else:
            user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids = parse_chat_template_ids(
                hf_tok, last_turn=(chunk_size < 0)
            )
        self._user_header_ids = user_header_ids
        self._user_footer_and_asst_header_ids = user_footer_and_asst_header_ids
        self._asst_footer_ids = asst_footer_ids

        # Always cache user_footer_first_id — needed by state machine inference
        # for both dynamic (chunk_size=0) and fixed chunking (use_state_machine_inference).
        self._user_footer_first_id = user_footer_and_asst_header_ids[0] if user_footer_and_asst_header_ids else None

        if chunk_size > 0:
            turn_ids = user_header_ids + [AUDIO_TOKEN_IDX] * chunk_size + user_footer_and_asst_header_ids
            self._turn_template_ids = turn_ids
            n_audio = turn_ids.count(AUDIO_TOKEN_IDX)
            logging.info(
                f"Streaming turn template ({len(turn_ids)} tokens, "
                f"{n_audio} audio slots, chunk_size={chunk_size}): {turn_ids}"
            )
        elif chunk_size == 0:
            # Dynamic chunking: no fixed turn template. Audio frames are fed
            # incrementally; the user header/footer are appended on demand.
            self._turn_template_ids = None
            logging.info(
                f"Dynamic chunking mode: user_footer_first_id={self._user_footer_first_id}, "
                f"user_header_ids={user_header_ids}"
            )
        else:
            self._turn_template_ids = None
            logging.info(f"Offline mode (chunk_size={chunk_size}): no fixed turn template")

        self._eos_id = getattr(hf_tok, 'eos_token_id', None)

        # When eos_token_id coincides with a token in the footer (e.g. Qwen3
        # where eos = <|im_end|> = footer[0]), detecting EOS acts as an
        # early-stop shortcut that avoids generating the remaining footer
        # tokens.  When eos_token_id is NOT in the footer it serves as a
        # safety-net stop only.
        self._eos_in_footer = self._eos_id is not None and self._eos_id in self._asst_footer_ids
        logging.info(
            f"Assistant footer IDs: {self._asst_footer_ids}, "
            f"blank ID: {self.blank_token_id}, EOS ID: {self._eos_id}, "
            f"EOS in footer: {self._eos_in_footer}"
        )
        self._inference_cache_ready = True

    def _sample_token(
        self,
        logits: Tensor,
        generated_ids: list[list[int]] | list[int] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> Tensor:
        """Select the next token from logits.

        Applies the following transforms in order (each is skipped when the
        corresponding parameter is at its default/off value):

        1. **Suppress tokens** — force listed token IDs to ``-inf``.
        2. **No-repeat-ngram** — block n-grams that already appear in
           *generated_ids*.
        3. **Repetition penalty** — scale logits for tokens that already appear
           in *generated_ids*.
        4. **Temperature** — divide logits by temperature.
        5. **Top-k** — keep only the *k* highest-scoring tokens.
        6. **Top-p (nucleus)** — keep the smallest set of tokens whose
           cumulative probability is ≥ *top_p*.
        7. If ``do_sample`` is ``True``, sample from the filtered distribution;
           otherwise return the argmax.

        Parameters are read from *generation_kwargs* first, falling back to
        *generation_config*, then to HuggingFace defaults.

        Args:
            logits: ``(B, vocab_size)`` logits for the last position.
            generated_ids: Token IDs generated so far.  For B=1, a flat list.
                For B>1, a list of B lists (one per stream).  Used for
                repetition-aware transforms.  May be ``None`` or empty.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(B,)`` tensor with the selected token IDs.
        """
        # Fast path: no config → greedy
        if generation_config is None and not generation_kwargs:
            return logits.argmax(dim=-1)

        cfg = generation_config or GenerationConfig()
        do_sample = generation_kwargs.get('do_sample', cfg.do_sample)
        temperature = generation_kwargs.get('temperature', cfg.temperature)
        top_k = generation_kwargs.get('top_k', cfg.top_k)
        top_p = generation_kwargs.get('top_p', cfg.top_p)
        repetition_penalty = generation_kwargs.get('repetition_penalty', cfg.repetition_penalty)
        no_repeat_ngram_size = generation_kwargs.get('no_repeat_ngram_size', cfg.no_repeat_ngram_size)
        suppress_tokens = generation_kwargs.get('suppress_tokens', cfg.suppress_tokens)

        # --- logit manipulation (order matters) ---

        # 1. Suppress tokens
        if suppress_tokens:
            logits[..., suppress_tokens] = float('-inf')

        # 2. No-repeat-ngram blocking
        if no_repeat_ngram_size > 0 and generated_ids and len(generated_ids) >= no_repeat_ngram_size - 1:
            ngram_prefix = generated_ids[-(no_repeat_ngram_size - 1) :]
            for i in range(len(generated_ids) - no_repeat_ngram_size + 1):
                if generated_ids[i : i + no_repeat_ngram_size - 1] == ngram_prefix:
                    # The token that followed this prefix last time is banned
                    logits[..., generated_ids[i + no_repeat_ngram_size - 1]] = float('-inf')

        # 3. Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            prev_token_ids = torch.tensor(list(set(generated_ids)), device=logits.device)
            scores = logits[..., prev_token_ids]
            # Penalize: divide positive scores, multiply negative scores
            logits[..., prev_token_ids] = torch.where(
                scores > 0, scores / repetition_penalty, scores * repetition_penalty
            )

        # Greedy fast path (no sampling-related transforms needed)
        if not do_sample:
            return logits.argmax(dim=-1)

        # 4. Temperature scaling
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # 5. Top-k filtering
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, k, dim=-1)[0][..., -1:]
            logits = logits.masked_fill(logits < kth_val, float('-inf'))

        # 6. Top-p (nucleus) filtering
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Mark tokens whose cumulative probability (excluding themselves) >= top_p
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            indices_to_remove = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 7. Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _autoregressive_decode(
        self,
        logits: Tensor,
        cache: tuple,
        state: Optional['StreamingState'],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        stop_on_blank: Union[bool, str] = True,
        **generation_kwargs,
    ) -> tuple[list[list[int]], tuple, list[bool], int]:
        """Autoregressive decoding (supports B streams).

        Token selection is delegated to :meth:`_sample_token`, which supports
        greedy (default), sampling (temperature / top-k / top-p), repetition
        penalty, no-repeat-ngram blocking, and token suppression.

        Generation stops per stream when any of these conditions is met:

        1. **EOS** — the tokenizer's ``eos_token_id`` is predicted.
        2. **Blank** — the ``<blank>`` token is predicted (controlled by
           ``stop_on_blank``).
        3. **Footer sequence** — the last *N* tokens match ``self._asst_footer_ids``.
        4. **Max tokens** — ``max_new_tokens`` is reached.

        Args:
            logits: ``(B, L, V)`` logits from the LLM forward pass.
            cache: HF ``past_key_values`` with batch dim B.
            max_new_tokens: Maximum tokens to generate per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            stop_on_blank: Controls blank-token stopping behavior:
                - ``True`` (default): stop whenever blank is predicted.
                  Use with dedicated ``<blank>`` special tokens.
                - ``"first"``: stop only if blank is the **first** token
                  generated (= "no speech this chunk").  Use when the blank
                  token is a natural text token (e.g. ``" "``).
                - ``False``: never stop on blank.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(generated_per_stream, updated_cache, footer_consumed_per_stream, num_feed_steps)``
            where ``generated_per_stream`` is a list of B token-ID lists and
            ``num_feed_steps`` is how many tokens were fed to the LLM cache.
        """
        B = logits.shape[0]
        footer = self._asst_footer_ids
        flen = len(footer)
        generated: list[list[int]] = [[] for _ in range(B)]
        footer_consumed = [False] * B
        finished = [False] * B
        num_feed_steps = 0
        # Track which streams need their token fed to the LLM this step.
        # EOS tokens must NOT be fed; blank/footer/normal tokens must be fed.
        feed_mask = [False] * B

        next_tokens = self._sample_token(logits[:, -1, :], None, generation_config, **generation_kwargs)  # (B,)

        for _ in range(max_new_tokens):
            for b in range(B):
                feed_mask[b] = False
                if finished[b]:
                    continue
                tid = next_tokens[b].item()

                # EOS: stop WITHOUT feeding to LLM. Append a chunk separator so
                # decode_with_blank can join per-chunk outputs correctly.
                if self._eos_id is not None and tid == self._eos_id:
                    finished[b] = True
                    # Blank token when enabled, else EOS id itself (matches decode_with_blank).
                    generated[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                    continue

                # All other tokens get appended and fed to LLM
                generated[b].append(tid)
                feed_mask[b] = True

                # Blank: stop (token IS fed to LLM, IS in generated).
                # When stop_on_blank == "first", only stop if blank is the
                # first generated token (= "no speech this chunk").  This
                # avoids false stops when the blank token collides with a
                # natural text token (e.g. " ") that appears mid-sentence.
                if tid == self.blank_token_id:
                    if stop_on_blank is True or (stop_on_blank == "first" and len(generated[b]) == 1):
                        finished[b] = True

                # Footer sequence match
                elif flen > 0 and len(generated[b]) >= flen and generated[b][-flen:] == footer:
                    generated[b] = generated[b][:-flen]
                    footer_consumed[b] = True
                    finished[b] = True

            # If no stream needs feeding, we're done
            if not any(feed_mask):
                break

            # Feed tokens to LLM. For finished streams, feed the blank token
            # (which the model was trained on) instead of a pad token, so the
            # KV cache stays clean — no foreign tokens that corrupt attention.
            # When blank is disabled, feed text_pad_id as a fallback.
            filler_id = self.blank_token_id if self.has_blank else self.text_pad_id
            tokens_to_feed = next_tokens.clone()
            for b in range(B):
                if not feed_mask[b]:
                    tokens_to_feed[b] = filler_id

            # All tokens are "real" (blank is a valid token), so all seq_lens grow
            if state is not None:
                for b in range(B):
                    state.seq_lens[b] += 1

            token_emb = self.embed_tokens(tokens_to_feed.unsqueeze(1))  # (B, 1, H)

            state.attention_mask = torch.cat(
                [
                    state.attention_mask,
                    torch.ones(B, 1, dtype=state.attention_mask.dtype, device=state.attention_mask.device),
                ],
                dim=1,
            )
            out = self.llm(
                inputs_embeds=token_emb,
                past_key_values=cache,
                attention_mask=state.attention_mask,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            num_feed_steps += 1

            if all(finished):
                break

            next_tokens = self._sample_token(out.logits[:, -1, :], None, generation_config, **generation_kwargs)

        return generated, cache, footer_consumed, num_feed_steps

    def get_audio_feature_buffer(
        self,
        batch_size: int,
        chunk_size_override: Optional[int] = None,
    ) -> BatchedCacheFeatureBufferer:
        """Get the audio feature buffer for the streaming state.

        Args:
            batch_size: Number of parallel streams.
            chunk_size_override: If provided, use this chunk size (in frames)
                instead of ``self.core_cfg.chunk_size``.  Used by dynamic
                chunking inference where the inference step size differs
                from the config chunk_size.
        """
        preprocessor_cfg: DictConfig = self.perception.cfg.preprocessor
        window_stride_in_secs = preprocessor_cfg.window_stride
        pre_encode_cache_size = self.perception.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(pre_encode_cache_size, list):
            pre_encode_cache_size = pre_encode_cache_size[1]
        pre_encode_cache_size_in_secs = pre_encode_cache_size * window_stride_in_secs
        cs = chunk_size_override if chunk_size_override is not None else max(self.core_cfg.chunk_size, 1)
        chunk_size_in_secs = cs * self.core_cfg.frame_length_in_secs
        buffer_size_in_secs = pre_encode_cache_size_in_secs + chunk_size_in_secs

        audio_feature_buffer = BatchedCacheFeatureBufferer(
            num_slots=batch_size,
            sample_rate=self.core_cfg.sample_rate,
            buffer_size_in_secs=buffer_size_in_secs,
            chunk_size_in_secs=buffer_size_in_secs,  # recalculate mel-spec for the whole buffer
            preprocessor_cfg=preprocessor_cfg,
            device=self.device,
        )
        return audio_feature_buffer

    def get_init_streaming_state(
        self,
        system_prompt: Union[str, List[str]],
        device: torch.device,
        batch_size: int = 1,
    ) -> StreamingState:
        """Forward the system prompt through the LLM and return a fresh :class:`StreamingState`.

        Args:
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            device: Target device.
            batch_size: Number of parallel streams (B).
        """
        hf_tok = self.tokenizer.tokenizer
        dtype = self.embed_tokens.weight.dtype

        if isinstance(system_prompt, str):
            prompts = [system_prompt] * batch_size
        else:
            prompts = system_prompt

        # Tokenize each prompt
        all_sys_ids = []
        for prompt in prompts:
            ids = hf_tok.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            all_sys_ids.append(ids)

        # Check if all prompts are the same length (common case: same prompt)
        sys_lens = [len(ids) for ids in all_sys_ids]
        needs_padding = len(set(sys_lens)) > 1

        # Capture hidden states from the prefill if the aux head will be used at
        # inference, so the aux backbone sees the same full-sequence context as
        # at training time.
        capture_hidden = self.core_cfg.use_chunk_classifier and self.core_cfg.chunk_classifier_use_at_inference

        if not needs_padding:
            # Fast path: all same length, no padding needed
            sys_embs = self.embed_tokens(
                torch.tensor(all_sys_ids[0], device=device, dtype=torch.long).unsqueeze(0)
            ).expand(batch_size, -1, -1)
            attention_mask = torch.ones(batch_size, sys_lens[0], dtype=torch.long, device=device)
            out = self.llm(
                inputs_embeds=sys_embs,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=capture_hidden,
                return_dict=True,
            )
            max_sys_len = sys_lens[0]
        else:
            # Per-sample prompts with different lengths: left-pad and use attention mask
            max_sys_len = max(sys_lens)
            H = self.embed_tokens.weight.shape[-1]
            sys_embs = torch.zeros(batch_size, max_sys_len, H, device=device, dtype=dtype)
            attention_mask = torch.zeros(batch_size, max_sys_len, dtype=torch.long, device=device)
            for b in range(batch_size):
                embs = self.embed_tokens(
                    torch.tensor(all_sys_ids[b], device=device, dtype=torch.long).unsqueeze(0)
                ).squeeze(
                    0
                )  # (L_b, H)
                offset = max_sys_len - sys_lens[b]
                sys_embs[b, offset:] = embs
                attention_mask[b, offset:] = 1
            out = self.llm(
                inputs_embeds=sys_embs,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=capture_hidden,
                return_dict=True,
            )

        aux_hidden_buffer = out.hidden_states[-1] if capture_hidden else None

        cache_last_channel, cache_last_time, cache_last_channel_len = self.perception.get_initial_cache_state(
            batch_size=batch_size, dtype=dtype, device=device
        )
        audio_feature_buffer = self.get_audio_feature_buffer(batch_size=batch_size)
        audio_cache = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        return StreamingState(
            cache=out.past_key_values,
            generated_tokens=[[] for _ in range(batch_size)],
            seq_lens=[max_sys_len] * batch_size,
            audio_cache=audio_cache,
            audio_feature_buffer=audio_feature_buffer,
            attention_mask=attention_mask,
            aux_hidden_buffer=aux_hidden_buffer,
            batch_size=batch_size,
        )

    @torch.no_grad()
    def _chunked_streaming_step(
        self,
        audio_chunks: Tensor,
        audio_chunk_lens: Optional[Tensor] = None,
        state: Optional[StreamingState] = None,
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        _audio_embs: Optional[Tensor] = None,
        **generation_kwargs,
    ) -> list[list[int]]:
        """
        Process B raw audio chunks and generate the assistant responses.

        Args:
            audio_chunks: ``(B, T_samples)`` raw waveforms for one chunk per stream.
            audio_chunk_lens: ``(B,)`` number of valid samples per stream.
            state: Mutable :class:`StreamingState` with ``batch_size=B`` (updated in place).
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            _audio_embs: Optional pre-computed audio embeddings ``(B, chunk_size, H)``.
                Diagnostic use only.
            generation_kwargs: Per-call overrides for generation parameters.
        Returns:
            List of B token-ID lists (one per stream).
        """

        self._ensure_inference_cache()
        device = audio_chunks.device
        B = state.batch_size

        if _audio_embs is not None:
            audio_chunk_embs = _audio_embs.type_as(self.embed_tokens.weight)
        else:
            # 0. Update audio feature buffer — B frames, one per stream
            if audio_chunk_lens is None:
                audio_chunk_lens = torch.tensor([audio_chunks.shape[-1]] * B, device=device)
            frames = [
                Frame(
                    samples=audio_chunks[b] if audio_chunks.dim() == 2 else audio_chunks,
                    length=int(audio_chunk_lens[b].item()),
                    stream_id=b,
                )
                for b in range(B)
            ]
            features, right_paddings = state.audio_feature_buffer.update(frames)
            # Stack B feature buffers → (B, D, fbl)
            processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)
            processed_signal_length = torch.tensor(
                [processed_signal.shape[-1] - int(rp) for rp in right_paddings],
                device=device,
            ).long()

            # 1. Encode audio chunks with streaming cache
            outputs = self.perception(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=state.audio_cache.cache_last_channel,
                cache_last_time=state.audio_cache.cache_last_time,
                cache_last_channel_len=state.audio_cache.cache_last_channel_len,
                streaming=True,
            )
            audio_chunk_embs, _, new_perception_cache = outputs

            # 2. Update streaming state with new perception cache
            if new_perception_cache is not None:
                state.audio_cache.cache_last_channel = new_perception_cache['cache_last_channel']
                state.audio_cache.cache_last_time = new_perception_cache['cache_last_time']
                state.audio_cache.cache_last_channel_len = new_perception_cache['cache_last_channel_len']

        # 3. Pad/trim to chunk_size frames
        chunk_size = self.core_cfg.chunk_size
        n_frames = audio_chunk_embs.shape[1]
        if n_frames < chunk_size:
            audio_chunk_embs = F.pad(audio_chunk_embs, (0, 0, 0, chunk_size - n_frames))
        elif n_frames > chunk_size:
            audio_chunk_embs = audio_chunk_embs[:, :chunk_size, :]

        # 4. Build input embeddings from cached turn template — (B, L, H)
        turn_ids_t = torch.tensor(self._turn_template_ids, device=device).unsqueeze(0).expand(B, -1)  # (B, L)
        audio_mask = turn_ids_t == AUDIO_TOKEN_IDX  # (B, L)

        text_tokens = turn_ids_t.where(~audio_mask, torch.zeros_like(turn_ids_t))
        input_embeds = self.embed_tokens(text_tokens)  # (B, L, H)

        # Replace audio placeholder positions with actual audio embeddings
        input_embeds[audio_mask] = audio_chunk_embs.reshape(-1, audio_chunk_embs.shape[-1])

        # 5. Forward through LLM with cache
        input_len = input_embeds.shape[1]
        state.attention_mask = torch.cat(
            [state.attention_mask, torch.ones(B, input_len, dtype=state.attention_mask.dtype, device=device)],
            dim=1,
        )
        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=state.cache,
            attention_mask=state.attention_mask,
            use_cache=True,
            return_dict=True,
        )
        state.cache = out.past_key_values
        for b in range(B):
            state.seq_lens[b] += input_len

        # 6. Autoregressive generation loop
        generated_per_stream, state.cache, footer_consumed, _ = self._autoregressive_decode(
            out.logits,
            state.cache,
            state,
            max_new_tokens,
            generation_config,
            **generation_kwargs,
        )

        # 7. Finalize turn — ensure end-of-turn tokens are in the cache.
        any_needs_footer = any(not fc for fc in footer_consumed)
        if any_needs_footer and self._asst_footer_ids:
            flen = len(self._asst_footer_ids)
            asst_footer_embs = self.embed_tokens(
                torch.tensor(self._asst_footer_ids, device=device).unsqueeze(0).expand(B, -1)
            )
            state.attention_mask = torch.cat(
                [state.attention_mask, torch.ones(B, flen, dtype=state.attention_mask.dtype, device=device)],
                dim=1,
            )
            out = self.llm(
                inputs_embeds=asst_footer_embs,
                past_key_values=state.cache,
                attention_mask=state.attention_mask,
                use_cache=True,
                return_dict=True,
            )
            state.cache = out.past_key_values
            for b in range(B):
                state.seq_lens[b] += flen
        elif all(footer_consumed):
            for b in range(B):
                state.seq_lens[b] += len(self._asst_footer_ids)

        # 8. Store and return
        for b in range(B):
            state.generated_tokens[b].extend(generated_per_stream[b])
        return generated_per_stream

    def _build_offline_emb_chunks(
        self,
        audio_wav: Tensor,
        n_samples: int,
        device: torch.device,
    ) -> list[Tensor]:
        """Pre-compute offline perception embeddings and slice into chunk_size groups.

        Runs the full perception module on the complete audio (the same path
        used during training), then splits the resulting embeddings into
        ``chunk_size``-frame groups that can be fed directly to the LLM turn
        template.  This bypasses both the feature buffer and the streaming
        encoder, isolating the LLM / generation logic from perception.

        Returns a list of ``(1, chunk_size, H)`` tensors, one per chunk.
        """
        chunk_size = self.core_cfg.chunk_size
        with torch.no_grad():
            offline_embs, _ = self.perception(
                input_signal=audio_wav.unsqueeze(0),
                input_signal_length=torch.tensor([n_samples], device=device),
            )
        total_frames = offline_embs.shape[1]
        chunks: list[Tensor] = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = offline_embs[:, start:end, :]
            if chunk.shape[1] < chunk_size:
                chunk = F.pad(chunk, (0, 0, 0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)
        return chunks

    def _generate_offline(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> list[str]:
        """Offline generation: process entire audio in a single LLM forward pass.

        Unlike the streaming path, this method runs offline perception on the
        full audio (no chunking, no streaming cache), builds one input sequence
        per sample (system prompt + user turn with all audio frames + assistant
        header), and prefills the LLM in a single forward pass before decoding.

        Args:
            audios: ``(B, T_samples)`` raw waveforms (zero-padded to max length).
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per sample.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of B transcription strings.
        """
        B = len(n_samples_list)
        if B == 0 or max(n_samples_list) == 0:
            return [""] * B
        device = audios.device
        dtype = self.embed_tokens.weight.dtype

        # 1. Encode system prompt(s)
        hf_tok = self.tokenizer.tokenizer
        prompts = [system_prompt] * B if isinstance(system_prompt, str) else system_prompt
        all_sys_embs = []
        for prompt in prompts:
            sys_ids = hf_tok.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            embs = self.embed_tokens(torch.tensor(sys_ids, device=device, dtype=torch.long).unsqueeze(0)).squeeze(
                0
            )  # (L_sys_b, H)
            all_sys_embs.append(embs)

        # 2. Embed turn template components (shared across batch)
        user_header_embs = self.embed_tokens(
            torch.tensor(self._user_header_ids, device=device, dtype=torch.long).unsqueeze(0)
        )  # (1, L_uh, H)
        uf_ah_embs = self.embed_tokens(
            torch.tensor(self._user_footer_and_asst_header_ids, device=device, dtype=torch.long).unsqueeze(0)
        )  # (1, L_uf, H)

        # 3. Run offline perception on the full batch
        audio_lens_t = torch.tensor(n_samples_list, device=device)
        batch_audio_embs, batch_emb_lens = self.perception(
            input_signal=audios,
            input_signal_length=audio_lens_t,
        )  # (B, T_enc_max, H), (B,)
        batch_audio_embs = batch_audio_embs.type_as(self.embed_tokens.weight)
        all_audio_embs = [batch_audio_embs[b, : int(batch_emb_lens[b].item())] for b in range(B)]

        # 4. Build per-sample input sequences:
        #    sys_embs[b] + user_header_embs + audio_embs[b] + user_footer_asst_header_embs
        sample_embs_list = []
        sample_lens = []
        for b in range(B):
            seq = torch.cat(
                [all_sys_embs[b], user_header_embs.squeeze(0), all_audio_embs[b], uf_ah_embs.squeeze(0)],
                dim=0,
            )  # (L_b, H)
            sample_embs_list.append(seq)
            sample_lens.append(seq.shape[0])

        # 5. Left-pad to max length and build attention mask
        max_len = max(sample_lens)
        H = sample_embs_list[0].shape[-1]
        input_embeds = torch.zeros(B, max_len, H, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b in range(B):
            offset = max_len - sample_lens[b]
            input_embeds[b, offset:] = sample_embs_list[b]
            attention_mask[b, offset:] = 1

        # 6. LLM prefill (single forward pass)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        # 7. Autoregressive decode
        state = StreamingState(
            cache=out.past_key_values,
            generated_tokens=[[] for _ in range(B)],
            seq_lens=[max_len] * B,
            attention_mask=attention_mask,
            batch_size=B,
        )
        generated_per_stream, _, _, _ = self._autoregressive_decode(
            out.logits,
            out.past_key_values,
            state,
            max_new_tokens,
            generation_config,
            **generation_kwargs,
        )

        # 8. Decode tokens to text
        return [decode_with_blank(toks, self.blank_token, self.tokenizer) for toks in generated_per_stream]

    def _generate_dynamic_streaming(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        inference_chunk_size: Optional[int] = None,
        dynamic_min_chunk_size: int = 0,
        dynamic_max_chunk_size: Optional[int] = None,
        **generation_kwargs,
    ) -> list[str]:
        """Batched dynamic-chunking generation.

        All B streams are processed in lockstep: each step feeds exactly 1
        embedding per stream to the LLM (audio frame, template token, or
        generated text token).  Perception runs with ``inference_chunk_size``
        frames and the resulting embeddings are buffered per-stream so the
        LLM still consumes them one at a time.

        Args:
            audios: ``(B, T_samples)`` raw waveforms.
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string or list.
            max_new_tokens: Max tokens per text generation segment.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            inference_chunk_size: Number of encoder frames per perception call
                (default 1).  Embeddings are buffered and fed to the LLM one
                at a time.
            dynamic_min_chunk_size: Minimum frames before the model is allowed
                to trigger generation (default 0, no minimum).
            dynamic_max_chunk_size: Maximum frames before forcing generation.
                ``None`` means no upper bound (default).
            generation_kwargs: Per-call overrides.
        """
        B = len(n_samples_list)
        if B == 0:
            return []
        device = audios.device
        # Default inference_chunk_size: use chunk_size for fixed chunking (matches
        # training perception), 1 for dynamic chunking (frame-by-frame).
        if inference_chunk_size is None:
            N = max(self.core_cfg.chunk_size, 1)
        else:
            N = inference_chunk_size
        chunk_samples = math.ceil(N * self.core_cfg.frame_length_in_secs * self.core_cfg.sample_rate)

        # --- Init state ---
        state = self.get_init_streaming_state(system_prompt, device=device, batch_size=B)
        state.audio_feature_buffer = self.get_audio_feature_buffer(
            batch_size=B,
            chunk_size_override=N,
        )

        # --- Per-stream state machine ---
        HEADER, LISTENING, FOOTER, GENERATING, BLANK_FEED, ASST_FOOTER, DONE = range(7)
        # When user_header_ids is empty (compact template), skip HEADER entirely.
        _initial_state = LISTENING if not self._user_header_ids else HEADER
        stream_state = [_initial_state] * B
        template_pos = [0] * B  # position within current template seq
        audio_sample_idx = [0] * B  # next audio sample offset for perception
        gen_token_count = [0] * B  # tokens generated in current GENERATING phase
        last_gen_token = [self.text_pad_id] * B  # last generated token per stream
        all_tokens: list[list[int]] = [[] for _ in range(B)]

        # Per-stream audio embedding buffer (filled by perception, consumed 1 at a time)
        audio_emb_buf: list[list[Tensor]] = [[] for _ in range(B)]

        # Fixed-chunk mode: count frames consumed per segment to transition
        # after exactly chunk_size frames (ignoring model predictions).
        fixed_chunk_mode = self.core_cfg.chunk_size > 0
        fixed_chunk_size = self.core_cfg.chunk_size if fixed_chunk_mode else 0
        frames_in_segment = [0] * B  # frames consumed in current LISTENING segment

        uf_ah_ids = self._user_footer_and_asst_header_ids
        uh_ids = self._user_header_ids
        af_ids = self._asst_footer_ids
        user_footer_first_id = self._user_footer_first_id

        # Max steps: LLM's max context length minus the system prompt already in KV cache.
        max_model_len = getattr(self.llm.config, 'max_position_embeddings', 40960)
        max_steps = max_model_len - max(state.seq_lens)

        # Padding embedding for DONE streams and empty-buffer LISTENING streams.
        # Use the blank token embedding when blank is enabled (a real token the
        # model knows). Otherwise fall back to the text pad id.
        pad_token_id = self.blank_token_id if self.has_blank else self.text_pad_id
        pad_emb = self.embed_tokens(torch.tensor([pad_token_id], device=device)).squeeze(0)  # (H,)

        for _step in range(max_steps):
            # --- Refill audio embedding buffers for LISTENING streams ---
            needs_refill = [
                b
                for b in range(B)
                if stream_state[b] == LISTENING
                and len(audio_emb_buf[b]) == 0
                and audio_sample_idx[b] < n_samples_list[b]
            ]
            if needs_refill:
                # Run perception only for streams that need refill.
                # The feature buffer selectively updates via stream_id.
                # The encoder cache is sliced to the subset, then scattered back.
                idx_t = torch.tensor(needs_refill, device=device)

                # Build frames (only for refill streams)
                frames = []
                for b in needs_refill:
                    start = audio_sample_idx[b]
                    end = min(start + chunk_samples, n_samples_list[b])
                    wav = audios[b, start:end]
                    if wav.shape[0] < chunk_samples:
                        wav = F.pad(wav, (0, chunk_samples - wav.shape[0]))
                    frames.append(Frame(samples=wav, stream_id=b, length=end - start))
                    audio_sample_idx[b] = end

                # Feature buffer selectively updates only the submitted stream_ids
                features, right_paddings = state.audio_feature_buffer.update(frames)
                processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)  # (S, D, T)
                processed_signal_length = torch.tensor(
                    [processed_signal.shape[-1] - int(rp) for rp in right_paddings],
                    device=device,
                ).long()

                # Slice encoder cache to the subset
                sub_cache_lc = state.audio_cache.cache_last_channel.index_select(1, idx_t)
                sub_cache_lt = state.audio_cache.cache_last_time.index_select(1, idx_t)
                sub_cache_lcl = state.audio_cache.cache_last_channel_len[idx_t]

                outputs = self.perception(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=sub_cache_lc,
                    cache_last_time=sub_cache_lt,
                    cache_last_channel_len=sub_cache_lcl,
                    streaming=True,
                )
                batch_embs, _, new_cache = outputs

                # Scatter updated cache back into the full B-sized cache
                if new_cache is not None:
                    for i, b in enumerate(needs_refill):
                        state.audio_cache.cache_last_channel[:, b] = new_cache['cache_last_channel'][:, i]
                        state.audio_cache.cache_last_time[:, b] = new_cache['cache_last_time'][:, i]
                        state.audio_cache.cache_last_channel_len[b] = new_cache['cache_last_channel_len'][i]

                # Distribute embeddings into per-stream buffers.
                # Pad to exactly N frames (matching fast path's pad/trim behavior).
                H_enc = batch_embs.shape[-1]
                for i, b in enumerate(needs_refill):
                    n_enc = batch_embs[i].shape[0]
                    for f in range(n_enc):
                        audio_emb_buf[b].append(batch_embs[i, f])
                    # Pad with zeros if encoder returned fewer than N frames
                    for _ in range(N - n_enc):
                        audio_emb_buf[b].append(torch.zeros(H_enc, device=device, dtype=batch_embs.dtype))

            # --- Build (B, 1, H) input embeddings based on per-stream state ---
            # Each entry is (H,); we stack → (B, H) then unsqueeze → (B, 1, H).
            embs_list = []
            for b in range(B):
                if stream_state[b] == LISTENING:
                    if audio_emb_buf[b]:
                        embs_list.append(audio_emb_buf[b].pop(0))  # (H,)
                    else:
                        embs_list.append(pad_emb)
                elif stream_state[b] == FOOTER:
                    tid = uf_ah_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                elif stream_state[b] == GENERATING:
                    embs_list.append(
                        self.embed_tokens(torch.tensor([last_gen_token[b]], device=device)).squeeze(0)  # (H,)
                    )
                elif stream_state[b] == BLANK_FEED:
                    # Only reached when has_blank is True (guarded at transition sites).
                    embs_list.append(
                        self.embed_tokens(torch.tensor([self.blank_token_id], device=device)).squeeze(0)  # (H,)
                    )
                elif stream_state[b] == ASST_FOOTER:
                    tid = af_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                elif stream_state[b] == HEADER:
                    tid = uh_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                else:  # DONE
                    embs_list.append(pad_emb)

            input_embs = torch.stack(embs_list).unsqueeze(1)  # (B, H) → (B, 1, H)

            # --- Single LLM forward ---
            use_aux = self.core_cfg.use_chunk_classifier and self.core_cfg.chunk_classifier_use_at_inference
            llm_kwargs = dict(
                inputs_embeds=input_embs,
                past_key_values=state.cache,
                use_cache=True,
                output_hidden_states=use_aux,
                return_dict=True,
            )
            if state.attention_mask is not None:
                state.attention_mask = torch.cat(
                    [state.attention_mask, torch.ones(B, 1, dtype=state.attention_mask.dtype, device=device)],
                    dim=1,
                )
                llm_kwargs["attention_mask"] = state.attention_mask
            out = self.llm(**llm_kwargs)
            state.cache = out.past_key_values
            for b in range(B):
                state.seq_lens[b] += 1

            # --- Aux chunk-boundary classifier: full-sequence forward ---
            # Append the new LLM hidden state to the running buffer, then run
            # the K-layer aux backbone over the entire accumulated buffer
            # (no aux KV cache — matches training, where the aux backbone sees
            # the full sequence in one pass). Cost is K layers × current length
            # per step; cheap at K≈2.
            aux_last_hidden = None
            if use_aux:
                new_h = out.hidden_states[-1]  # (B, 1, H)
                if state.aux_hidden_buffer is None:
                    state.aux_hidden_buffer = new_h
                else:
                    state.aux_hidden_buffer = torch.cat([state.aux_hidden_buffer, new_h], dim=1)
                aux_out = self.chunk_classifier_backbone(
                    inputs_embeds=state.aux_hidden_buffer,
                    attention_mask=state.attention_mask,
                    return_dict=True,
                )
                aux_last_hidden = aux_out.last_hidden_state  # (B, L_so_far, H)

            # --- Per-stream state transitions ---
            for b in range(B):
                if stream_state[b] == DONE:
                    continue

                if stream_state[b] == LISTENING:
                    frames_in_segment[b] += 1
                    if fixed_chunk_mode:
                        # Fixed chunking: transition after exactly chunk_size frames
                        if frames_in_segment[b] >= fixed_chunk_size:
                            stream_state[b] = FOOTER
                            template_pos[b] = 0
                            frames_in_segment[b] = 0
                        elif not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                            stream_state[b] = DONE
                    else:
                        # Dynamic chunking: transition when model predicts <user_footer>,
                        # subject to [min_chunk_size, max_chunk_size] bounds.
                        if dynamic_max_chunk_size is not None and frames_in_segment[b] >= dynamic_max_chunk_size:
                            # Forced transition — hit upper bound
                            stream_state[b] = FOOTER
                            template_pos[b] = 0
                            frames_in_segment[b] = 0
                        elif frames_in_segment[b] < dynamic_min_chunk_size:
                            # Below minimum — ignore model prediction, keep listening
                            if not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                                # Audio exhausted before reaching min — still emit
                                stream_state[b] = FOOTER
                                template_pos[b] = 0
                                frames_in_segment[b] = 0
                        else:
                            # In [min, max] window — use model prediction.
                            # Either the aux classifier head (when enabled) or
                            # the LM head's vocab sample (legacy path).
                            if use_aux and aux_last_hidden is not None:
                                h_last = aux_last_hidden[b, -1, :]  # (H,)
                                p_emit = torch.sigmoid(self.chunk_classifier_head(h_last)).item()
                                emit = p_emit >= self.core_cfg.chunk_classifier_threshold
                            else:
                                token = self._sample_token(
                                    out.logits[b : b + 1, -1, :],
                                    None,
                                    generation_config,
                                    **generation_kwargs,
                                ).item()
                                emit = token == user_footer_first_id
                            if emit:
                                stream_state[b] = FOOTER
                                template_pos[b] = 0
                                frames_in_segment[b] = 0
                            elif not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                                stream_state[b] = DONE

                elif stream_state[b] == FOOTER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(uf_ah_ids):
                        stream_state[b] = GENERATING
                        gen_token_count[b] = 0
                        # Use the logit from this step as the first generation logit
                        first_token = self._sample_token(
                            out.logits[b : b + 1, -1, :],
                            None,
                            generation_config,
                            **generation_kwargs,
                        ).item()
                        first_is_stop = (
                            self._eos_id is not None and first_token == self._eos_id
                        ) or first_token == self.blank_token_id
                        if first_is_stop:
                            # Immediately done generating — append chunk separator
                            # (blank when enabled, else EOS so decode_with_blank splits chunks)
                            all_tokens[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                            if fixed_chunk_mode and self.has_blank:
                                # Feed blank to LLM first (matches training sequence)
                                stream_state[b] = BLANK_FEED
                            elif af_ids:
                                stream_state[b] = ASST_FOOTER
                                template_pos[b] = 0
                            else:
                                self._dynamic_finish_generating(
                                    b,
                                    stream_state,
                                    template_pos,
                                    audio_emb_buf,
                                    audio_sample_idx,
                                    n_samples_list,
                                    _initial_state,
                                    DONE,
                                )
                        else:
                            all_tokens[b].append(first_token)
                            last_gen_token[b] = first_token
                            gen_token_count[b] = 1

                elif stream_state[b] == GENERATING:
                    token = self._sample_token(
                        out.logits[b : b + 1, -1, :],
                        None,
                        generation_config,
                        **generation_kwargs,
                    ).item()
                    # Stop on EOS, blank, or max tokens (matching _autoregressive_decode)
                    is_eos = self._eos_id is not None and token == self._eos_id
                    is_blank = token == self.blank_token_id
                    is_max = gen_token_count[b] >= max_new_tokens
                    if is_eos or is_blank or is_max:
                        # Append chunk separator (blank when enabled, else EOS).
                        # decode_with_blank splits per-chunk outputs on this.
                        all_tokens[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                        # In fixed-chunk mode, blank must be fed to LLM before footer
                        # (training sequence: ...text <blank> <asst_footer>...).
                        # This applies for ALL stop conditions (EOS, blank, max).
                        if fixed_chunk_mode and self.has_blank:
                            stream_state[b] = BLANK_FEED
                        elif af_ids:
                            stream_state[b] = ASST_FOOTER
                            template_pos[b] = 0
                        else:
                            self._dynamic_finish_generating(
                                b,
                                stream_state,
                                template_pos,
                                audio_emb_buf,
                                audio_sample_idx,
                                n_samples_list,
                                _initial_state,
                                DONE,
                            )
                    else:
                        all_tokens[b].append(token)
                        last_gen_token[b] = token
                        gen_token_count[b] += 1

                elif stream_state[b] == BLANK_FEED:
                    # Blank was fed to LLM this step. Transition to ASST_FOOTER.
                    if af_ids:
                        stream_state[b] = ASST_FOOTER
                        template_pos[b] = 0
                    else:
                        self._dynamic_finish_generating(
                            b,
                            stream_state,
                            template_pos,
                            audio_emb_buf,
                            audio_sample_idx,
                            n_samples_list,
                            _initial_state,
                            DONE,
                        )

                elif stream_state[b] == ASST_FOOTER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(af_ids):
                        self._dynamic_finish_generating(
                            b,
                            stream_state,
                            template_pos,
                            audio_emb_buf,
                            audio_sample_idx,
                            n_samples_list,
                            _initial_state,
                            DONE,
                        )

                elif stream_state[b] == HEADER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(uh_ids):
                        stream_state[b] = LISTENING

            if all(s == DONE for s in stream_state):
                break

        return [decode_with_blank(toks, self.blank_token, self.tokenizer) for toks in all_tokens]

    @staticmethod
    def _dynamic_finish_generating(
        b,
        stream_state,
        template_pos,
        audio_emb_buf,
        audio_sample_idx,
        n_samples_list,
        next_listen_state,
        DONE,
    ):
        """Transition stream b from GENERATING to next_listen_state (HEADER or LISTENING) or DONE.

        ``next_listen_state`` is HEADER when user_header_ids is non-empty, or LISTENING
        when user_header_ids is empty (compact template) — the HEADER state is skipped
        since there are no header tokens to feed.
        """
        has_more_audio = bool(audio_emb_buf[b]) or audio_sample_idx[b] < n_samples_list[b]
        if has_more_audio:
            stream_state[b] = next_listen_state
            template_pos[b] = 0
        else:
            stream_state[b] = DONE

    def _generate_chunked_streaming(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        **generation_kwargs,
    ) -> list[str]:
        """Chunk-by-chunk streaming generation for B samples in lockstep.

        Args:
            audios: ``(B, T_samples)`` raw waveforms (zero-padded to max length).
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            use_offline_embs: When True, bypass streaming perception with offline embeddings.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of B transcription strings.
        """
        assert self.core_cfg.chunk_size > 0, (
            f"chunk_size must be positive for streaming mode, got {self.core_cfg.chunk_size}. "
            f"Use generate() which dispatches to _generate_offline() for chunk_size < 0."
        )
        B = len(n_samples_list)
        if B == 0 or max(n_samples_list) == 0:
            return [""] * B
        device = audios.device
        chunk_size = self.core_cfg.chunk_size
        chunk_samples = math.ceil(chunk_size * self.core_cfg.frame_length_in_secs * self.core_cfg.sample_rate)
        state = self.get_init_streaming_state(system_prompt, device=device, batch_size=B)

        offline_emb_chunks_list = None
        if use_offline_embs:
            offline_emb_chunks_list = [
                self._build_offline_emb_chunks(audios[b, : n_samples_list[b]], n_samples_list[b], device)
                for b in range(B)
            ]

        num_chunks_per_stream = [math.ceil(ns / chunk_samples) if ns > 0 else 0 for ns in n_samples_list]
        max_chunks = max(num_chunks_per_stream)
        all_token_ids: list[list[int]] = [[] for _ in range(B)]

        for chunk_i in range(max_chunks):
            # Build B audio chunks (zero-pad finished streams)
            chunks = []
            chunk_lens = []
            for b in range(B):
                start = chunk_i * chunk_samples
                end = min(start + chunk_samples, n_samples_list[b])
                if start >= n_samples_list[b]:
                    # Stream b has finished — send zeros with zero valid length
                    chunks.append(torch.zeros(chunk_samples, device=device, dtype=audios.dtype))
                    chunk_lens.append(0)
                else:
                    wav = audios[b, start:end]
                    if wav.shape[0] < chunk_samples:
                        wav = F.pad(wav, (0, chunk_samples - wav.shape[0]))
                    chunks.append(wav)
                    chunk_lens.append(end - start)

            audio_batch = torch.stack(chunks)  # (B, chunk_samples)
            lens_batch = torch.tensor(chunk_lens, device=device)

            extra_kwargs = {}
            if offline_emb_chunks_list is not None:
                emb_chunks = []
                for b in range(B):
                    if chunk_i < len(offline_emb_chunks_list[b]):
                        emb_chunks.append(offline_emb_chunks_list[b][chunk_i])
                    else:
                        H = offline_emb_chunks_list[0][0].shape[-1]
                        emb_chunks.append(torch.zeros(1, chunk_size, H, device=device, dtype=audios.dtype))
                extra_kwargs["_audio_embs"] = torch.cat(emb_chunks, dim=0)

            chunk_tokens = self._chunked_streaming_step(
                audio_batch,
                lens_batch,
                state,
                max_new_tokens,
                generation_config,
                **extra_kwargs,
                **generation_kwargs,
            )
            for b in range(B):
                # Only collect tokens for streams that are still active
                if chunk_i < num_chunks_per_stream[b]:
                    all_token_ids[b].extend(chunk_tokens[b])

        return [decode_with_blank(toks, self.blank_token, self.tokenizer) for toks in all_token_ids]

    @torch.no_grad()
    def generate(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        system_prompt: Union[str, List[str]] = "Transcribe the audio into text.",
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        use_state_machine_inference: bool = False,
        dynamic_min_chunk_size: int = 0,
        dynamic_max_chunk_size: Optional[int] = None,
        **generation_kwargs,
    ) -> list[str]:
        """
        Transcribe full audio(s).

        Args:
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace GenerationConfig object.
            use_offline_embs: When True, bypass streaming perception with
                offline embeddings. Diagnostic use only.
            dynamic_min_chunk_size: For dynamic chunking — minimum frames before
                the model is allowed to trigger generation (default 0).
            dynamic_max_chunk_size: For dynamic chunking — maximum frames before
                forcing generation. ``None`` means no upper bound (default).
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of transcription strings, one per sample.
        """
        self._ensure_inference_cache()

        with move_embedding(self):
            B = audios.shape[0]
            n_samples_list = [int(audio_lens[b].item()) for b in range(B)]

            if self.core_cfg.chunk_size < 0:
                results = self._generate_offline(
                    audios,
                    n_samples_list,
                    system_prompt,
                    max_new_tokens,
                    generation_config,
                    **generation_kwargs,
                )
            elif self.core_cfg.chunk_size == 0 or use_state_machine_inference:
                # Dynamic chunking (chunk_size=0) or state machine inference opted in for chunk_size > 0.
                # Note that for chunk_size > 0, use_state_machine_inference is not recommended.
                results = self._generate_dynamic_streaming(
                    audios,
                    n_samples_list,
                    system_prompt,
                    max_new_tokens,
                    generation_config,
                    dynamic_min_chunk_size=dynamic_min_chunk_size,
                    dynamic_max_chunk_size=dynamic_max_chunk_size,
                    **generation_kwargs,
                )
            else:
                # Static chunking (chunk_size > 0): bulk prefill + auto-regressive decode.
                results = self._generate_chunked_streaming(
                    audios,
                    n_samples_list,
                    system_prompt,
                    max_new_tokens,
                    generation_config,
                    use_offline_embs=use_offline_embs,
                    **generation_kwargs,
                )

        return results
