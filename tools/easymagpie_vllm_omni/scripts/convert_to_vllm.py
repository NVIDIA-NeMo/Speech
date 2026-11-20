# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Convert an EasyMagpieTTS ``.nemo`` checkpoint to a vLLM-Omni model directory.

The output directory is self-contained and ready to be passed as ``model=<dir>``
to the ``easymagpie_vllm_omni`` vLLM-Omni model
(:class:`easymagpie_vllm_omni.easymagpie.EasyMagpieTTSForConditionalGeneration`).
It contains:

* ``config.json`` — the flat HF-style config the vLLM model reads at
  construction (the Nemotron-H backbone fields + the EasyMagpie scalars consumed
  by :class:`easymagpie_vllm_omni.config.EasyMagpieOmniArch`).
* ``model.safetensors`` (+ ``model.safetensors.index.json``) — the converted
  weights using the reference EasyMagpieTTS key layout expected by the vLLM
  model's ``load_weights`` (``decoder.*`` backbone + top-level TTS submodules).
* the checkpoint's **text-conditioning tokenizer** saved via
  ``AutoTokenizer.save_pretrained`` so the model can tokenize per-request
  ``context_text`` in-engine.
* ``speaker_embeddings/<name>.pt`` (optional) — pre-computed speaker-encoder
  outputs for one or more reference audio files, selected by ``speaker_id`` at
  inference time.
* ``codec_native/`` — the causal audio codec decoder converted to a stateful
  vLLM model for the in-engine second stage.
* ``codec_encoder.safetensors`` + ``codec_encoder.json`` (only with
  ``--bundle-audio-encoders``) — the codec encoder and reference-speaker
  Transformer used for zero-shot voice cloning and multi-turn user history.

Compared to running the reference model, the character-aware subword (CAS)
encoder is collapsed into a single pre-computed lookup table mapping
``subword_id -> embedding`` (the CAS encoder is fully deterministic per subword
id, so it is baked once at conversion time and never run inside the engine). The
``decoder``'s unused token-embedding table is replaced by a tiny dummy (the
backbone is always fed via ``inputs_embeds``).

Example::

    python tools/easymagpie_vllm_omni/scripts/convert_to_vllm.py \\
        --nemo_file /path/to/EMTTS_SmallMamba.nemo \\
        --codec_model_path /path/to/25fps_spectral_codec.nemo \\
        --outdir ./easymagpie_vllm_model \\
        --context_audio /path/to/reference_voice.wav --speaker_name eng
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from math import prod

import torch
import tqdm
from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file


# Top-level checkpoint key prefixes the vLLM model's ``load_weights`` consumes
# for the TTS submodules (everything else under these names maps 1:1 into the
# vLLM model). ``text_embedding.*`` is intentionally excluded here: it is
# replaced by the pre-computed per-subword lookup table.
_TTS_PREFIXES = (
    "audio_embeddings.",
    "audio_in_projection.",
    "local_transformer.",
    "local_transformer_in_projection.",
    "local_transformer_audio_out_projection.",
    "local_transformer_out_projections.",
    "phoneme_embeddings.",
    "phoneme_final_proj.",
    "task_embedding.",
)

# The backbone token-embedding table is never consumed at runtime (the model
# runs off ``inputs_embeds``), so we ship a dummy table. It must still be >= 2:
# vLLM's profiling ``_dummy_sampler_run`` sets ``top_k = vocab_size - 1`` and then
# gathers at index ``vocab_size - top_k``, which is out of bounds for a width-1
# logits tensor (device-side "scatter gather index out of bounds" assert).
_BACKBONE_VOCAB_SIZE = 2
_PHONEME_TEXT_TOKENIZER_FILE = "phoneme_text_tokenizer/tokenizer.json"

# Nemotron-H backbone config fields forwarded into the flat vLLM ``config.json``.
# Names match the HF/vLLM Nemotron-H config (and the NeMo ``NemotronHConfig``).
_NEMOTRON_CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "attention_dropout",
    "attention_bias",
    "max_position_embeddings",
    "mamba_num_heads",
    "mamba_head_dim",
    "ssm_state_size",
    "conv_kernel",
    "n_groups",
    "chunk_size",
    "mamba_hidden_act",
    "use_conv_bias",
    "use_bias",
    "intermediate_size",
    "mlp_hidden_act",
    "mlp_bias",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "moe_shared_expert_intermediate_size",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
    "norm_topk_prob",
    "hybrid_override_pattern",
    "layer_norm_epsilon",
    "residual_in_fp32",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an EasyMagpieTTS .nemo checkpoint to a vLLM-Omni model directory."
    )
    parser.add_argument("--nemo_file", required=True, help="Path to the EasyMagpieTTS .nemo checkpoint.")
    parser.add_argument("--codec_model_path", required=True, help="Path to the audio codec .nemo checkpoint.")
    parser.add_argument("--outdir", required=True, help="Output directory for the vLLM model.")
    parser.add_argument(
        "--phoneme_tokenizer_path",
        default=None,
        help="Override the phoneme (IPA BPE) tokenizer path baked into the checkpoint.",
    )
    parser.add_argument(
        "--disable_cas_for_context_text",
        action="store_true",
        help="Set for legacy checkpoints trained without CAS embeddings on context text.",
    )
    parser.add_argument(
        "--text_tokenizer",
        default=None,
        help="HuggingFace tokenizer name/path to export. Defaults to the checkpoint's "
        "text-conditioning AutoTokenizer (`pretrained_model`).",
    )
    parser.add_argument(
        "--context_audio",
        default=None,
        help="Optional reference wav for which to pre-compute a speaker embedding.",
    )
    parser.add_argument(
        "--speaker_name",
        default="default",
        help="Name for the saved speaker embedding (speaker_embeddings/<name>.pt).",
    )
    parser.add_argument(
        "--bundle-audio-encoders",
        dest="bundle_audio_encoders",
        action="store_true",
        default=False,
        help="Explicitly bundle the codec encoder and reference-speaker encoder. This enables raw/reference-audio "
        "conditioning and can make a capable checkpoint usable for zero-shot TTS; disabled by default.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=EasyMagpieOmniArch.max_audio_seconds,
        help="Maximum duration accepted for each raw reference or user audio item. Stored in config.json and used "
        "for vLLM multimodal profiling; longer items are rejected.",
    )
    parser.add_argument("--context_audio_duration", type=float, default=5.0)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["bfloat16", "float16", "float32"],
        help="Saved weight dtype / config torch_dtype. bf16 matches the reference inference setup.",
    )
    parser.add_argument(
        "--precompute_batch_size",
        type=int,
        default=1024,
        help="Batch size for pre-computing per-subword text embeddings.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def precompute_text_embeddings(model, batch_size: int) -> torch.Tensor:
    """Bake the per-subword text embedding into a single lookup table.

    Runs ``embed_text_tokens`` (decoder subword embedding + the deterministic
    char-aware subword encoder) once per subword id so the vLLM model can replace
    the whole text-embedding path with a single ``nn.Embedding`` lookup.

    Returns:
        Tensor of shape ``[vocab_size, embedding_dim]`` (float32).
    """
    device = next(model.parameters()).device

    # New checkpoints expose the complete text id space explicitly. This is
    # required for CAS-only pronunciation-control checkpoints whose appended IPA
    # ids extend beyond CFG_UNK/interruption even though text_embedding is absent.
    if getattr(model, "text_vocab_size", None) is not None:
        vocab_size = int(model.text_vocab_size)
    elif getattr(model, "text_embedding", None) is not None:
        vocab_size = model.text_embedding.num_embeddings
    else:
        # Legacy CAS-only checkpoints end at the last ordinary text special id.
        last_special_id = max(
            int(model.cfg_unk_token_id),
            int(getattr(model, "interruption_token_id", model.cfg_unk_token_id)),
        )
        vocab_size = last_special_id + 1
    embedding_dim = int(model.cfg.embedding_dim)

    table = torch.zeros((vocab_size, embedding_dim), dtype=torch.float32, device=device)
    logging.info(f"Pre-computing text embeddings for {vocab_size} subword ids on {device}")
    for start in tqdm.tqdm(range(0, vocab_size, batch_size), desc="Pre-computing text embeddings"):
        end = min(start + batch_size, vocab_size)
        ids = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)  # (1, n)
        lens = torch.tensor([end - start], dtype=torch.long, device=device)
        embeds = model.embed_text_tokens(ids, text_lens=lens, disable_cas_embedding=False)  # (1, n, E)
        table[start:end] = embeds.squeeze(0).to(torch.float32)
    return table.cpu()


@torch.no_grad()
def extract_speaker_embedding(model, context_audio_path: str, context_audio_duration: float) -> torch.Tensor:
    """Reproduce the audio branch of ``prepare_context_tensors`` for one wav.

    Mirrors ``easy_magpietts_extract_speaker_encoding.py``: encode the (trimmed)
    reference audio to codec codes, add special tokens, frame-stack, embed the
    per-codebook tokens, and (when enabled) run the speaker encoder. Returns the
    ``(T_audio, embedding_dim)`` tensor saved as named-speaker model state.
    """
    from nemo.collections.tts.modules.magpietts_modules import add_special_tokens

    device = next(model.parameters()).device

    context_audio = model._load_audio_for_inference(context_audio_path, model.sample_rate)
    context_audio = model._adjust_audio_to_duration_for_inference(
        context_audio,
        model.sample_rate,
        context_audio_duration,
        model.codec_model_samples_per_frame,
    )
    context_audio = context_audio.to(device)
    context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long, device=device)
    context_audio_codes, context_audio_codes_lens = model._codec_helper.audio_to_codes(
        context_audio, context_audio_lens
    )

    if model._codec_converter is not None:
        context_audio_codes = model._codec_converter.convert_original_to_new(
            audio_tokens=context_audio_codes, audio_lens=context_audio_codes_lens
        ).long()

    context_audio_codes, context_audio_codes_lens = add_special_tokens(
        codes=context_audio_codes,
        codes_len=context_audio_codes_lens,
        bos_id=model.context_audio_bos_id,
        eos_id=model.context_audio_eos_id,
    )
    context_audio_codes, context_audio_codes_lens = model.stack_codes(
        context_audio_codes,
        context_audio_codes_lens,
        model.context_audio_bos_id,
        model.context_audio_eos_id,
        model.frame_stacking_factor,
        model.num_audio_codebooks,
    )

    context_audio_embedded = model.embed_audio_tokens(context_audio_codes)  # (B, T_audio, E)
    if bool(getattr(model, "use_speaker_encoder", False)):
        context_audio_embedded = model.encode_context_audio_embeddings(
            context_audio_embedded=context_audio_embedded,
            context_audio_lens=context_audio_codes_lens,
        )

    audio_len = int(context_audio_codes_lens[0].item())
    return context_audio_embedded[0, :audio_len].contiguous().float().detach().cpu()


def validate_model_config(model) -> None:
    """Validate checkpoint features implemented by the vLLM serving model."""
    cfg = model.cfg
    decoder_type = str(cfg.get("decoder_type", "huggingface"))
    if decoder_type != "nemotron_h":
        raise ValueError(
            "The easymagpie_vllm_omni model only supports a Nemotron-H backbone "
            f"(decoder_type='nemotron_h'); got '{decoder_type}'. Add a vLLM backbone adapter to support it."
        )

    hidden_dim = int(cfg.hidden_dim)
    embedding_dim = int(cfg.embedding_dim)
    if hidden_dim != embedding_dim:
        raise ValueError(
            "hidden_dim must equal embedding_dim because the current vLLM model has no projection between them; "
            f"got hidden_dim={hidden_dim}, embedding_dim={embedding_dim}. Add the projection to support this checkpoint."
        )
    backbone_hidden_dim = int(cfg.get("nemotron_h_config", {}).get("hidden_size", embedding_dim))
    if backbone_hidden_dim != embedding_dim:
        raise ValueError(
            "nemotron_h_config.hidden_size must equal embedding_dim because the backbone consumes inputs_embeds "
            f"directly; got {backbone_hidden_dim} and {embedding_dim}. Add an input projection to support this layout."
        )

    local_transformer_type = str(cfg.get("local_transformer_type", "none"))
    if local_transformer_type not in {"ar", "autoregressive"}:
        raise ValueError(
            "The serving code currently requires local_transformer_type='ar'/'autoregressive'; "
            "extend EasyMagpieCodePredictor "
            f"to support '{local_transformer_type}'."
        )

    default_mode = model.mode_name_to_mode.get(model.default_inference_mode)
    if default_mode is None:
        raise ValueError(f"default inference mode '{model.default_inference_mode}' is missing from mode_name_to_mode")
    if default_mode.text_input_mode != "streaming":
        raise ValueError(
            "The serving code currently requires text_input_mode='streaming' for the default inference mode; "
            f"got '{default_mode.text_input_mode}'. Implement full-text conditioning to support this checkpoint."
        )


def build_config(
    model,
    vocab_size: int,
    torch_dtype: str,
    *,
    bundle_audio_encoders: bool = False,
    max_audio_seconds: float = EasyMagpieOmniArch.max_audio_seconds,
) -> dict:
    """Build the flat vLLM ``config.json`` dict from the loaded NeMo model."""
    from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig

    validate_model_config(model)
    cfg = model.cfg

    hidden_dim = int(cfg.hidden_dim)
    embedding_dim = int(cfg.embedding_dim)

    # Resolve the backbone config exactly as NeMo does (fills head_dim, expands
    # the hybrid pattern to num_hidden_layers, etc.).
    nemotron_dict = dict(OmegaConf.to_container(cfg.nemotron_h_config, resolve=True))
    nemotron_dict.setdefault("hidden_size", embedding_dim)
    nemotron_cfg = NemotronHConfig(**nemotron_dict)

    config: dict = {"architectures": ["EasyMagpieTTSForConditionalGeneration"], "model_type": "nemotron_h"}
    for field in _NEMOTRON_CONFIG_FIELDS:
        if hasattr(nemotron_cfg, field):
            config[field] = getattr(nemotron_cfg, field)
    config["tie_word_embeddings"] = False
    config["torch_dtype"] = torch_dtype
    # The backbone token-embedding table is never consumed (inputs_embeds path);
    # the dummy logits width follows it. Must be >= 2 (see ``_BACKBONE_VOCAB_SIZE``).
    # The text path is driven by ``text_vocab_size`` / the baked ``text_embedding``
    # table instead.
    config["vocab_size"] = _BACKBONE_VOCAB_SIZE

    # ── EasyMagpie scalars (read by EasyMagpieOmniArch.from_hf_config) ──
    config["text_vocab_size"] = vocab_size
    config["text_eos_id"] = int(model.eos_id)
    config["use_multiturn_dataset"] = bool(cfg.get("use_multiturn_dataset", False))
    config["condition_on_user_speech"] = bool(cfg.get("condition_on_user_speech", False))
    config["use_user_speaking_token"] = bool(cfg.get("use_user_speaking_token", False))
    config["use_user_speaking_end_token"] = bool(cfg.get("use_user_speaking_end_token", False))
    config["codec_encoder_bundled"] = bundle_audio_encoders
    if bundle_audio_encoders:
        config["audio_input_token_id"] = 1
        config["max_audio_seconds"] = float(max_audio_seconds)
        speaker_encoder = getattr(model, "speaker_encoder", None)
        if speaker_encoder is None or not bool(getattr(model, "use_speaker_encoder", False)):
            raise ValueError(
                "--bundle-audio-encoders requires a checkpoint with use_speaker_encoder=True and speaker weights"
            )
        first_layer = speaker_encoder.layers[0]
        config["codec_input_sample_rate"] = int(model.sample_rate)
        config["codec_samples_per_frame"] = int(model.codec_model_samples_per_frame)
        config["reference_speaker_encoder_n_layers"] = len(speaker_encoder.layers)
        config["reference_speaker_encoder_d_ffn"] = int(first_layer.pos_ff.proj.conv.out_channels)
        config["reference_speaker_encoder_n_heads"] = int(first_layer.self_attention.n_heads)
        config["reference_speaker_encoder_kernel_size"] = int(first_layer.pos_ff.proj.conv.kernel_size[0])
        config["reference_speaker_encoder_max_length"] = int(speaker_encoder.position_embeddings.num_embeddings)
    if hasattr(model, "interruption_token_id"):
        config["text_interruption_id"] = int(model.interruption_token_id)
    config["embedding_dim"] = embedding_dim
    config["audio_embedding_dim"] = int(cfg.get("audio_embedding_dim", hidden_dim))
    config["num_audio_codebooks"] = int(model.num_audio_codebooks)
    config["codebook_size"] = int(model.codebook_size)
    config["frame_stacking_factor"] = int(model.frame_stacking_factor)

    enable_phoneme_text_input = bool(getattr(model, "enable_phoneme_text_input", False))
    config["enable_phoneme_text_input"] = enable_phoneme_text_input
    if enable_phoneme_text_input:
        text_phoneme_token_offset = int(model.text_phoneme_token_offset)
        text_phoneme_vocab_size = int(model.text_phoneme_vocab_size)
        if text_phoneme_token_offset + text_phoneme_vocab_size != vocab_size:
            raise ValueError(
                "Pronunciation-control text vocabulary must be a trailing contiguous range: "
                f"offset={text_phoneme_token_offset}, size={text_phoneme_vocab_size}, "
                f"text_vocab_size={vocab_size}."
            )
        config["text_phoneme_token_offset"] = text_phoneme_token_offset
        config["text_phoneme_vocab_size"] = text_phoneme_vocab_size
        config["phoneme_text_bop_marker"] = str(model.phoneme_text_bop_marker)
        config["phoneme_text_eop_marker"] = str(model.phoneme_text_eop_marker)
        config["text_phoneme_tokenizer_file"] = _PHONEME_TEXT_TOKENIZER_FILE

    has_phoneme = getattr(model, "phoneme_tokenizer", None) is not None
    config["phoneme_stacking_factor"] = int(getattr(model, "phoneme_stacking_factor", 0)) if has_phoneme else 0
    config["phoneme_vocab_size"] = int(getattr(model, "phoneme_vocab_size", 0)) if has_phoneme else 0
    if has_phoneme:
        # Phoneme special-token ids + the confidence→UNK replacement threshold,
        # consumed by the in-engine phoneme stream (BOS seeding, EOS-stop, UNK).
        config["phoneme_bos_id"] = int(model.phoneme_tokenizer.bos_token_id)
        config["phoneme_eos_id"] = int(model.phoneme_tokenizer.eos_token_id)
        unk_id = getattr(model.phoneme_tokenizer, "unk_token_id", None)
        if unk_id is not None:
            config["phoneme_unk_id"] = int(unk_id)
        config["phoneme_confidence_unk_threshold"] = float(getattr(model, "phoneme_confidence_unk_threshold", 0.0))

    # ── Streaming delays from the default inference mode ──
    # The reference offsets the text/phoneme/audio streams by these per-mode
    # delays; the vLLM model reproduces them in its decode step. A 0/0 (or "full")
    # mode runs the three streams in lock-step.
    default_mode = model.mode_name_to_mode.get(model.default_inference_mode)
    config["streaming_phonemes_delay"] = int(default_mode.streaming_phonemes_delay)
    config["streaming_speech_delay"] = int(default_mode.streaming_speech_delay)

    config["num_task_embeddings"] = len(model.training_modes) if model.task_embedding is not None else 0

    config["local_transformer_n_layers"] = int(cfg.get("local_transformer_n_layers", 2))
    config["local_transformer_n_heads"] = int(cfg.get("local_transformer_n_heads", 1))
    config["local_transformer_hidden_dim"] = int(cfg.get("local_transformer_hidden_dim", hidden_dim))

    # Pin the exact special-token ids (covers legacy ``forced_*`` checkpoints).
    config["forced_audio_bos_id"] = int(model.audio_bos_id)
    config["forced_audio_eos_id"] = int(model.audio_eos_id)
    config["forced_mask_token_id"] = int(model.mask_token_id)
    config["forced_audio_user_speaking_id"] = int(getattr(model, "audio_user_speaking_id", model.codebook_size + 5))
    config["forced_audio_user_speaking_end_id"] = int(
        getattr(model, "audio_user_speaking_end_id", model.codebook_size + 6)
    )

    return config


def select_weights(state_dict: dict, hidden_dim: int, dtype: torch.dtype) -> dict:
    """Select + rename checkpoint weights into the vLLM ``load_weights`` layout."""
    weights: dict = {}

    # Backbone: keep all ``decoder.*`` except the unused token-embedding table.
    for key, value in state_dict.items():
        if not key.startswith("decoder."):
            continue
        if key == "decoder.embeddings.weight":
            continue
        if key.endswith(".causal_mask"):
            continue
        weights[key] = value.to(dtype) if value.is_floating_point() else value

    # Dummy backbone embeddings (size ``_BACKBONE_VOCAB_SIZE``) — never consumed
    # at runtime; sized to match ``config.vocab_size``.
    weights["decoder.embeddings.weight"] = torch.zeros(_BACKBONE_VOCAB_SIZE, hidden_dim, dtype=dtype)

    # TTS submodules copied 1:1.
    for key, value in state_dict.items():
        if key.endswith(".causal_mask"):
            continue
        if any(key.startswith(prefix) for prefix in _TTS_PREFIXES):
            weights[key] = value.to(dtype) if value.is_floating_point() else value

    return weights


def resolve_codec_layout(model) -> tuple[int, list[int], int]:
    """Return the checkpoint codec layout supported by the native vLLM codec."""
    codec_converter = getattr(model, "_codec_converter", None)
    if codec_converter is not None:
        quantizer = getattr(codec_converter, "vector_quantizer_new", None)
    else:
        codec_model = getattr(model, "_codec_model", None)
        quantizer = getattr(codec_model, "vector_quantizer", None)

    fsqs = getattr(quantizer, "fsqs", None)
    if fsqs is None or len(fsqs) == 0:
        raise ValueError("The native vLLM codec reimplementation supports only GroupFiniteScalarQuantizer layouts")

    levels_by_group: list[list[int]] = []
    for fsq in fsqs:
        num_levels = getattr(fsq, "num_levels", None)
        if not isinstance(num_levels, torch.Tensor) or num_levels.numel() == 0:
            raise ValueError(
                "The native vLLM codec reimplementation requires each FSQ group to expose non-empty num_levels"
            )
        levels_by_group.append([int(level) for level in num_levels.reshape(-1).tolist()])

    levels = levels_by_group[0]
    if any(group_levels != levels for group_levels in levels_by_group[1:]):
        raise ValueError("The native vLLM codec reimplementation requires one shared num_levels_per_group layout")

    num_codebooks = int(getattr(quantizer, "num_codebooks", len(fsqs)))
    if num_codebooks != len(fsqs):
        raise ValueError(
            "The checkpoint codec layout is inconsistent: "
            f"num_codebooks={num_codebooks}, but the quantizer contains {len(fsqs)} FSQ groups"
        )
    if num_codebooks != int(model.num_audio_codebooks):
        raise ValueError(
            "The checkpoint codec layout is inconsistent: "
            f"quantizer has {num_codebooks} codebooks, model expects {model.num_audio_codebooks}"
        )

    codebook_size = prod(levels)
    if codebook_size != int(model.codebook_size):
        raise ValueError(
            "The checkpoint codec layout is inconsistent: "
            f"num_levels_per_group implies codebook_size={codebook_size}, model expects {model.codebook_size}"
        )

    frame_stacking_factor = int(model.frame_stacking_factor)
    if frame_stacking_factor <= 0:
        raise ValueError(f"frame_stacking_factor must be positive, got {frame_stacking_factor}")
    return num_codebooks, levels, frame_stacking_factor


def convert_codec_artifacts(
    codec_model_path: str,
    outdir: str,
    *,
    num_codebooks: int,
    frame_stacking_factor: int,
    num_levels_per_group: list[int],
    bundle_audio_encoders: bool = False,
) -> str | None:
    """Always convert the decoder; optionally export the guarded raw-audio tower."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    converter = os.path.join(script_dir, "convert_codec.py")
    output = os.path.join(outdir, "codec_native")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = project_dir + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    logging.info("Converting native stateful codec %s -> %s", codec_model_path, output)
    command = [
        sys.executable,
        converter,
        codec_model_path,
        output,
        "--num-codebooks",
        str(num_codebooks),
        "--frame-stacking-factor",
        str(frame_stacking_factor),
        "--num-levels-per-group",
        *[str(level) for level in num_levels_per_group],
    ]
    encoder_output = None
    if bundle_audio_encoders:
        encoder_output = os.path.join(outdir, "codec_encoder.safetensors")
        command.extend(["--encoder-output", encoder_output])
    subprocess.run(command, check=True, env=env)
    return encoder_output


def configure_codec_reference_speaker_encoder(outdir: str, config: dict) -> None:
    """Add checkpoint speaker-transformer metadata to ``codec_encoder.json``."""
    path = os.path.join(outdir, "codec_encoder.json")
    with open(path) as source:
        encoder_config = json.load(source)
    fields = (
        "embedding_dim",
        "reference_speaker_encoder_n_layers",
        "reference_speaker_encoder_d_ffn",
        "reference_speaker_encoder_n_heads",
        "reference_speaker_encoder_kernel_size",
        "reference_speaker_encoder_max_length",
    )
    encoder_config.update({field: config[field] for field in fields if field in config})
    encoder_config["context_audio_bos_id"] = int(config["codebook_size"]) + 2
    encoder_config["context_audio_eos_id"] = int(config["codebook_size"]) + 3
    with open(path, "w") as output_file:
        json.dump(encoder_config, output_file, indent=2)


def append_reference_speaker_encoder_weights(encoder_path: str, state_dict: dict[str, torch.Tensor]) -> None:
    """Keep codec and speaker encoder weights in one reviewable tower shard."""
    encoder_weights = load_file(encoder_path, device="cpu")
    speaker_weights = {
        f"reference_speaker_encoder.{key.removeprefix('speaker_encoder.')}": value.detach().cpu().float()
        for key, value in state_dict.items()
        if key.startswith("speaker_encoder.") and not key.endswith(".causal_mask")
    }
    if not speaker_weights:
        raise ValueError("The EasyMagpie checkpoint has no reference-speaker Transformer weights to bundle")
    encoder_weights.update(speaker_weights)
    save_file(encoder_weights, encoder_path, metadata={"format": "pt"})


def save_text_tokenizer(model, outdir: str, override: str | None) -> None:
    """Export the checkpoint's text-conditioning tokenizer into ``outdir``."""
    from transformers import AutoTokenizer

    pretrained = override
    if pretrained is None:
        tok_name = model.text_conditioning_tokenizer_name
        tok_cfg = model.cfg.text_tokenizers[tok_name]
        if tok_cfg.get("_target_", None) != "AutoTokenizer" or tok_cfg.get("pretrained_model", None) is None:
            raise ValueError(
                "Could not infer the text-conditioning AutoTokenizer from the checkpoint config. "
                "Pass --text_tokenizer explicitly."
            )
        pretrained = tok_cfg.pretrained_model

    logging.info(f"Saving text tokenizer '{pretrained}' to {outdir}")
    AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True).save_pretrained(outdir)


def save_phoneme_text_tokenizer(model, outdir: str) -> None:
    """Export the IPA BPE used by marked target-text spans when enabled."""
    if not bool(getattr(model, "enable_phoneme_text_input", False)):
        return

    phoneme_tokenizer = getattr(model, "phoneme_tokenizer", None)
    raw_tokenizer = getattr(phoneme_tokenizer, "_tokenizer", None)
    if raw_tokenizer is None:
        raise ValueError("Pronunciation-control conversion requires an IPABPETokenizer with a raw tokenizer.")

    raw_vocab_size = len(raw_tokenizer.get_vocab())
    text_phoneme_vocab_size = int(model.text_phoneme_vocab_size)
    if raw_vocab_size + 3 != text_phoneme_vocab_size:
        raise ValueError(
            "IPA tokenizer vocabulary does not match the pronunciation-control text range: "
            f"raw tokenizer size={raw_vocab_size}, reserved special tokens=3, "
            f"text range size={text_phoneme_vocab_size}."
        )

    output_path = os.path.join(outdir, _PHONEME_TEXT_TOKENIZER_FILE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_tokenizer.save(output_path)
    logging.info("Saved pronunciation-control IPA tokenizer to %s", output_path)


def convert(args) -> None:
    from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_easy_magpie_model

    os.makedirs(args.outdir, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    model, ckpt_name = load_easy_magpie_model(
        ModelLoadConfig(
            nemo_file=args.nemo_file,
            codecmodel_path=args.codec_model_path,
            phoneme_tokenizer_path=args.phoneme_tokenizer_path,
            disable_cas_for_context_text=args.disable_cas_for_context_text,
        ),
        device=args.device,
    )
    logging.info(f"Loaded EasyMagpieTTS checkpoint: {ckpt_name}")

    hidden_dim = int(model.cfg.hidden_dim)
    num_codebooks, num_levels_per_group, frame_stacking_factor = resolve_codec_layout(model)

    # ── 1. Pre-compute the per-subword text embedding table ──────────────
    text_table = precompute_text_embeddings(model, args.precompute_batch_size)
    vocab_size = int(text_table.shape[0])

    # ── 2. config.json ───────────────────────────────────────────────────
    config = build_config(
        model,
        vocab_size,
        args.dtype,
        bundle_audio_encoders=args.bundle_audio_encoders,
        max_audio_seconds=args.max_audio_seconds,
    )
    encoder_path = convert_codec_artifacts(
        args.codec_model_path,
        args.outdir,
        num_codebooks=num_codebooks,
        frame_stacking_factor=frame_stacking_factor,
        num_levels_per_group=num_levels_per_group,
        bundle_audio_encoders=args.bundle_audio_encoders,
    )
    if encoder_path is not None:
        configure_codec_reference_speaker_encoder(args.outdir, config)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logging.info("Saved config.json")

    # ── 3. weights ───────────────────────────────────────────────────────
    state_dict = model.state_dict()
    if encoder_path is not None:
        append_reference_speaker_encoder_weights(encoder_path, state_dict)
    weights = select_weights(state_dict, hidden_dim, dtype)
    weights["text_embedding.weight"] = text_table.to(dtype)

    safetensors_path = os.path.join(args.outdir, "model.safetensors")
    save_file(weights, safetensors_path, metadata={"format": "pt"})
    total_size = sum(w.numel() * w.element_size() for w in weights.values())
    weight_map = {name: "model.safetensors" for name in weights}
    if encoder_path is not None:
        encoder_weights = load_file(encoder_path, device="cpu")
        total_size += sum(weight.numel() * weight.element_size() for weight in encoder_weights.values())
        weight_map.update({name: os.path.basename(encoder_path) for name in encoder_weights})
        del encoder_weights
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.outdir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    logging.info(f"Saved {len(weights)} weights to {safetensors_path}")

    # ── 4. text tokenizer ────────────────────────────────────────────────
    save_text_tokenizer(model, args.outdir, args.text_tokenizer)
    save_phoneme_text_tokenizer(model, args.outdir)

    # ── 5. optional speaker embedding ────────────────────────────────────
    if args.context_audio is not None:
        speaker_dir = os.path.join(args.outdir, "speaker_embeddings")
        os.makedirs(speaker_dir, exist_ok=True)
        speaker_encoding = extract_speaker_embedding(model, args.context_audio, args.context_audio_duration)
        out_path = os.path.join(speaker_dir, f"{args.speaker_name}.pt")
        torch.save(
            {
                "speaker_encoding": speaker_encoding,
                "context_audio": args.context_audio,
                "embedding_dim": int(speaker_encoding.size(-1)),
                "num_frames": int(speaker_encoding.size(0)),
                "checkpoint": ckpt_name,
            },
            out_path,
        )
        logging.info(f"Saved speaker embedding '{args.speaker_name}' {tuple(speaker_encoding.shape)} to {out_path}")

    logging.info(f"Done. vLLM model directory: {args.outdir}")


if __name__ == "__main__":
    convert(parse_args())
