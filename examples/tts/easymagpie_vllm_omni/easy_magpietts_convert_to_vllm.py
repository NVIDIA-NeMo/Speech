# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
  outputs for one or more reference audio files, used as the ``speaker_embedding``
  input at inference time.

Compared to running the reference model, the character-aware subword (CAS)
encoder is collapsed into a single pre-computed lookup table mapping
``subword_id -> embedding`` (the CAS encoder is fully deterministic per subword
id, so it is baked once at conversion time and never run inside the engine). The
``decoder``'s unused token-embedding table is replaced by a tiny dummy (the
backbone is always fed via ``inputs_embeds``).

Example::

    python examples/tts/easymagpie_vllm_omni/easy_magpietts_convert_to_vllm.py \\
        --nemo_file /path/to/EMTTS_SmallMamba.nemo \\
        --codec_model_path /path/to/25fps_spectral_codec.nemo \\
        --outdir ./easymagpie_vllm_model \\
        --context_audio /path/to/reference_voice.wav --speaker_name eng
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import tqdm
from omegaconf import OmegaConf
from safetensors.torch import save_file

from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_easy_magpie_model
from nemo.collections.tts.modules.magpietts_modules import add_special_tokens
from nemo.utils import logging

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
    parser.add_argument("--context_audio_duration", type=float, default=5.0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
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

    # Vocabulary size of the subword id space (decoder text-embedding table when
    # present; otherwise the CAS-only id range, which ends at cfg_unk_token_id).
    if getattr(model, "text_embedding", None) is not None:
        vocab_size = model.text_embedding.num_embeddings
    else:
        vocab_size = int(model.cfg_unk_token_id) + 1
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
    ``(T_audio, embedding_dim)`` tensor consumed as the model's ``speaker_embedding``.
    """
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
    if getattr(model, "use_speaker_encoder", False):
        context_audio_embedded = model.encode_context_audio_embeddings(
            context_audio_embedded=context_audio_embedded,
            context_audio_lens=context_audio_codes_lens,
        )
    else:
        logging.warning(
            "Checkpoint has use_speaker_encoder=False; saving raw per-codebook audio embeddings "
            "(no speaker encoder applied)."
        )

    audio_len = int(context_audio_codes_lens[0].item())
    return context_audio_embedded[0, :audio_len].contiguous().float().detach().cpu()


def build_config(model, vocab_size: int, torch_dtype: str) -> dict:
    """Build the flat vLLM ``config.json`` dict from the loaded NeMo model."""
    from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig

    cfg = model.cfg
    if cfg.get("decoder_type", "huggingface") != "nemotron_h":
        raise ValueError(
            "The easymagpie_vllm_omni model only supports a Nemotron-H backbone "
            f"(decoder_type='nemotron_h'); got '{cfg.get('decoder_type')}'."
        )

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
    config["embedding_dim"] = embedding_dim
    config["audio_embedding_dim"] = int(cfg.get("audio_embedding_dim", hidden_dim))
    config["num_audio_codebooks"] = int(model.num_audio_codebooks)
    config["codebook_size"] = int(model.codebook_size)
    config["frame_stacking_factor"] = int(model.frame_stacking_factor)

    has_phoneme = getattr(model, "phoneme_tokenizer", None) is not None
    config["phoneme_stacking_factor"] = int(getattr(model, "phoneme_stacking_factor", 0)) if has_phoneme else 0
    config["phoneme_vocab_size"] = int(getattr(model, "phoneme_vocab_size", 0)) if has_phoneme else 0

    config["num_task_embeddings"] = len(model.training_modes) if model.task_embedding is not None else 0

    config["local_transformer_n_layers"] = int(cfg.get("local_transformer_n_layers", 2))
    config["local_transformer_n_heads"] = int(cfg.get("local_transformer_n_heads", 1))
    config["local_transformer_hidden_dim"] = int(cfg.get("local_transformer_hidden_dim", hidden_dim))

    # Pin the exact special-token ids (covers legacy ``forced_*`` checkpoints).
    config["forced_audio_bos_id"] = int(model.audio_bos_id)
    config["forced_audio_eos_id"] = int(model.audio_eos_id)
    config["forced_mask_token_id"] = int(model.mask_token_id)

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


def convert(args) -> None:
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

    # ── 1. Pre-compute the per-subword text embedding table ──────────────
    text_table = precompute_text_embeddings(model, args.precompute_batch_size)
    vocab_size = int(text_table.shape[0])

    # ── 2. config.json ───────────────────────────────────────────────────
    config = build_config(model, vocab_size, args.dtype)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logging.info("Saved config.json")

    # ── 3. weights ───────────────────────────────────────────────────────
    state_dict = model.state_dict()
    weights = select_weights(state_dict, hidden_dim, dtype)
    weights["text_embedding.weight"] = text_table.to(dtype)

    safetensors_path = os.path.join(args.outdir, "model.safetensors")
    save_file(weights, safetensors_path, metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(w.numel() * w.element_size() for w in weights.values())},
        "weight_map": {name: "model.safetensors" for name in weights},
    }
    with open(os.path.join(args.outdir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    logging.info(f"Saved {len(weights)} weights to {safetensors_path}")

    # ── 4. text tokenizer ────────────────────────────────────────────────
    save_text_tokenizer(model, args.outdir, args.text_tokenizer)

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
