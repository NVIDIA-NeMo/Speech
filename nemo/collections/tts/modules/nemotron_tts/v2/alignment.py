# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-only acoustic alignment for the modular Nemotron-TTS V2 model.

The flow sees frozen codec latents and detached input-text embeddings. Its MAS
path supervises selected backbone self-attention heads during training.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from nemo.collections.tts.modules.attention_alignment import AttentionAlignmentGuidance, AttentionAlignmentTurn
from nemo.collections.tts.modules.nemotron_tts.v2.turn_alignment import TurnAlignmentOutput, align_text_turns
from nemo.utils import logging


@dataclass
class AttentionAlignmentTrainingOutput:
    teacher: TurnAlignmentOutput
    guidance: AttentionAlignmentGuidance | None
    metrics: dict


def init_training_alignment(model, cfg):
    """Create the training teacher only when explicitly enabled."""
    model.use_attention_alignment = cfg.get('use_attention_alignment', False)
    if type(model.use_attention_alignment) is not bool:
        raise ValueError('use_attention_alignment must be a boolean')
    model.attention_alignment_config = resolve_attention_alignment_config(cfg.get('attention_alignment'))
    model.aligner_loss_weight = float(cfg.get('aligner_loss_weight', 1.0))
    if not math.isfinite(model.aligner_loss_weight) or model.aligner_loss_weight < 0:
        raise ValueError('aligner_loss_weight must be finite and nonnegative')
    for name, default in (('aligned_text_max_turns_per_batch', 32), ('aligned_text_max_frames_per_batch', 8192)):
        value = cfg.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not model.use_attention_alignment:
        return

    from nemo.collections.tts.modules.glow_tts_aligner import TinyGlowTTSAligner
    from nemo.collections.tts.modules.nemotron_tts.v2.alignment_checkpoint import (
        load_pretrained_text_speech_aligner,
        resolve_aligner_config,
    )

    restoring = bool(model._is_model_being_restored())
    settings = resolve_aligner_config(cfg.get('aligned_text_aligner'))
    channels = int(model._codec_model.vector_quantizer.codebook_dim) * model.frame_stacking_factor
    dimensions = dict(audio_channels=channels, text_channels=int(cfg.embedding_dim))
    for name, actual in dimensions.items():
        configured = settings.pop(name, actual)
        if configured != actual:
            raise ValueError(f'aligned_text_aligner.{name}={configured} does not match model features ({actual})')
    model.text_aligner = TinyGlowTTSAligner(**dimensions, **settings)
    path = cfg.get('pretrained_model_text_speech_aligner')
    if path and not restoring:
        model.text_aligner, settings = load_pretrained_text_speech_aligner(
            model.text_aligner, path, aligner_config=settings
        )
    model.aligned_text_aligner_config = settings
    # Persist the hydrated architecture and codec-ASP settings with the weights
    # to keep the checkpoint self-contained.
    with open_dict(model.cfg):
        model.cfg.aligned_text_aligner = settings
    attention_blocks = [
        index for index, kind in enumerate(model.decoder.config.layers_block_type) if kind == 'attention'
    ]
    selection = AttentionAlignmentGuidance(turns=[], layers=model.attention_alignment_config['layers'])
    chosen = selection.resolve_layers(len(attention_blocks))
    heads = model.attention_alignment_config['heads']
    if heads is not None and any(head >= model.decoder.config.num_attention_heads for head in heads):
        raise ValueError('attention_alignment.heads contains an index outside the backbone head count')
    logging.info(
        f'V2 alignment supervises main-backbone attention ordinals {chosen}, '
        f'blocks {tuple(attention_blocks[index] for index in chosen)}, heads {heads}.'
    )


def reload_pretrained_aligner(model, checkpoint_path=None):
    """Apply an explicit teacher archive after loading ordinary TTS pretraining."""
    path = checkpoint_path or model.cfg.get('pretrained_model_text_speech_aligner')
    if checkpoint_path and not model.use_attention_alignment:
        raise ValueError('pretrained_model_text_speech_aligner requires use_attention_alignment=True')
    if not model.use_attention_alignment or not path:
        return
    from nemo.collections.tts.modules.nemotron_tts.v2.alignment_checkpoint import load_pretrained_text_speech_aligner

    model.text_aligner, settings = load_pretrained_text_speech_aligner(
        model.text_aligner, path, aligner_config=model.aligned_text_aligner_config
    )
    model.aligned_text_aligner_config = settings
    with open_dict(model.cfg):
        model.cfg.aligned_text_aligner = settings
        model.cfg.pretrained_model_text_speech_aligner = path


def resolve_attention_alignment_config(config=None):
    settings = dict(config or {})
    defaults = dict(
        start_step=10000,
        ramp_steps=1000,
        prior_anneal_steps=5000,
        query_chunk_size=128,
        look_back_tokens=0,
        look_ahead_tokens=0,
        loss_weight=0.1,
        prior_strength=1.0,
        band_width=1.0,
        heads=[0],
        layers='middle',
    )
    # V2 always aligns the exact conditioning tokens; retain this legacy option
    # only as an explicit assertion rather than selecting another transcript.
    force_raw = settings.pop('force_raw_text', True)
    if type(force_raw) is not bool:
        raise ValueError('attention_alignment.force_raw_text must be a boolean')
    unknown = settings.keys() - defaults.keys()
    if unknown:
        raise ValueError(f'Unknown attention_alignment options: {sorted(unknown)}')
    defaults.update(settings)
    for name, minimum in (
        ('start_step', 0),
        ('ramp_steps', 0),
        ('prior_anneal_steps', 1),
        ('query_chunk_size', 1),
        ('look_back_tokens', 0),
        ('look_ahead_tokens', 0),
    ):
        if type(defaults[name]) is not int or defaults[name] < minimum:
            raise ValueError(f'attention_alignment.{name} must be an integer >= {minimum}')
    for name in ('loss_weight', 'prior_strength', 'band_width'):
        value = defaults[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'attention_alignment.{name} must be finite and nonnegative')
        defaults[name] = float(value)
    if defaults['band_width'] <= 0 or defaults['prior_strength'] > 1:
        raise ValueError('attention_alignment requires band_width > 0 and prior_strength <= 1')
    heads = defaults['heads']
    layers = defaults['layers']
    if OmegaConf.is_list(layers):
        layers = list(layers)
    checked = AttentionAlignmentGuidance(turns=[], heads=heads, layers=layers)
    defaults['heads'], defaults['layers'] = checked.heads, checked.layers
    return defaults


def alignment_schedule(settings, step):
    elapsed = int(step) - settings['start_step']
    ramp = float(elapsed >= 0) if settings['ramp_steps'] == 0 else min(1.0, max(0.0, elapsed / settings['ramp_steps']))
    decay = min(1.0, max(0.0, (elapsed - settings['ramp_steps']) / settings['prior_anneal_steps']))
    return ramp, settings['prior_strength'] * ramp * (1.0 - decay)


def compact_conditioning_turns(model, text, text_lens, audio_lens, is_multiturn):
    """Recover raw conditioning IDs, exact key positions, and acoustic spans.

    Multiturn text is already on the stacked acoustic grid. BOS marks the
    supervision start; text EOS ends its transcript, while the interruption
    marker supplies its exclusive speech end. The last marker may fall exactly
    at the crop boundary and therefore be absent. No lexical token is dropped.
    """
    ends = torch.div(audio_lens + model.frame_stacking_factor - 1, model.frame_stacking_factor, rounding_mode='floor')
    rows, indices, positions = [], [], []
    special = {model.pad_id, model.bos_id, model.eos_id, model.interruption_token_id}
    source = text.detach().cpu().tolist()
    lengths = text_lens.detach().cpu().tolist()
    acoustic_ends = ends.detach().cpu().tolist()
    for sample, (values, length, acoustic_end) in enumerate(zip(source, lengths, acoustic_ends)):
        values = values[:length]
        if not is_multiturn:
            keys = [i for i, token in enumerate(values) if token not in special and 0 <= token < model.text_vocab_size]
            if keys:
                rows.append([values[i] for i in keys])
                positions.append(keys)
                indices.append([sample, 0, acoustic_end])
            continue
        start, ids, keys, ended_text = None, [], [], False

        def append_turn(end):
            if ids:
                rows.append(list(ids))
                positions.append(list(keys))
                indices.append([sample, start, min(end, acoustic_end)])

        for position, token in enumerate(values):
            if token == model.bos_id:
                append_turn(position)
                start, ids, keys, ended_text = position, [], [], False
            elif token == model.interruption_token_id:
                append_turn(position)
                start, ids, keys, ended_text = None, [], [], False
            elif token == model.eos_id:
                ended_text = True
            elif token not in special and 0 <= token < model.text_vocab_size:
                if ended_text:
                    append_turn(position)
                    start, ids, keys, ended_text = position, [], [], False
                if start is None:
                    start = position
                ids.append(token)
                keys.append(position)
        append_turn(acoustic_end)
    width = max((len(row) for row in rows), default=0)
    compact = text.new_full((len(rows), width), model.pad_id)
    key_positions = text.new_full((len(rows), width), -1)
    for row, (ids, keys) in enumerate(zip(rows, positions)):
        compact[row, : len(ids)] = text.new_tensor(ids)
        key_positions[row, : len(keys)] = text.new_tensor(keys)
    return compact, text.new_tensor([len(row) for row in rows]), text.new_tensor(indices).reshape(-1, 3), key_positions


def prepare_attention_alignment_training(
    model,
    audio_codes,
    audio_lens,
    audio_targets,
    text,
    text_lens,
    context_lens,
    audio_delay,
    reference_codes=None,
    reference_lens=None,
    dropout_conditioning=False,
):
    """Train the acoustic likelihood and prepare a detached, causal attention teacher."""
    tokens, token_lens, turn_indices, key_positions = compact_conditioning_turns(
        model, text, text_lens, audio_lens, bool(model.cfg.get('use_multiturn_dataset', False))
    )
    raw_valid = valid_original_codec_frames(model, audio_codes, audio_lens)
    within = torch.arange(audio_codes.size(2), device=audio_codes.device)[None, :] < audio_lens[:, None]
    invalid = F.pad(within & ~raw_valid, (0, (-audio_codes.size(2)) % model.frame_stacking_factor))
    stacked_steps = (audio_codes.size(2) + model.frame_stacking_factor - 1) // model.frame_stacking_factor
    invalid = invalid.view(audio_codes.size(0), stacked_steps, model.frame_stacking_factor).any(-1)
    for row, (sample, start, end) in enumerate(turn_indices.detach().cpu().tolist()):
        if start < 0 or end <= start or bool(invalid[sample, start:end].any()):
            turn_indices[row, 2] = start
    features = decode_alignment_features(model, audio_codes, audio_lens)
    target_steps = audio_targets.size(2)
    features = F.pad(features, (0, 0, 0, max(0, target_steps - features.size(1))))[:, :target_steps]
    reference_features, reference_lengths = None, None
    if model.text_aligner.use_speaker_conditioning and reference_codes is not None:
        reference_codes, reference_lens = compact_speaker_reference(model, reference_codes, reference_lens)
        reference_features = decode_alignment_features(model, reference_codes, reference_lens)
        reference_lengths = torch.div(
            reference_lens + model.frame_stacking_factor - 1, model.frame_stacking_factor, rounding_mode='floor'
        )
    teacher = align_text_turns(
        model.text_aligner,
        features,
        tokens,
        token_lens,
        turn_indices,
        embed_text=lambda ids, lengths: model.embed_text_tokens(ids, text_lens=lengths),
        max_turns_per_batch=model.cfg.get('aligned_text_max_turns_per_batch', 32),
        max_frames_per_batch=model.cfg.get('aligned_text_max_frames_per_batch', 8192),
        speaker_reference_features=reference_features,
        speaker_reference_lens=reference_lengths,
    )
    settings = model.attention_alignment_config
    loss_scale, prior_strength = alignment_schedule(settings, model.global_step)
    turns, causal_frames = [], tokens.new_zeros(())
    metadata = torch.cat([turn_indices, token_lens[:, None]], dim=1).detach().cpu().tolist()
    for row, (sample, start, end, count) in enumerate(metadata):
        if start < 0 or end <= start or end > teacher.token_indices.size(1) or not count:
            continue
        path = teacher.token_indices[sample, start:end].detach()
        keys = key_positions[row, :count].long() + context_lens[sample]
        queries = torch.arange(start, end, device=text.device) + audio_delay[sample]
        selected_keys = keys.gather(0, path.clamp(0, count - 1))
        causal = (path >= 0) & (selected_keys <= queries)
        causal_frames += causal.sum()
        if not dropout_conditioning and loss_scale > 0 and causal.any():
            turns.append(AttentionAlignmentTurn(sample, queries[causal].long(), keys, path[causal].long()))
    guidance = (
        AttentionAlignmentGuidance(
            turns=turns,
            prior_strength=prior_strength,
            band_width=settings['band_width'],
            heads=settings['heads'],
            query_chunk_size=settings['query_chunk_size'],
            layers=settings['layers'],
            look_back_tokens=settings['look_back_tokens'],
            look_ahead_tokens=settings['look_ahead_tokens'],
        )
        if turns
        else None
    )
    spans = turn_indices[:, 2] - turn_indices[:, 1]
    metrics = dict(
        turns=teacher.turn_count,
        valid_turns=teacher.valid_turn_count,
        teacher_frames=teacher.token_indices.ge(0).sum(),
        causal_frames=causal_frames,
        supervised_turns=tokens.new_tensor(len(turns)),
        invalid_spans=spans.le(0).sum(),
        short_turns=((spans > 0) & (spans < token_lens)).sum(),
        empty_turns=token_lens.eq(0).sum(),
        one_frame_tokens=teacher.one_frame_tokens,
        token_count=teacher.token_count,
        loss_scale=loss_scale,
        prior_strength=prior_strength,
    )
    return AttentionAlignmentTrainingOutput(teacher, guidance, metrics)


def decode_alignment_features(model, codes, lengths):
    """Frozen original-codec latents, concatenated across each acoustic stack."""
    if codes.ndim != 3 or lengths.shape != (codes.size(0),):
        raise ValueError('Expected original codec codes [B,C,T] and lengths [B]')
    with torch.no_grad(), torch.autocast(device_type=codes.device.type, enabled=False):
        if codes.size(2) == 0:
            return torch.zeros(codes.size(0), 0, model.text_aligner.audio_channels, device=codes.device)
        mask = valid_original_codec_frames(model, codes, lengths)
        safe = codes.long().masked_fill(~mask[:, None, :], 0)
        features = model._codec_model.vector_quantizer.decode(indices=safe.transpose(0, 1), input_len=lengths).float()
        features = features.masked_fill(~mask[:, None, :], 0.0)
        stack = model.frame_stacking_factor
        features = F.pad(features, (0, (-features.size(2)) % stack))
        batch, channels, frames = features.shape
        return (
            features.view(batch, channels, frames // stack, stack)
            .permute(0, 2, 1, 3)
            .reshape(batch, frames // stack, channels * stack)
        )


def valid_original_codec_frames(model, codes, lengths):
    """Exclude controls using the original codec vocabulary, before conversion."""
    size = int(model._codec_model.codebook_size)
    within = torch.arange(codes.size(2), device=codes.device)[None, :] < lengths[:, None]
    return within & ((codes >= 0) & (codes < size)).all(dim=1)


def compact_speaker_reference(model, codes, lengths):
    """Pool real reference frames only; missing-reference control pairs have none."""
    valid = valid_original_codec_frames(model, codes, lengths)
    # With codec index conversion enabled the dummy IDs use the *new* vocabulary
    # and may numerically overlap legal original codec IDs. Match the complete
    # two-frame sentinel instead of treating every such integer as a control.
    if codes.size(2) >= 2:
        dummy = (
            lengths.eq(2)
            & codes[:, :, 0].eq(model.context_audio_bos_id).all(1)
            & codes[:, :, 1].eq(model.context_audio_eos_id).all(1)
        )
        valid = valid & ~dummy[:, None]
    counts = valid.sum(1)
    width = int(counts.max().item()) if counts.numel() else 0
    packed = codes.new_zeros(codes.size(0), codes.size(1), width)
    for row in range(codes.size(0)):
        selected = codes[row, :, valid[row]]
        packed[row, :, : selected.size(1)] = selected
    return packed, counts.to(dtype=torch.long, device=lengths.device)
