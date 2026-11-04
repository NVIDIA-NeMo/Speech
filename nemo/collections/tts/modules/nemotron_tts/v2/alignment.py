# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acoustic alignment for the modular Nemotron-TTS V2 model.

Initializes and restores the flow teacher, batches compact text/audio turns for
MAS, and maps its positions onto backbone attention queries. The flow and text
encoder learn from detached codec latents and conditioning-text embeddings.
"""

import math
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch import nn

from nemo.collections.tts.modules.attention_alignment import AttentionAlignmentGuidance, AttentionAlignmentTurn
from nemo.utils import logging


@dataclass
class TurnAlignmentOutput:
    """Acoustic likelihood and MAS positions on the original conversation grid."""

    loss: torch.Tensor
    token_indices: torch.Tensor
    valid_turn_count: torch.Tensor
    turn_count: torch.Tensor
    one_frame_tokens: torch.Tensor
    token_count: torch.Tensor


@dataclass
class AttentionAlignmentTrainingOutput:
    teacher: TurnAlignmentOutput
    guidance: AttentionAlignmentGuidance | None
    metrics: dict


def resolve_aligner_config(config: Mapping | None = None) -> dict:
    """Resolve the ST-TTS flow and relative-transformer architecture configuration."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('aligned_text_aligner must be a configuration mapping')
    values = dict(config or {})
    defaults = {
        'flow_type': 'sttts',
        'hidden_channels': 128,
        'num_flows': 6,
        'num_text_layers': 2,
        'text_encoder_type': 'relative_transformer',
        'text_num_heads': 2,
        'text_filter_channels': None,
        'text_kernel_size': 3,
        'text_dropout': 0.1,
        'text_rel_attn_window_size': 4,
        'use_speaker_conditioning': False,
        'speaker_hidden_channels': 64,
        'speaker_embedding_dim': 64,
        'speaker_num_layers': 2,
        'coupling_num_layers': 4,
        'coupling_kernel_size': 5,
        'flow_n_split': 4,
        'max_log_scale': None,
        'dequantization_noise': 0.01,
        'mas_backend': 'auto',
    }
    unknown = values.keys() - defaults.keys() - {'audio_channels', 'text_channels'}
    if unknown:
        raise ValueError(f'Unknown aligned_text_aligner options: {sorted(unknown)}')
    defaults.update(values)
    for name, supported in (('flow_type', 'sttts'), ('text_encoder_type', 'relative_transformer')):
        if defaults[name] != supported:
            raise ValueError(f'aligned_text_aligner.{name} must be {supported!r}, got {defaults[name]!r}')
    return defaults


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


def load_pretrained_text_speech_aligner(
    aligner: nn.Module, checkpoint_path: str | Path | None, *, aligner_config: Mapping | None = None
) -> tuple[nn.Module, dict]:
    """Return a completely initialized aligner and its architecture configuration.

    Only ``text_aligner.*`` weights (including common model/DDP wrappers and the
    modular ``alignment.text_aligner.*`` namespace) are used. Bare aligner state
    dictionaries and Lightning ``state_dict`` checkpoints are also accepted.
    A ``.nemo`` donor supplies its own ``aligned_text_aligner`` architecture,
    including optional codec-reference speaker conditioning. Its audio/text
    feature dimensions must match those supplied by the receiving TTS model.

    No source TTS model is instantiated and no receiving module is modified.
    Loading into a separate candidate makes failures atomic. Call before the
    optimizer is created, then assign the returned module to ``text_aligner``.
    ``None`` preserves the receiving object. Non-NeMo checkpoint metadata uses
    pickle and must come from a trusted source.
    """
    config = (
        resolve_aligner_config(aligner_config)
        if aligner_config is not None
        else {key: getattr(aligner, key, default) for key, default in resolve_aligner_config().items()}
    )
    if checkpoint_path is None:
        return aligner, config
    source_path = str(checkpoint_path)
    state, source_config = _read_checkpoint(source_path)
    if source_config is not None:
        config = resolve_aligner_config(source_config)
    for name in ('audio_channels', 'text_channels'):
        supplied = getattr(aligner, name)
        requested = config.pop(name, supplied)
        if requested != supplied:
            raise ValueError(
                f'Checkpoint {source_path!r} requires aligner {name}={requested}, '
                f'but the receiving model provides {supplied}.'
            )

    # Candidate initialization must not advance TTS dropout/dequantization RNG.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        candidate = type(aligner)(audio_channels=aligner.audio_channels, text_channels=aligner.text_channels, **config)
    parameter = next(aligner.parameters(), None)
    if parameter is not None:
        candidate = candidate.to(device=parameter.device, dtype=parameter.dtype)
    candidate.train(aligner.training)
    trainable = {name: parameter.requires_grad for name, parameter in aligner.named_parameters()}
    for name, parameter in candidate.named_parameters():
        parameter.requires_grad_(trainable.get(name, True))

    expected = candidate.state_dict()
    selected = _select_aligner_weights(state, expected, source_path)
    if any(isinstance(value, torch.Tensor) and value.is_meta for value in selected.values()):
        raise ValueError(f'Checkpoint {source_path!r} is incompatible with text_aligner: meta tensors have no values.')
    try:
        candidate.load_state_dict(selected, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f'Checkpoint {source_path!r} is incompatible with text_aligner: {error}') from error
    return candidate, config


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

    model.text_aligner, settings = load_pretrained_text_speech_aligner(
        model.text_aligner, path, aligner_config=model.aligned_text_aligner_config
    )
    model.aligned_text_aligner_config = settings
    with open_dict(model.cfg):
        model.cfg.aligned_text_aligner = settings
        model.cfg.pretrained_model_text_speech_aligner = path


def alignment_schedule(settings, step):
    elapsed = int(step) - settings['start_step']
    ramp = float(elapsed >= 0) if settings['ramp_steps'] == 0 else min(1.0, max(0.0, elapsed / settings['ramp_steps']))
    decay = min(1.0, max(0.0, (elapsed - settings['ramp_steps']) / settings['prior_anneal_steps']))
    return ramp, settings['prior_strength'] * ramp * (1.0 - decay)


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


def align_text_turns(
    aligner: nn.Module,
    audio_features: torch.Tensor,
    text_tokens: torch.Tensor,
    text_lens: torch.Tensor,
    turn_indices: torch.Tensor,
    embed_text: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_turns_per_batch: int = 32,
    max_frames_per_batch: int = 8192,
    speaker_reference_features: torch.Tensor | None = None,
    speaker_reference_lens: torch.Tensor | None = None,
) -> TurnAlignmentOutput:
    """Align compact text turns and restore MAS positions to the conversation grid.

    ``turn_indices`` has rows ``(sample, start, end_exclusive)`` on the acoustic
    grid. Text contains content tokens only: the caller owns BOS/EOS handling,
    channel delays and autoregressive shifts. Acoustic inputs and the existing
    text embeddings are detached from the TTS generator; the likelihood still
    trains every aligner parameter.

    Turns are sorted by audio length and packed under a padded-frame budget.
    No acoustic-score tensor is transferred to CPU. Only the small index/length
    table is copied once to organize batches. Invalid turns are excluded from
    likelihood and attention supervision. A single turn longer than the frame
    budget raises rather than truncating speech.

    Detached MAS positions restart from zero in each compact transcript;
    invalid frames contain -1. These are positions, not vocabulary token IDs.

    Optional speaker references use the conversation batch axis, rather than
    the compact turn axis. The learnable encoder pools each reference once;
    turn packs gather its vector using the original sample index. Reference
    features are detached, while the encoder and flow conditioning learn from
    the acoustic likelihood. Missing references use zero conditioning.
    """
    if audio_features.ndim != 3:
        raise ValueError('Expected audio_features [B,T,C]')
    if text_tokens.ndim != 2 or text_lens.shape != (text_tokens.size(0),):
        raise ValueError('Expected compact text_tokens [turns,N] and text_lens [turns]')
    if turn_indices.shape != (text_tokens.size(0), 3):
        raise ValueError('turn_indices must have shape [turns,3]')
    if max_turns_per_batch < 1 or max_frames_per_batch < 1:
        raise ValueError('Turn and frame batch limits must be positive')
    if text_lens.dtype not in (torch.int32, torch.int64) or turn_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('Text lengths and turn indices must contain integers')
    if text_tokens.device != audio_features.device:
        raise ValueError('Audio and compact text must be on the same device')

    metadata = torch.cat([turn_indices, text_lens.to(turn_indices.device).unsqueeze(1)], dim=1).detach().cpu().tolist()
    batch_size, audio_steps, channels = audio_features.shape
    valid_rows = []
    for row, (sample, start, end, num_text) in enumerate(metadata):
        if (
            0 <= sample < batch_size
            and 0 <= start < end <= audio_steps
            and 0 < num_text <= text_tokens.size(1)
            and end - start >= num_text
        ):
            if end - start > max_frames_per_batch:
                raise ValueError(
                    f'Turn {row} has {end - start} frames, exceeding max_frames_per_batch={max_frames_per_batch}'
                )
            valid_rows.append(row)
    # Overlapping turns would make restoration depend on the sorting order.
    previous_ends = {}
    for row in sorted(valid_rows, key=lambda index: metadata[index][:2]):
        sample, start, end, _ = metadata[row]
        if start < previous_ends.get(sample, 0):
            raise ValueError('Alignment turn spans must not overlap within a sample')
        previous_ends[sample] = end
    valid_rows.sort(key=lambda index: metadata[index][2] - metadata[index][1])
    batches = _turn_batches(valid_rows, metadata, max_turns_per_batch, max_frames_per_batch)

    token_indices = text_tokens.new_full((batch_size, audio_steps), -1)
    weighted_loss = audio_features.new_zeros((), dtype=torch.float32)
    counts = torch.zeros(4, dtype=torch.long, device=audio_features.device)
    speaker_embeddings = None
    if getattr(aligner, 'use_speaker_conditioning', False):
        if (speaker_reference_features is None) != (speaker_reference_lens is None):
            raise ValueError('Speaker reference features and lengths must be provided together')
        if speaker_reference_features is not None:
            if speaker_reference_features.size(0) != batch_size:
                raise ValueError('Speaker references must use the conversation batch dimension')
            speaker_embeddings = aligner.encode_speaker_reference(
                speaker_reference_features.detach(), speaker_reference_lens
            )
    # Counts: valid turns, one-frame text tokens, text tokens, acoustic frames.
    for batch_rows in batches:
        max_audio = max(metadata[row][2] - metadata[row][1] for row in batch_rows)
        max_text = max(metadata[row][3] for row in batch_rows)
        packed_audio = audio_features.new_zeros(len(batch_rows), max_audio, channels)
        audio_lengths = []
        for packed_row, row in enumerate(batch_rows):
            sample, start, end, _ = metadata[row]
            packed_audio[packed_row, : end - start] = audio_features[sample, start:end].detach()
            audio_lengths.append(end - start)
        row_indices = torch.tensor(batch_rows, device=text_tokens.device, dtype=torch.long)
        packed_text = text_tokens.index_select(0, row_indices)[:, :max_text]
        packed_text_lens = text_lens.to(audio_features.device).index_select(0, row_indices)
        packed_audio_lens = torch.tensor(audio_lengths, device=audio_features.device, dtype=torch.long)
        with torch.no_grad():
            text_features = embed_text(packed_text, packed_text_lens).detach()
        speaker_kwargs = {}
        if speaker_embeddings is not None:
            sample_indices = torch.tensor(
                [metadata[row][0] for row in batch_rows], device=audio_features.device, dtype=torch.long
            )
            speaker_kwargs['speaker_embeddings'] = speaker_embeddings.index_select(0, sample_indices)
        result = aligner(packed_audio, text_features, packed_audio_lens, packed_text_lens, **speaker_kwargs)
        valid = result.valid_turns.to(dtype=torch.bool, device=audio_features.device)
        frame_count = (packed_audio_lens * valid).sum()
        weighted_loss = weighted_loss + result.loss * frame_count
        counts[0] += valid.sum()
        counts[1] += ((result.durations == 1) & valid[:, None]).sum()
        counts[2] += (packed_text_lens * valid).sum()
        counts[3] += frame_count
        for packed_row, row in enumerate(batch_rows):
            sample, start, end, _ = metadata[row]
            path = result.token_indices[packed_row, : end - start]
            valid_frames = valid[packed_row] & (path >= 0)
            token_indices[sample, start:end] = torch.where(valid_frames, path, -1)

    loss = weighted_loss / counts[3].clamp_min(1)
    if not batches:
        loss = loss + _zero_parameter_loss(aligner, audio_features)
    return TurnAlignmentOutput(
        loss=loss,
        valid_turn_count=counts[0],
        turn_count=torch.tensor(text_tokens.size(0), device=audio_features.device, dtype=torch.long),
        one_frame_tokens=counts[1],
        token_count=counts[2],
        token_indices=token_indices,
    )


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


def _read_checkpoint(source_path):
    config = None
    if Path(source_path).suffix.lower() == '.nemo':
        # Read only the two required members. Do not extract tokenizer/codec
        # assets or reconstruct the much larger source TTS model.
        with tarfile.open(source_path, 'r:*') as archive:
            members = archive.getmembers()
            weight_members = [
                member for member in members if member.isfile() and Path(member.name).name == 'model_weights.ckpt'
            ]
            config_members = [
                member for member in members if member.isfile() and Path(member.name).name == 'model_config.yaml'
            ]
            if len(weight_members) != 1 or len(config_members) > 1:
                raise ValueError(f'Checkpoint {source_path!r} must contain one unsharded model_weights.ckpt.')
            if config_members:
                content = archive.extractfile(config_members[0]).read().decode('utf-8')
                config = _find_aligner_config(OmegaConf.create(content))
            with archive.extractfile(weight_members[0]) as stream:
                state = torch.load(stream, map_location='cpu', weights_only=True)
    else:
        state = torch.load(source_path, map_location='cpu', weights_only=False)
        if isinstance(state, Mapping):
            for metadata_key in ('hyper_parameters', 'cfg', 'config', 'model_config'):
                candidate = _find_aligner_config(state.get(metadata_key))
                if candidate is not None:
                    config = candidate
                    break
    if isinstance(state, Mapping) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, Mapping):
        raise ValueError(f'Checkpoint {source_path!r} does not contain a model state dictionary.')
    return state, config


def _find_aligner_config(config):
    if not isinstance(config, Mapping):
        return None
    for wrapper in ('cfg', 'model'):
        if isinstance(config.get(wrapper), Mapping):
            config = config[wrapper]
    value = config.get('aligned_text_aligner')
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('The source aligned_text_aligner configuration must be a mapping.')
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


def _select_aligner_weights(state, expected, source_path):
    normalized = {}
    wrappers = ('model.', 'module.', '_orig_mod.')
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError(f'Checkpoint {source_path!r} contains a non-string state dictionary key.')
        while key.startswith(wrappers):
            key = key.split('.', 1)[1]
        if key in normalized:
            raise ValueError(f'Checkpoint {source_path!r} contains duplicate unwrapped state key {key!r}.')
        normalized[key] = value
    namespaces = [
        {key[len(prefix) :]: value for key, value in normalized.items() if key.startswith(prefix)}
        for prefix in ('text_aligner.', 'alignment.text_aligner.')
    ]
    namespaces = [values for values in namespaces if values]
    if len(namespaces) > 1:
        raise ValueError(f'Checkpoint {source_path!r} contains multiple text_aligner namespaces.')
    if namespaces:
        return namespaces[0]
    if expected.keys() & normalized.keys():
        return normalized
    raise ValueError(f'Checkpoint {source_path!r} contains no text_aligner weights.')


def _turn_batches(rows, metadata, max_turns, max_frames):
    batches, current = [], []
    max_audio = 0
    for row in rows:
        audio_length = metadata[row][2] - metadata[row][1]
        next_max = max(max_audio, audio_length)
        if current and (len(current) >= max_turns or (len(current) + 1) * next_max > max_frames):
            batches.append(current)
            current, max_audio = [], 0
        current.append(row)
        max_audio = max(max_audio, audio_length)
    if current:
        batches.append(current)
    return batches


def _zero_parameter_loss(module, reference):
    return sum(
        (parameter.sum() * 0.0 for parameter in module.parameters()), reference.new_zeros((), dtype=torch.float32)
    )
