# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen Nemotron ASR encoder and CTC head for ground-truth turn timestamps.

Only the mel preprocessor, ASR encoder, and timestamp adapter are constructed. The
checkpoint's language model, modality adapter, and diarization model are never
constructed or read. Keep this teacher outside the training model's registered
module tree to exclude its weights from training checkpoints and optimizers.
"""

from __future__ import annotations

import inspect
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from nemo.utils.cast_utils import fp32_precision


def load_ctc_timestamp_tokenizer(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Read the teacher's tokenizer and dimensions without constructing its networks.

    A saved student supplies its registered ``tokenizer_model`` artifact and scalar
    metadata. Initial training instead reads only the SentencePiece tensor and JSON
    metadata from the teacher safetensors bundle. The returned ``model_proto`` bytes
    allow callers to register a self-contained NeMo tokenizer artifact.
    """
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _load_sentencepiece, _validate_vocabulary

    config = _plain_config(config)
    if config.get('tokenizer_model'):
        tokenizer_options = dict(config.get('tokenizer_options') or {})
        tokenizer = SentencePieceTokenizer(model_path=config['tokenizer_model'], **tokenizer_options)
        metadata = {
            'num_classes': int(config['num_classes']),
            'blank_id': int(config['blank_id']),
            'frame_seconds': float(config.get('frame_seconds', 0.04)),
            'upsampling_factor': int(config.get('upsampling_factor', 2)),
            'tokenizer_options': tokenizer_options,
            'model_proto': tokenizer.tokenizer.serialized_model_proto(),
        }
    else:
        from safetensors import safe_open

        teacher_config = config.get('aligner') or config
        snapshot = Path(teacher_config.get('snapshot_path', '')).expanduser()
        path = Path(teacher_config.get('ctc_artifact_path') or snapshot / 'model.safetensors').expanduser()
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            artifact_metadata = handle.metadata() or {}
            if artifact_metadata.get('format') == 'nemo_ctc_timestamp_artifact_v1':
                prefix, decoder_key, tokenizer_key = '', 'decoder_config', 'tokenizer_config'
            elif artifact_metadata.get('ctc_timestamp_format') == 'nemo_ctc_timestamp_artifact_v1':
                prefix = 'ctc_timestamp.'
                decoder_key, tokenizer_key = 'ctc_timestamp_decoder_config', 'ctc_timestamp_tokenizer_config'
            else:
                raise ValueError('The teacher bundle has no supported CTC timestamp tokenizer artifact.')
            decoder_config = json.loads(artifact_metadata[decoder_key])
            tokenizer_options = json.loads(artifact_metadata[tokenizer_key])
            proto = handle.get_tensor(prefix + 'tokenizer.model_proto')
        if proto.dtype != torch.uint8 or proto.numel() == 0:
            raise ValueError('The CTC tokenizer model must be a nonempty uint8 tensor.')
        model_proto = proto.contiguous().numpy().tobytes()
        tokenizer = _load_sentencepiece(dict(tokenizer_options, model_proto=model_proto))
        _validate_vocabulary(tokenizer, decoder_config)
        num_classes = int(decoder_config['num_classes']) + int(decoder_config.get('add_blank', True))
        upsampling = int(decoder_config.get('upsampling_factor', 1))
        metadata = {
            'num_classes': num_classes,
            'blank_id': num_classes - 1,
            'frame_seconds': 0.08 / upsampling,
            'upsampling_factor': upsampling,
            'tokenizer_options': tokenizer_options,
            'model_proto': model_proto,
        }
    if metadata['upsampling_factor'] != 2 or not math.isclose(metadata['frame_seconds'], 0.04):
        raise ValueError('CTC timestamp distillation requires the teacher two-subframe, 40 ms adapter.')
    if metadata['num_classes'] != tokenizer.vocab_size + 1 or metadata['blank_id'] != tokenizer.vocab_size:
        raise ValueError('CTC tokenizer vocabulary and the appended blank class must match the teacher output.')
    return tokenizer, metadata


class NemotronASRTimestampAligner(nn.Module):
    """Align independent ground-truth audio turns with a frozen Nemotron teacher.

    ``config.snapshot_path`` points to a local Nemotron 3.5 Transcribe Hugging Face
    snapshot containing ``config.json`` and ``model.safetensors``. ``encoder_dtype``
    defaults to ``checkpoint`` and preserves the stored mixed precision;
    ``float32`` is also supported. ``bfloat16`` preserves any checkpoint FP32
    tensors while converting other floating encoder weights to BF16. The frontend
    and CTC adapter always run in FP32. ``alignment`` overrides the supported
    parameters of NeMo's existing CTC word aligner. ``measure_timings`` enables
    synchronized wall-clock measurements in seconds. ``max_batch_size`` (8) and
    ``max_batch_duration`` (120 padded audio seconds) bound acoustic microbatches;
    an individual longer turn retains its full encoder context.
    Attention uses PyTorch SDPA with its existing weights, positions, and masks,
    avoiding per-shape FlexAttention compilation during variable-length training.
    The first word starts at zero. Each word ends at the following CTC word
    onset, and the final word ends at the exact input audio duration.
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = _plain_config(config)
        components = _load_teacher_components(self.config)
        self.preprocessor = components.preprocessor
        self.encoder = components.encoder
        self.ctc_decoder = components.decoder
        self.tokenizer = components.tokenizer
        self.normalize_type = components.normalize_type
        self.teacher_info = components.info
        self._alignment_options = {
            **components.alignment_options,
            'refine_word_timestamps': False,
            **dict(self.config.get('alignment', {}) or {}),
        }
        self.teacher_info['refine_word_timestamps'] = self._alignment_options['refine_word_timestamps']
        self.teacher_info['word_interval_policy'] = 'zero_then_ctc_onsets_to_audio_end'
        self._aligners = {}
        self.measure_timings = bool(self.config.get('measure_timings', False))
        self.max_batch_size = int(self.config.get('max_batch_size', 8))
        self.max_batch_duration = float(self.config.get('max_batch_duration', 120.0))
        self.use_cuda_alignment = bool(self.config.get('use_cuda_alignment', True))
        self.group_languages_by_policy = bool(self.config.get('group_languages_by_policy', True))
        self.ctc_max_batch_elements = int(self.config.get('ctc_max_batch_elements', 128 * 2**20))
        if self.max_batch_size < 1 or not 0 < self.max_batch_duration < float('inf'):
            raise ValueError('max_batch_size and max_batch_duration must be positive finite values.')
        if self.ctc_max_batch_elements < 1:
            raise ValueError('ctc_max_batch_elements must be positive.')
        self.teacher_info.update(
            use_cuda_alignment=self.use_cuda_alignment,
            group_languages_by_policy=self.group_languages_by_policy,
            ctc_max_batch_elements=self.ctc_max_batch_elements,
        )
        self.last_timings = {}
        self.register_buffer('_device_anchor', torch.empty(0), persistent=False)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True):
        """Keep every teacher component in evaluation mode, including after parent.train()."""
        return super().train(False)

    def _apply(self, fn, recurse=True):
        """Allow device moves while preserving the teacher's own precision policy.

        Probe an empty tensor before conversion: converting a real FP32 tensor to
        BF16 and back would already have destroyed the original FP32 values.
        """

        def preserve_dtype(tensor):
            if not tensor.is_floating_point() and not tensor.is_complex():
                return fn(tensor)
            probe = fn(tensor.new_empty(0))
            if probe.dtype != tensor.dtype:
                return tensor.to(device=probe.device)
            return fn(tensor)

        return super()._apply(preserve_dtype, recurse=recurse)

    @torch.inference_mode()
    @fp32_precision()
    def forward(
        self,
        audio: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int,
        texts: Sequence[str],
        languages: Sequence[str | None] | None = None,
        *,
        return_ctc: bool = False,
        align_words: bool = True,
        distillation_topk: int = 32,
    ) -> list[dict[str, Any]]:
        """Return one ``{words: [{word, start, end}], error}`` record per turn.

        ``audio`` is mono audio shaped ``(B, samples)``; ``lengths`` excludes padding.
        Resampling trims each turn before converting it to FP32 16 kHz audio.
        Word times are relative to each input turn, in seconds. ``ctc_start`` and
        ``ctc_end`` retain the native CTC emission bounds for inspection. Invalid transcripts
        are reported independently, so one failure does not remove other records.
        ``return_ctc`` additionally returns sparse teacher probabilities and transcript
        IDs in ``ctc``. Set ``align_words=False`` to skip word dynamic programming;
        the encoder is still run only once per acoustic microbatch. ``distillation_topk``
        counts retained nonblank classes, in addition to explicit blank and tail mass.
        """
        _validate_batch(audio, lengths, sample_rate, texts, languages)
        if not align_words and not return_ctc:
            raise ValueError('Request word alignment or CTC distillation targets.')
        if isinstance(distillation_topk, bool) or not isinstance(distillation_topk, int) or distillation_topk < 1:
            raise ValueError('distillation_topk must be a positive integer.')
        self.eval()
        device = audio.device
        started = _timestamp(device, self.measure_timings)
        source_lengths = lengths.detach().cpu().tolist()
        batches = _turn_batches(source_lengths, sample_rate, self.max_batch_size, self.max_batch_duration)
        if len(batches) > 1:
            results, timings = [], defaultdict(float)
            for start, end in batches:
                results.extend(
                    self(
                        audio[start:end],
                        lengths[start:end],
                        sample_rate,
                        texts[start:end],
                        languages[start:end] if languages is not None else None,
                        return_ctc=return_ctc,
                        align_words=align_words,
                        distillation_topk=distillation_topk,
                    )
                )
                for key, value in self.last_timings.items():
                    timings[key] += value
            timings['total'] = _timestamp(device, self.measure_timings) - started
            self.last_timings = dict(timings)
            if self.measure_timings:
                for result in results:
                    result['timings'] = dict(timings)
            return results
        if self._device_anchor.device != device:
            self.to(device=device)
        languages = list(languages) if languages is not None else [None] * audio.shape[0]
        results = [{'words': [], 'error': None} for _ in texts]
        frontend = getattr(self.preprocessor, 'featurizer', None)
        expected_feature_lengths = None
        if hasattr(frontend, 'get_seq_len'):
            resampled_lengths = [(length * 16000 + sample_rate - 1) // sample_rate for length in source_lengths]
            expected_feature_lengths = frontend.get_seq_len(torch.tensor(resampled_lengths, device='cpu')).tolist()
        valid = []
        for index, (text, length) in enumerate(zip(texts, source_lengths)):
            if not isinstance(text, str) or not text.strip():
                results[index]['error'] = 'empty_transcript'
            elif length <= 0 or length > audio.shape[1]:
                results[index]['error'] = 'invalid_audio_length'
            elif expected_feature_lengths is not None and expected_feature_lengths[index] < 2:
                results[index]['error'] = 'audio_too_short'
            else:
                valid.append(index)
        if not valid:
            self.last_timings = {'total': _timestamp(device, self.measure_timings) - started}
            if self.measure_timings:
                for result in results:
                    result['timings'] = dict(self.last_timings)
            return results

        waveforms, waveform_lengths = _resample_turns(audio, source_lengths, valid, int(sample_rate))
        after_resample = _timestamp(device, self.measure_timings)
        with torch.autocast(device_type=device.type, enabled=False):
            features, feature_lengths = self.preprocessor(input_signal=waveforms, length=waveform_lengths)
            retained = [row for row, length in enumerate(feature_lengths.detach().cpu().tolist()) if length >= 2]
            if len(retained) != len(valid):
                for row, index in enumerate(valid):
                    if row not in retained:
                        results[index]['error'] = 'audio_too_short'
                valid = [valid[row] for row in retained]
                if not valid:
                    self.last_timings = {'total': _timestamp(device, self.measure_timings) - started}
                    if self.measure_timings:
                        for result in results:
                            result['timings'] = dict(self.last_timings)
                    return results
                selected = torch.tensor(retained, device=features.device)
                features = features.index_select(0, selected)
                feature_lengths = feature_lengths.index_select(0, selected)
            features = _normalize_features(features.float(), feature_lengths, self.normalize_type)
        after_preprocess = _timestamp(device, self.measure_timings)
        parameter = next(self.encoder.parameters(), None)
        encoder_dtype = parameter.dtype if parameter is not None else torch.float32
        # NeMo's ParallelExpertEncoderPT matches the ASR branch's input dtype.
        features = features.to(dtype=encoder_dtype)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=encoder_dtype == torch.bfloat16):
            states, state_lengths = self.encoder(audio_signal=features, length=feature_lengths)
        after_encoder = _timestamp(device, self.measure_timings)

        groups = defaultdict(list)
        for row, index in enumerate(valid):
            language = _timestamp_language(languages[index])
            if self.group_languages_by_policy and language not in (None, 'zh', 'ja', 'th'):
                language = 'whitespace'
            groups[language].append((row, index))
        alignment_device = device if self.use_cuda_alignment and device.type == 'cuda' else None
        prepare_seconds = align_seconds = 0.0
        # The existing deferred-head API only retains transcript token columns on
        # the host, avoiding a full B x T x vocabulary tensor in device memory.
        with torch.autocast(device_type=device.type, enabled=False):
            for language, rows in groups.items():
                aligner = self._get_aligner(language)
                indices = [index for _, index in rows]
                selected = torch.tensor([row for row, _ in rows], device=device)
                inputs = _timestamp_inputs(states.index_select(0, selected), state_lengths.index_select(0, selected))
                durations = [source_lengths[index] / sample_rate for index in indices]
                transcripts = [texts[index] for index in indices]
                before_prepare = _timestamp(device, self.measure_timings)
                if return_ctc:
                    ctc_records = _ctc_distillation_targets(aligner, inputs, transcripts, durations, distillation_topk)
                    for index, record in zip(indices, ctc_records):
                        results[index].update(record)
                if not align_words:
                    prepare_seconds += _timestamp(device, self.measure_timings) - before_prepare
                    continue
                try:
                    prepared = aligner.prepare_from_inputs(inputs, transcripts, durations)
                except (ValueError, UnicodeError):
                    # Retry independently only when a transcript invalidates the
                    # shared preparation (e.g. unknown pieces or too few frames).
                    for offset, index in enumerate(indices):
                        single = _timestamp_inputs(
                            inputs.asr_encoded[offset : offset + 1], inputs.asr_encoded_lengths[offset : offset + 1]
                        )
                        try:
                            prepared_one = aligner.prepare_from_inputs(single, [texts[index]], [durations[offset]])
                            results[index].update(
                                _checked_alignment_results(
                                    aligner, prepared_one, [durations[offset]], device=alignment_device
                                )[0]
                            )
                        except (ValueError, UnicodeError) as error:
                            results[index]['error'] = f'{type(error).__name__}: {error}'
                    prepare_seconds += _timestamp(device, self.measure_timings) - before_prepare
                    continue
                after_prepare = _timestamp(device, self.measure_timings)
                aligned = _checked_alignment_results(aligner, prepared, durations, device=alignment_device)
                after_align = _timestamp(device, self.measure_timings)
                prepare_seconds += after_prepare - before_prepare
                align_seconds += after_align - after_prepare
                for index, result in zip(indices, aligned):
                    results[index].update(result)

        self.last_timings = {
            'resample': after_resample - started,
            'preprocess': after_preprocess - after_resample,
            'encoder': after_encoder - after_preprocess,
            'ctc_prepare': prepare_seconds,
            'alignment': align_seconds,
            'total': _timestamp(device, self.measure_timings) - started,
        }
        if self.measure_timings:
            for result in results:
                result['timings'] = dict(self.last_timings)
        return results

    def _get_aligner(self, language):
        if language not in self._aligners:
            options = dict(self._alignment_options)
            if language is not None:
                options['timestamp_language'] = language
            aligner = _make_aligner(self.ctc_decoder, self.tokenizer, options)
            aligner._CTC_WINDOW_MAX_ELEMENTS = self.ctc_max_batch_elements
            self._aligners[language] = aligner
        return self._aligners[language]


@dataclass
class _TeacherComponents:
    preprocessor: nn.Module
    encoder: nn.Module
    decoder: nn.Module
    tokenizer: Any
    normalize_type: str | None
    alignment_options: dict[str, Any]
    info: dict[str, Any]


class _CTCSDPAAttention(nn.Module):
    """Run the existing CTC attention weights with public PyTorch SDPA kernels."""

    def __init__(self, attention):
        super().__init__()
        if getattr(attention, '_uses_rel_pos', False):
            raise ValueError('The CTC SDPA adapter supports RoPE and absolute-position attention.')
        self.attention = attention

    def forward(self, x, block_mask=None, pos_emb=None):
        attention = self.attention
        batch, frames, _ = x.shape
        heads, width = attention.n_heads, attention.head_dim
        positions = torch.arange(frames, device=x.device).repeat(batch) if attention._uses_rope else None
        q, k, v = attention._project_sequence_packed_qkv(
            x.reshape(batch * frames, attention.d_model), position_ids=positions, fused_qkv=True
        )
        q, k, v = (projection.view(batch, frames, heads, width).transpose(1, 2) for projection in (q, k, v))
        mask = None
        if block_mask is not None:
            # Reuse the encoder's exact validity/causal mask. For the timestamp
            # adapter this broadcasts to (B, 1, T, T), independent of head count.
            mask = block_mask.mask_mod(
                torch.arange(batch, device=x.device).view(batch, 1, 1, 1),
                torch.arange(heads, device=x.device).view(1, heads, 1, 1),
                torch.arange(frames, device=x.device).view(1, 1, frames, 1),
                torch.arange(frames, device=x.device).view(1, 1, 1, frames),
            )
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        output = output.transpose(1, 2).contiguous().view(batch, frames, attention.d_model)
        return attention.out_proj(output)


def _load_teacher_components(config):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
        MultiSpeakerSOTWordTimestampAligner,
        load_ctc_timestamp_artifact,
    )

    snapshot = Path(config.get('snapshot_path', '')).expanduser().resolve()
    if not config.get('snapshot_path') or not (snapshot / 'config.json').is_file():
        raise ValueError('snapshot_path must name a local Nemotron ASR snapshot containing config.json.')
    checkpoint_config = json.loads((snapshot / 'config.json').read_text(encoding='utf-8'))
    perception = checkpoint_config['perception']
    expert = checkpoint_config.get('pe_encoder_config') or {}
    encoder_config = dict(expert.get('asr_encoder_cfg') or perception['encoder'])
    preprocessor_config = dict(perception['preprocessor'])
    # Restrict construction to these two acoustic components. In particular, do
    # not instantiate perception, pe_encoder_config, SALM, or any HF AutoModel.
    if encoder_config.get('_target_') != 'nemo.collections.asr.modules.transformer_encoder.TransformerEncoder':
        raise ValueError('The timestamp teacher requires the Nemotron Transformer ASR encoder config.')
    if preprocessor_config.get('_target_') != 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor':
        raise ValueError('The timestamp teacher requires the Nemotron mel preprocessor config.')
    if int(preprocessor_config.get('sample_rate', 16000)) != 16000:
        raise ValueError('The Nemotron timestamp teacher requires a 16 kHz preprocessor.')
    encoder_config['sync_max_audio_length'] = False
    preprocessor_config['dither'] = 0.0
    dtype_policy = str(config.get('encoder_dtype', 'checkpoint'))
    if dtype_policy not in ('checkpoint', 'float32', 'bfloat16'):
        raise ValueError('encoder_dtype must be checkpoint, float32, or bfloat16.')
    weights = snapshot / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'The local Nemotron ASR snapshot has no {weights.name}.')
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device('cpu'):
            preprocessor = instantiate(OmegaConf.create(preprocessor_config)).float()
            encoder = instantiate(OmegaConf.create(encoder_config))
            _load_selected_weights(preprocessor, weights, 'perception.preprocessor.', 'float32')
            _load_selected_weights(encoder, weights, 'perception.encoder.asr_encoder.', dtype_policy)
            artifact_path = Path(config.get('ctc_artifact_path') or weights).expanduser()
            artifact = load_ctc_timestamp_artifact(artifact_path, map_location='cpu', dtype=torch.float32)
    finally:
        torch.set_default_dtype(previous_dtype)
    for transformer in (encoder, artifact.decoder.transformer):
        for layer in transformer.layers:
            layer.attn = _CTCSDPAAttention(layer.attn)
    for module in (preprocessor, encoder, artifact.decoder):
        module.eval().requires_grad_(False)
        for child in module.modules():
            if hasattr(child, 'sync_max_audio_length'):
                child.sync_max_audio_length = False
    parameters = checkpoint_config.get('ctc_timestamps', {}).get('alignment_config', {}).get('parameters', {})
    accepted = set(inspect.signature(MultiSpeakerSOTWordTimestampAligner.__init__).parameters)
    alignment_options = {key: value for key, value in parameters.items() if key in accepted}
    for name in ('online_inference_length', 'chunk_left_context', 'chunk_right_context'):
        if name in config:
            alignment_options[name] = int(config[name])
    return _TeacherComponents(
        preprocessor=preprocessor,
        encoder=encoder,
        decoder=artifact.decoder,
        tokenizer=artifact.tokenizer,
        normalize_type=expert.get('asr_normalize_type', None),
        alignment_options=alignment_options,
        info={
            'snapshot_path': str(snapshot),
            'encoder_dtype': dtype_policy,
            'asr_encoder_prefix': 'perception.encoder.asr_encoder.',
            'ctc_artifact_path': str(artifact_path),
            'ctc_attention_backend': 'sdpa',
            'encoder_attention_backend': 'sdpa',
        },
    )


def _ctc_distillation_targets(aligner, inputs, texts, durations, topk):
    """Retain ordered sparse categorical targets from the existing deferred head.

    Every row keeps the strongest nonblank classes, the blank class, and the
    probability mass of all remaining classes. This defines a coarsened teacher
    distribution without retaining a full time-by-vocabulary tensor. Transcript
    tokenization reuses the forced aligner's language and text-cleaning policy.
    """
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _timestamp_units

    num_classes = int(aligner.ctc_decoder.num_classes_with_blank)
    upsampling = int(getattr(aligner.ctc_decoder, 'upsampling_factor', 1))
    if upsampling != 2:
        raise ValueError('CTC distillation requires two teacher subframes per 80 ms encoder frame.')
    blank_id = aligner._resolve_blank_id(num_classes)
    lengths = (inputs.asr_encoded_lengths * upsampling).detach().cpu().tolist()
    count = min(topk, num_classes - 1)
    targets, chunks = [], [[] for _ in texts]
    for text, length, duration in zip(texts, lengths, durations):
        record = {'error': None}
        try:
            words = aligner._tokenize_words(
                _timestamp_units(aligner.parse_sot_words(text), aligner.timestamp_language or 'auto'), blank_id
            )
            token_ids = [token for word in words for token in word['token_ids']]
            if not token_ids:
                raise ValueError('The transcript has no CTC target tokens.')
            minimum = len(token_ids) + sum(left == right for left, right in zip(token_ids, token_ids[1:]))
            if minimum > length:
                raise ValueError('The transcript is too long for the CTC timeline.')
            with torch.inference_mode(False):
                record['ctc'] = {
                    'token_ids': torch.tensor(token_ids, dtype=torch.long),
                    'num_classes': num_classes,
                    'blank_id': blank_id,
                    'frame_seconds': 0.08 / upsampling,
                    'audio_duration': duration,
                }
        except (ValueError, UnicodeError) as error:
            record['error'] = f'{type(error).__name__}: {error}'
        targets.append(record)
    for batch_start, start, log_probs in aligner._iter_ctc_windows(inputs):
        if log_probs.shape[-1] != num_classes:
            raise ValueError('The teacher CTC vocabulary changed during distillation.')
        values, indices = log_probs.float().topk(count + 1, dim=-1)
        # Rank only this small candidate set, preserving descending nonblank order.
        order = torch.arange(count + 1, device=indices.device).expand_as(indices)
        order = order.masked_fill(indices == blank_id, count + 1).argsort(dim=-1)[..., :count]
        indices = indices.gather(-1, order)
        values = values.gather(-1, order)
        indices = torch.cat((indices, torch.full_like(indices[..., :1], blank_id)), dim=-1)
        probabilities = torch.cat((values.exp(), log_probs[..., blank_id : blank_id + 1].float().exp()), dim=-1)
        tail = (1 - probabilities.sum(dim=-1)).clamp_min(0)
        for offset in range(log_probs.size(0)):
            row = batch_start + offset
            length = min(log_probs.size(1), lengths[row] - start)
            if length > 0 and 'ctc' in targets[row]:
                chunks[row].append(
                    tuple(_ordinary_cpu(value[offset, :length]) for value in (indices, probabilities, tail))
                )
    with torch.inference_mode(False):
        for record, pieces in zip(targets, chunks):
            if 'ctc' not in record:
                continue
            if not pieces:
                record.pop('ctc')
                record['error'] = 'empty_ctc_timeline'
                continue
            for name, values in zip(('indices', 'probabilities', 'tail_probabilities'), zip(*pieces)):
                record['ctc'][name] = torch.cat(values)
            probabilities = record['ctc']['probabilities']
            if (
                not torch.isfinite(probabilities).all()
                or not torch.isfinite(record['ctc']['tail_probabilities']).all()
            ):
                record.pop('ctc')
                record['error'] = 'invalid_ctc_probabilities'
    return targets


def _ordinary_cpu(tensor):
    with torch.inference_mode(False):
        return tensor.detach().to(device='cpu').clone()


def _load_selected_weights(module, path, prefix, dtype_policy):
    """Read only this acoustic component's tensors, retaining checkpoint FP32 values."""
    from safetensors import safe_open

    with safe_open(str(path), framework='pt', device='cpu') as handle:
        available = {key[len(prefix) :] for key in handle.keys() if key.startswith(prefix)}
        expected = set(module.state_dict())
        if available != expected:
            raise ValueError(
                f'Checkpoint component {prefix!r} does not match its configuration: '
                f'missing={sorted(expected - available)}, unexpected={sorted(available - expected)}.'
            )
        state = {}
        for name in sorted(expected):
            tensor = handle.get_tensor(prefix + name)
            if tensor.is_floating_point():
                if dtype_policy == 'float32':
                    tensor = tensor.float()
                elif dtype_policy == 'bfloat16' and tensor.dtype != torch.float32:
                    tensor = tensor.bfloat16()
            state[name] = tensor
    module.load_state_dict(state, strict=True, assign=True)


def _make_aligner(decoder, tokenizer, options):
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import MultiSpeakerSOTWordTimestampAligner

    return MultiSpeakerSOTWordTimestampAligner(ctc_decoder=decoder, tokenizer=tokenizer, **options)


def _normalize_features(features, lengths, normalize_type):
    if normalize_type:
        from nemo.collections.asr.parts.preprocessing.features import normalize_batch

        features, _, _ = normalize_batch(features, lengths, normalize_type=normalize_type)
    return features


def _resample_turns(audio, source_lengths, indices, sample_rate):
    if sample_rate != 16000:
        from torchaudio.functional import resample

    waveforms = []
    for index in indices:
        waveform = audio[index, : source_lengths[index]].detach().float()
        if sample_rate != 16000:
            waveform = resample(waveform, orig_freq=sample_rate, new_freq=16000)
        waveforms.append(waveform)
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], device=audio.device, dtype=torch.long)
    return pad_sequence(waveforms, batch_first=True), lengths


def _timestamp_inputs(states, lengths):
    return SimpleNamespace(
        asr_encoded=states,
        asr_encoded_lengths=lengths,
        sortformer_sigmoids=None,
        sortformer_lengths=lengths,
        diarization_labels=None,
        diarization_lengths=None,
        diarization_frame_seconds=0.01,
    )


def _turn_batches(lengths, sample_rate, max_batch_size, max_batch_duration):
    batches = []
    start = 0
    maximum = 0
    for end, length in enumerate(lengths):
        candidate = max(maximum, length)
        if end > start and (
            end - start >= max_batch_size or (end - start + 1) * candidate / sample_rate > max_batch_duration
        ):
            batches.append((start, end))
            start, maximum = end, 0
        maximum = max(maximum, length)
    if start < len(lengths):
        batches.append((start, len(lengths)))
    return batches


def _checked_alignment_results(aligner, prepared, audio_durations, *, device=None):
    aligned = aligner.align_prepared(prepared) if device is None else aligner.align_prepared(prepared, device=device)
    records = prepared['records']
    if len(aligned) != len(records):
        raise RuntimeError(f'CTC alignment returned {len(aligned)} records for {len(records)} input turns.')
    if len(audio_durations) != len(records):
        raise RuntimeError('CTC alignment requires one exact audio duration per input turn.')
    return [
        _public_result(result, duration, expected_words=sum(len(stream['words']) for stream in record['streams']))
        for result, record, duration in zip(aligned, records, audio_durations)
    ]


def _public_result(result, audio_duration, expected_words=None):
    """Partition the whole turn from zero using subsequent CTC onsets as word ends."""
    words = [dict(word) for group in result.get('speaker_word_timestamps', {}).values() for word in group]
    failures = result.get('alignment_diagnostics', {}).get('failed_streams', [])
    error = 'no_legal_ctc_path' if failures else ('no_aligned_words' if not words else None)
    if error is None and expected_words is not None and len(words) != expected_words:
        error = 'incomplete_alignment'
    if error is None:
        try:
            audio_duration = float(audio_duration)
            starts = [float(word['start']) for word in words]
            ends = [float(word['end']) for word in words]
        except (KeyError, TypeError, ValueError, OverflowError):
            error = 'invalid_word_timestamps'
        else:
            tolerance = max(1e-8, 4 * math.ulp(audio_duration)) if math.isfinite(audio_duration) else 0.0
            valid = (
                math.isfinite(audio_duration)
                and audio_duration > 0
                and all(math.isfinite(start) and 0 <= start < audio_duration for start in starts)
                and all(right > left for left, right in zip(starts, starts[1:]))
                and all(
                    math.isfinite(end) and start < end <= audio_duration + tolerance
                    for start, end in zip(starts, ends)
                )
            )
            if valid:
                for index, (word, ctc_start, ctc_end, end) in enumerate(
                    zip(words, starts, ends, [*starts[1:], audio_duration])
                ):
                    word.update(start=0.0 if index == 0 else ctc_start, end=end, ctc_start=ctc_start, ctc_end=ctc_end)
            else:
                error = 'invalid_word_timestamps'
        if error is not None:
            words = []
    return {'words': words, 'error': error, 'alignment': result}


def _timestamp_language(language):
    if language is None:
        return None
    language = str(language).lower().replace('_', '-').split('-')[0]
    if language in ('', 'und', 'unknown', 'auto'):
        return None
    return {
        'zho': 'zh',
        'cmn': 'zh',
        'chinese': 'zh',
        'jpn': 'ja',
        'japanese': 'ja',
        'tha': 'th',
        'thai': 'th',
        'kor': 'ko',
        'korean': 'ko',
    }.get(language, language)


def _timestamp(device, synchronize):
    if synchronize and device.type == 'cuda':
        torch.cuda.synchronize(device)
    return perf_counter()


def _plain_config(config):
    if not isinstance(config, Mapping):
        raise TypeError('The Nemotron timestamp teacher configuration must be a mapping.')
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)


def _validate_batch(audio, lengths, sample_rate, texts, languages):
    if audio.ndim != 2 or not audio.is_floating_point():
        raise ValueError('audio must be floating-point mono audio shaped (batch, samples).')
    if lengths.ndim != 1 or lengths.shape[0] != audio.shape[0] or lengths.is_floating_point():
        raise ValueError('lengths must be integer sample counts shaped (batch,).')
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError('sample_rate must be a positive integer.')
    if isinstance(texts, str) or len(texts) != audio.shape[0]:
        raise ValueError('texts must contain one transcript per audio turn.')
    if languages is not None and (isinstance(languages, str) or len(languages) != audio.shape[0]):
        raise ValueError('languages must contain one language per audio turn.')
