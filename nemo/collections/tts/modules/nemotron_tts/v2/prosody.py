# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learnable low-rate target-audio conditioning for EasyMagpie."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer, MultiResolutionSTFTEncoder
from nemo.collections.tts.parts.utils.helpers import binarize_attention_parallel


@dataclass
class ProsodyEncoding:
    quantized: torch.Tensor  # (B, FSQ dimensions, text slots), with straight-through gradients
    codes: torch.Tensor  # (B, text slots), compact single-turn or sparse restored multiturn codes
    code_lens: torch.Tensor  # (B,) number of real codes, excluding timing PAD
    alignment: torch.Tensor | tuple[torch.Tensor, ...]  # dense single-turn or unpadded per-turn MAS paths
    audio_lens: torch.Tensor
    alignment_loss: torch.Tensor
    turn_indices: torch.Tensor | None = None  # (turns, 3): conversation, start, end on stacked codec grid
    turn_code_lens: torch.Tensor | None = None


def compact_conditioning_turns(model, text, text_lens, audio_lens, is_multiturn, include_eos=False):
    """Recover compact transcripts, original token positions, and acoustic spans.

    Adapted from Speech PR #16519, commit bcf3a4e5, ``compact_conditioning_turns``.
    BOS marks a turn start, text EOS ends its transcript, and interruption marks
    its exclusive acoustic end. A crop may omit the final interruption marker.
    Prosody keeps text EOS explicitly; attention alignment in the reference PR
    excludes it. Timing PAD and interruption tokens never become prosody codes.
    """
    ends = torch.div(audio_lens + model.frame_stacking_factor - 1, model.frame_stacking_factor, rounding_mode='floor')
    rows, indices, positions = [], [], []
    interruption_id = getattr(model, 'interruption_token_id', -1)
    special = {model.pad_id, model.bos_id, model.eos_id, interruption_id}
    source = text.detach().cpu().tolist()
    lengths = text_lens.detach().cpu().tolist()
    acoustic_ends = ends.detach().cpu().tolist()
    for sample, (values, length, acoustic_end) in enumerate(zip(source, lengths, acoustic_ends)):
        values = values[:length]
        if not is_multiturn:
            keys = [
                i
                for i, token in enumerate(values)
                if (token not in special and 0 <= token < model.text_vocab_size)
                or (include_eos and token == model.eos_id)
            ]
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
            elif token == interruption_id:
                append_turn(position)
                start, ids, keys, ended_text = None, [], [], False
            elif token == model.eos_id:
                if include_eos and ids and not ended_text:
                    ids.append(token)
                    keys.append(position)
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


class ProsodyChannel(nn.Module):
    """Pool a 3.125 Hz audio encoder into one FSQ code per text token using MAS.

    The default encoder adds two stride-2 blocks to the 16 kHz semantic codec
    (320 * 2**4 = 5120 samples/frame). A duration-normalized monotonic path pools
    continuous audio latents onto the text grid before quantization. If text is
    longer, MAS runs along the text axis and adjacent tokens can share a latent.
    Text embeddings are detached only on the alignment branch. FSQ vectors are projected into backbone
    embeddings, rather than looking up integer indices, to preserve the quantizer's
    straight-through gradient to the encoder. At inference, decoding predicted IDs
    through the same projection gives exactly the same embeddings.

    PAD is predicted alongside the 2048 codes; MASK is an input-only learned dropout
    embedding. Neither special embedding is frozen or forced to zero.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        input_sample_rate: int,
        dropout: float = 0.5,
        loss_scale: float = 1.0,
        lead_steps: int = 1,
        encoder_channels: int = 512,
        latent_dim: int = 128,
        alignment_dim: int = 64,
        alignment_temperature: float = 0.1,
        alignment_loss_scale: float = 0.1,
    ):
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("prosody.dropout must be in [0, 1]")
        if lead_steps < 0:
            raise ValueError("prosody.lead_steps must be nonnegative")
        if latent_dim < 2 or alignment_dim < 1 or alignment_temperature <= 0 or alignment_loss_scale < 0:
            raise ValueError("Prosody latent/alignment dimensions and temperature must be positive; loss scale >= 0")
        self.input_sample_rate = input_sample_rate
        self.sample_rate = 16000
        self.samples_per_frame = 5120
        self.dropout = dropout
        self.loss_scale = loss_scale
        self.lead_steps = lead_steps
        self.alignment_temperature = alignment_temperature
        self.alignment_loss_scale = alignment_loss_scale
        self.quantizer = FiniteScalarQuantizer(num_levels=[8, 8, 8, 4])
        self.vocab_size = self.quantizer.codebook_size
        self.pad_id = self.vocab_size
        self.mask_id = self.vocab_size + 1
        self.encoder = MultiResolutionSTFTEncoder(
            out_dim=latent_dim,
            resolutions=[[1280, 320, 1280]],
            resolution_filter_list=[encoder_channels],
            down_sample_filter_list=[encoder_channels] * 4,
            pad_mode='replicate',
        )
        self.audio_alignment_projection = nn.Linear(latent_dim, alignment_dim)
        self.text_alignment_projection = nn.Linear(embedding_dim, alignment_dim)
        self.alignment_loss_fn = ForwardSumLoss()
        self.quantizer_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, self.quantizer.dim))
        self.input_projection = nn.Linear(self.quantizer.dim, embedding_dim)
        self.special_embeddings = nn.Embedding(2, embedding_dim)
        self.output_projection = nn.Linear(hidden_dim, self.vocab_size + 1)

    def encode_audio(self, audio, audio_lens):
        """Encode waveforms into continuous 3.125 Hz latents, without quantizing."""
        audio = resample(audio, self.input_sample_rate, self.sample_rate)
        # Validation collators may supply int32 lengths. Promote before the
        # sample-rate multiplication; casting its result is too late to avoid overflow.
        audio_lens = torch.div(audio_lens.long() * self.sample_rate, self.input_sample_rate, rounding_mode='floor')
        # Codec downsampling uses floor lengths. Pad each utterance to retain its
        # final partial prosody frame, including utterances shorter than one frame.
        frame_lens = torch.div(audio_lens + self.samples_per_frame - 1, self.samples_per_frame, rounding_mode='floor')
        padded_lens = frame_lens * self.samples_per_frame
        positions = torch.arange(audio.size(1), device=audio.device)
        audio = audio * (positions[None, :] < audio_lens[:, None])
        audio = F.pad(audio, (0, max(0, int(padded_lens.max()) - audio.size(1))))
        # Spectrograms and alignment need FP32 even under mixed precision.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            latents, encoded_lens = self.encoder(audio=audio.float(), audio_len=padded_lens)
        return latents, encoded_lens

    def align_and_pool(self, latents, audio_lens, text_embeddings, text_lens):
        """Return text-grid latents, a hard path, and differentiable forward-sum loss.

        Search is discrete. The loss trains both alignment projections and the audio
        encoder, while the pooled latent path preserves TTS gradients to the encoder.
        No alignment gradient reaches the original text embedding/CAS parameters.
        """
        if latents.ndim != 3 or text_embeddings.ndim != 3 or latents.size(0) != text_embeddings.size(0):
            raise ValueError("Prosody audio/text features must have matching batches and three dimensions")
        batch_size, _, audio_steps = latents.shape
        text_steps = text_embeddings.size(1)
        if audio_lens.shape != (batch_size,) or text_lens.shape != (batch_size,):
            raise ValueError("Prosody audio/text lengths must have shape (batch_size,)")
        if ((audio_lens < 1) | (audio_lens > audio_steps)).any() or ((text_lens < 0) | (text_lens > text_steps)).any():
            raise ValueError("Prosody lengths must be in range and audio must have at least one frame")
        with torch.autocast(device_type=latents.device.type, enabled=False):
            queries = F.normalize(self.audio_alignment_projection(latents.float().transpose(1, 2)), dim=-1)
            keys = F.normalize(self.text_alignment_projection(text_embeddings.detach().float()), dim=-1)
            scores = torch.bmm(queries, keys.transpose(1, 2)) / self.alignment_temperature
            path, alignment_loss = self._align(scores, audio_lens, text_lens)
            weights = path / path.sum(dim=-1, keepdim=True).clamp_min(1)
            pooled = torch.bmm(latents.float(), weights.transpose(1, 2))
        return pooled, path, alignment_loss

    def encode(self, audio, audio_lens, text_embeddings, text_lens):
        """Align continuous target-audio latents to text, then quantize on the text grid."""
        latents, encoded_lens = self.encode_audio(audio, audio_lens)
        pooled, path, alignment_loss = self.align_and_pool(latents, encoded_lens, text_embeddings, text_lens)
        with torch.autocast(device_type=audio.device.type, enabled=False):
            projected = self.quantizer_projection(pooled.transpose(1, 2)).transpose(1, 2)
            quantized, indices = self.quantizer(inputs=projected, input_len=text_lens)
        valid = torch.arange(text_embeddings.size(1), device=audio.device)[None, :] < text_lens[:, None]
        codes = torch.where(valid, indices[0].long(), self.pad_id)
        return ProsodyEncoding(quantized, codes, text_lens, path, encoded_lens, alignment_loss)

    def encode_turns(
        self,
        audio,
        audio_lens,
        text_tokens,
        text_lens,
        turn_indices,
        key_positions,
        embed_text,
        samples_per_frame,
        timeline_lens,
        max_turns_per_batch=32,
        max_frames_per_batch=8192,
    ):
        """Encode independent turns and scatter one code per token onto the timeline.

        Turn extraction, length sorting, padded-frame budgets, and restoration
        follow PR #16519's ``align_text_turns``. Here waveforms are cropped *before*
        the trainable encoder, and MAS supports either sequence being longer.
        Frame budgets use the original stacked codec grid, matching the PR's
        ``aligned_text_max_*_per_batch`` settings. Paths remain unpadded per turn
        rather than allocating a conversation-wide quadratic alignment tensor.
        """
        if max_turns_per_batch < 1 or max_frames_per_batch < 1 or samples_per_frame < 1:
            raise ValueError('Turn/frame batch limits and samples_per_frame must be positive')
        num_turns = text_tokens.size(0)
        if (
            audio.ndim != 2
            or audio_lens.shape != (audio.size(0),)
            or timeline_lens.shape != (audio.size(0),)
            or text_tokens.ndim != 2
            or text_lens.shape != (num_turns,)
            or turn_indices.shape != (num_turns, 3)
            or key_positions.shape != text_tokens.shape
        ):
            raise ValueError('Expected conversation waveforms, compact turn tokens/positions, and (turns,3) spans')
        batch_size = audio.size(0)
        width = int(timeline_lens.max())
        codes = text_tokens.new_full((batch_size, width), self.pad_id, dtype=torch.long)
        quantized = audio.new_zeros(batch_size, self.quantizer.dim, width, dtype=torch.float32)
        code_lens = text_lens.new_zeros(batch_size)
        turn_code_lens = text_lens.new_zeros(num_turns)
        encoded_lens = text_lens.new_zeros(num_turns)
        paths = [audio.new_zeros(0, 0) for _ in range(num_turns)]
        metadata = torch.cat([turn_indices, text_lens[:, None]], dim=1).detach().cpu().tolist()
        sample_lens = audio_lens.detach().cpu().tolist()
        valid_rows = []
        for row, (sample, start, end, num_text) in enumerate(metadata):
            if (
                0 <= sample < batch_size
                and 0 <= start < end
                and 0 < num_text <= text_tokens.size(1)
                and start * samples_per_frame < sample_lens[sample]
            ):
                if end - start > max_frames_per_batch:
                    raise ValueError(
                        f'Turn {row} has {end - start} frames, exceeding max_frames_per_batch={max_frames_per_batch}'
                    )
                valid_rows.append(row)
        previous_ends = {}
        for row in sorted(valid_rows, key=lambda index: metadata[index][:2]):
            sample, start, end, _ = metadata[row]
            if start < previous_ends.get(sample, 0):
                raise ValueError('Prosody turn spans must not overlap within a sample')
            previous_ends[sample] = end
        valid_rows.sort(key=lambda index: metadata[index][2] - metadata[index][1])
        batches = _turn_batches(valid_rows, metadata, max_turns_per_batch, max_frames_per_batch)
        alignment_loss = audio.new_zeros((), dtype=torch.float32)
        for batch_rows in batches:
            sample_spans = [
                (
                    metadata[row][0],
                    metadata[row][1] * samples_per_frame,
                    min(metadata[row][2] * samples_per_frame, sample_lens[metadata[row][0]]),
                )
                for row in batch_rows
            ]
            max_samples = max(end - start for _, start, end in sample_spans)
            packed_audio = audio.new_zeros(len(batch_rows), max_samples)
            for packed_row, (sample, start, end) in enumerate(sample_spans):
                packed_audio[packed_row, : end - start] = audio[sample, start:end]
            packed_audio_lens = audio_lens.new_tensor([end - start for _, start, end in sample_spans])
            row_indices = torch.tensor(batch_rows, device=text_tokens.device, dtype=torch.long)
            max_text = max(metadata[row][3] for row in batch_rows)
            packed_text = text_tokens.index_select(0, row_indices)[:, :max_text]
            packed_text_lens = text_lens.index_select(0, row_indices)
            with torch.no_grad():
                text_features = embed_text(packed_text, packed_text_lens).detach()
            encoded = self.encode(packed_audio, packed_audio_lens, text_features, packed_text_lens)
            # Forward-sum loss is a mean over turns, unlike the PR's per-frame NLL.
            alignment_loss = alignment_loss + encoded.alignment_loss * len(batch_rows)
            for packed_row, row in enumerate(batch_rows):
                sample, _, _, num_text = metadata[row]
                keys = key_positions[row, :num_text].long()
                codes[sample, keys] = encoded.codes[packed_row, :num_text]
                quantized[sample, :, keys] = encoded.quantized[packed_row, :, :num_text]
                code_lens[sample] += num_text
                turn_code_lens[row] = num_text
                encoded_lens[row] = encoded.audio_lens[packed_row]
                paths[row] = encoded.alignment[packed_row, :num_text, : int(encoded.audio_lens[packed_row])].clone()
        if valid_rows:
            alignment_loss = alignment_loss / len(valid_rows)
        else:
            # Empty/user-only local batches still participate in DDP backward.
            alignment_loss = sum((p.reshape(-1)[0] * 0.0 for p in self.parameters()), alignment_loss)
        return ProsodyEncoding(
            quantized, codes, code_lens, tuple(paths), encoded_lens, alignment_loss, turn_indices, turn_code_lens
        )

    def embed(self, codes, quantized=None, apply_dropout=False):
        """Embed (B, T) IDs, optionally using training's straight-through FSQ vectors."""
        if quantized is None:
            quantized = self.quantizer.decode(indices=codes.clamp(max=self.vocab_size - 1)[None].int())
        embedded = self.input_projection(quantized.transpose(1, 2))
        special = self.special_embeddings((codes - self.vocab_size).clamp(min=0, max=1))
        embedded = torch.where((codes >= self.vocab_size)[..., None], special, embedded)
        if apply_dropout:
            drop = (torch.rand_like(codes, dtype=torch.float) < self.dropout) & (codes < self.vocab_size)
            embedded = torch.where(drop[..., None], self.special_embeddings.weight[1], embedded)
        return embedded

    def teacher_force(
        self, quantized, codes, code_lens, target_lens, speech_delay, apply_dropout=False, timeline_lens=None
    ):
        """Build compact AR inputs/targets, preserving the learned PAD tail.

        The channel starts with text. The first real target is at
        ``speech_delay - lead_steps``; that PAD input predicts code 0 for the next step.
        ``timeline_lens`` selects sparse, restored multiturn codes instead of a
        compact single-turn sequence. There is one real code per text token.
        PAD targets continue through
        acoustic EOS, with at least one PAD target after the last delayed code.
        """
        prefix = speech_delay - self.lead_steps
        if prefix < 0:
            raise ValueError("speech_delay must be >= prosody.lead_steps")
        source_lens = code_lens if timeline_lens is None else timeline_lens
        sequence_lens = torch.maximum(target_lens + speech_delay, prefix + source_lens + 1)
        width = int(sequence_lens.max())
        if codes.size(1) == 0:
            targets = codes.new_full((codes.size(0), width), self.pad_id)
            embedded = self.embed(targets, quantized=quantized.new_zeros(codes.size(0), self.quantizer.dim, width))
            return embedded, targets, sequence_lens
        positions = torch.arange(width, device=codes.device)[None, :]
        source_positions = (positions - prefix).clamp(min=0, max=codes.size(1) - 1)
        source_positions = source_positions.expand(codes.size(0), -1)
        targets = codes.gather(1, source_positions)
        valid_code = (positions >= prefix) & (positions < prefix + source_lens[:, None])
        targets = torch.where(valid_code, targets, self.pad_id)
        inputs = F.pad(targets[:, :-1], (1, 0), value=self.pad_id)
        input_positions = F.pad(source_positions[:, :-1], (1, 0))
        input_quantized = quantized.gather(2, input_positions[:, None].expand(-1, quantized.size(1), -1))
        embedded = self.embed(inputs, quantized=input_quantized, apply_dropout=apply_dropout)
        return embedded, targets, sequence_lens

    def compute_loss(self, hidden, targets, lengths):
        """Supervise every position through acoustic EOS, including prosody PAD."""
        logits = self.output_projection(hidden)
        loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
        valid = torch.arange(targets.size(1), device=targets.device)[None, :] < lengths[:, None]
        return (loss * valid).sum() / valid.sum().clamp_min(1), logits

    def _align(self, scores, audio_lens, text_lens):
        """Run width-one MAS on whichever axis is longer, covering both sequences."""
        path = scores.new_zeros(scores.size(0), scores.size(2), scores.size(1))
        loss = scores.sum() * 0.0
        count = (text_lens > 0).sum().clamp_min(1)
        for text_is_longer in (False, True):
            rows = torch.where((text_lens > 0) & ((text_lens > audio_lens) == text_is_longer))[0]
            if rows.numel() == 0:
                continue
            group_scores = scores.index_select(0, rows)
            long_lens, short_lens = audio_lens[rows], text_lens[rows]
            if text_is_longer:
                group_scores = group_scores.transpose(1, 2)
                long_lens, short_lens = short_lens, long_lens
            loss = loss + self.alignment_loss_fn(
                attn_logprob=group_scores[:, None], in_lens=short_lens, out_lens=long_lens
            ) * (rows.numel() / count)
            with torch.no_grad():
                hard = torch.zeros_like(group_scores)
                single = short_lens == 1
                # The only valid path in a one-column problem is deterministic.
                # Legacy CPU MAS backtracking can decrement column zero below
                # -1 and access invalid memory for longer one-column sequences.
                long_positions = torch.arange(group_scores.size(1), device=scores.device)
                hard[single, :, 0] = (long_positions[None, :] < long_lens[single, None]).to(hard.dtype)
                short_positions = torch.arange(group_scores.size(2), device=scores.device)
                probabilities = group_scores.masked_fill(
                    short_positions[None, None, :] >= short_lens[:, None, None], -1e4
                ).softmax(dim=-1)
                multiple = torch.where(~single)[0]
                if multiple.numel():
                    hard.index_copy_(
                        0,
                        multiple,
                        binarize_attention_parallel(
                            probabilities.index_select(0, multiple)[:, None].clamp_min(1e-12),
                            short_lens[multiple],
                            long_lens[multiple],
                        )[:, 0],
                    )
                path.index_copy_(0, rows, hard if text_is_longer else hard.transpose(1, 2))
        return path, loss


def _turn_batches(rows, metadata, max_turns, max_frames):
    """PR #16519's length-sorted turn packing under a padded codec-frame budget."""
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
