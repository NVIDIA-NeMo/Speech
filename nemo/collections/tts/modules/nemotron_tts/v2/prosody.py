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
    quantized: torch.Tensor  # (B, FSQ dimensions, text steps), with straight-through gradients
    codes: torch.Tensor  # (B, text steps), PAD outside text_lens
    code_lens: torch.Tensor
    alignment: torch.Tensor  # (B, text steps, audio steps), binary monotonic path
    audio_lens: torch.Tensor
    alignment_loss: torch.Tensor


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
        audio_lens = torch.div(audio_lens * self.sample_rate, self.input_sample_rate, rounding_mode='floor').long()
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

    def teacher_force(self, quantized, codes, code_lens, target_lens, speech_delay, apply_dropout=False):
        """Build compact AR inputs/targets, preserving the learned PAD tail.

        The channel starts with text. The first real target is at
        ``speech_delay - lead_steps``; that PAD input predicts code 0 for the next step.
        There is one real code per valid text token. PAD targets continue through
        acoustic EOS, with at least one PAD target after the last delayed code.
        """
        prefix = speech_delay - self.lead_steps
        if prefix < 0:
            raise ValueError("speech_delay must be >= prosody.lead_steps")
        sequence_lens = torch.maximum(target_lens + speech_delay, prefix + code_lens + 1)
        width = int(sequence_lens.max())
        if codes.size(1) == 0:
            targets = codes.new_full((codes.size(0), width), self.pad_id)
            embedded = self.embed(targets, quantized=quantized.new_zeros(codes.size(0), self.quantizer.dim, width))
            return embedded, targets, sequence_lens
        positions = torch.arange(width, device=codes.device)[None, :]
        source_positions = (positions - prefix).clamp(min=0, max=codes.size(1) - 1)
        source_positions = source_positions.expand(codes.size(0), -1)
        targets = codes.gather(1, source_positions)
        valid_code = (positions >= prefix) & (positions < prefix + code_lens[:, None])
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
                short_positions = torch.arange(group_scores.size(2), device=scores.device)
                probabilities = group_scores.masked_fill(
                    short_positions[None, None, :] >= short_lens[:, None, None], -1e4
                ).softmax(dim=-1)
                hard = binarize_attention_parallel(probabilities[:, None].clamp_min(1e-12), short_lens, long_lens)[
                    :, 0
                ]
                path.index_copy_(0, rows, hard if text_is_longer else hard.transpose(1, 2))
        return path, loss
