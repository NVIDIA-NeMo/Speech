# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learnable low-rate target-audio conditioning for EasyMagpie."""

import torch
from torch import nn
from torch.nn import functional as F

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer, MultiResolutionSTFTEncoder


class ProsodyChannel(nn.Module):
    """A 6.25 Hz spectrogram encoder, one FSQ codebook, and AR input/output projections.

    The default encoder is the 16 kHz semantic codec encoder with one extra stride-2
    block (320 * 2**3 = 2560 samples/token). FSQ vectors are projected into backbone
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
    ):
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("prosody.dropout must be in [0, 1]")
        if lead_steps < 0:
            raise ValueError("prosody.lead_steps must be nonnegative")
        self.input_sample_rate = input_sample_rate
        self.sample_rate = 16000
        self.samples_per_frame = 2560
        self.dropout = dropout
        self.loss_scale = loss_scale
        self.lead_steps = lead_steps
        self.quantizer = FiniteScalarQuantizer(num_levels=[8, 8, 8, 4])
        self.vocab_size = self.quantizer.codebook_size
        self.pad_id = self.vocab_size
        self.mask_id = self.vocab_size + 1
        self.encoder = MultiResolutionSTFTEncoder(
            out_dim=self.quantizer.dim,
            resolutions=[[1280, 320, 1280]],
            resolution_filter_list=[encoder_channels],
            down_sample_filter_list=[encoder_channels] * 3,
            pad_mode='replicate',
        )
        self.input_projection = nn.Linear(self.quantizer.dim, embedding_dim)
        self.special_embeddings = nn.Embedding(2, embedding_dim)
        self.output_projection = nn.Linear(hidden_dim, self.vocab_size + 1)

    def encode(self, audio, audio_lens):
        """Encode target waveforms at the acoustic codec's input sample rate."""
        audio = resample(audio, self.input_sample_rate, self.sample_rate)
        audio_lens = torch.div(audio_lens * self.sample_rate, self.input_sample_rate, rounding_mode='floor').long()
        # Codec downsampling uses floor lengths. Pad each utterance to retain its
        # final partial prosody frame, including utterances shorter than one frame.
        frame_lens = torch.div(audio_lens + self.samples_per_frame - 1, self.samples_per_frame, rounding_mode='floor')
        padded_lens = frame_lens * self.samples_per_frame
        positions = torch.arange(audio.size(1), device=audio.device)
        audio = audio * (positions[None, :] < audio_lens[:, None])
        audio = F.pad(audio, (0, max(0, int(padded_lens.max()) - audio.size(1))))
        # Spectrograms/FSQ need FP32 even under mixed precision.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            latents, encoded_lens = self.encoder(audio=audio.float(), audio_len=padded_lens)
            quantized, indices = self.quantizer(inputs=latents.float(), input_len=encoded_lens)
        return quantized, indices[0].long(), encoded_lens

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
        After the compact 6.25 Hz sequence ends, PAD targets continue through audio EOS.
        """
        prefix = speech_delay - self.lead_steps
        if prefix < 0:
            raise ValueError("speech_delay must be >= prosody.lead_steps")
        sequence_lens = target_lens + speech_delay
        width = int(sequence_lens.max())
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
