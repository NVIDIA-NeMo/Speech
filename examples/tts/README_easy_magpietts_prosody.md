# EasyMagpie v2 prosody stream

This feature lives in `nemo.collections.tts.models.nemotron_tts.v2.NemotronTTSModel`.
Use `--config-name=easy_magpietts_lhotse_prosody` with the EasyMagpie launcher,
dataset/codec overrides, and your v2 `model.acoustic_codes_predictor_schedule`.
For example, the 13-codebook 12 fps spectral/semantic codec with frame stacking 2
can use `[2, 24]`. Alternatively, add this block to a streaming v2 model config:

```yaml
prosody:
  enabled: true
  encoder_channels: 512
  latent_dim: 128
  alignment_dim: 64
  alignment_temperature: 0.1
  alignment_loss_scale: 0.1
  dropout: 0.5
  loss_scale: 1.0
  lead_steps: 1
```

The target waveform is resampled to 16 kHz and passed through a trainable
`MultiResolutionSTFTEncoder`: STFT `[1280, 320, 1280]`, 512 channels, and four
stride-2 downsampling blocks. This produces continuous 128-dimensional latents at
3.125 Hz (approximately 3 Hz, 320 ms per frame). The final partial frame is
zero-padded and retained. The encoder starts from random weights and learns
jointly with TTS.

Continuous latents are aligned to clean, detached text embeddings before
quantization. Learned audio/text projections produce cosine-similarity scores.
NeMo's width-one monotonic alignment search (MAS) finds a hard path, following the
discrete-search/detached-text pattern in
[Speech PR #16519](https://github.com/NVIDIA-NeMo/Speech/pull/16519).
The projections are trained with a forward-sum CTC alignment loss. This loss
also trains the audio encoder; it does not update the shared text embedding or
CAS encoder. Their normal TTS gradients remain enabled. Text conditioning dropout
and CFG dropout do not corrupt the alignment inputs.

Standard MAS requires its long axis to have at least as many positions as its
short axis. A 3 Hz encoder often has fewer frames than text has BPE tokens, so
the search orientation is selected per utterance. With more audio frames, MAS
assigns consecutive frames to text tokens. With more text tokens, MAS assigns
consecutive text tokens to audio frames, allowing adjacent tokens to share one
coarse audio latent. Both sequences are fully covered, including one-frame audio.
Batch padding is excluded from search.

The resulting path has shape `[B, T_text, T_audio]`. Normalize each text row by
its number of assigned audio frames, then multiply it by the continuous audio
latents. Layer normalization and a projection to four dimensions precede a
single FSQ codebook with levels `[8, 8, 8, 4]` (2048 entries). Thus **the number of
real prosody codes is exactly `text_lens`**, including EOS when present in the
text input. There is no quantization before alignment. If coarse latents are
shared, adjacent text tokens can have identical codes.

The text-aligned code sequence is predicted one code per backbone step. After
the last text-aligned code, the target is a learned PAD token (ID 2048) through
acoustic EOS, with at least one PAD target after the code sequence. The training
timeline extends if necessary to preserve every delayed text code. These PAD
targets contribute to cross-entropy; only padding beyond the per-utterance
supervision timeline is excluded. PAD does not end acoustic generation.

For acoustic delay 5 and `lead_steps: 1`, the first prosody prediction is at step 4.
Steps below are counted from the first target-text input, after context:

| Backbone step | Prosody input | Prosody target | Acoustic input / target |
| --- | --- | --- | --- |
| 0–3 | PAD | PAD | delayed |
| 4 | PAD | code for text token 0 | delayed |
| 5 | code for text token 0 | code for text token 1 | BOS / acoustic frame 0 |
| 6 | code for text token 1 | code for text token 2 | frame 0 / frame 1 |
| after codes end | last code, then PAD | PAD | acoustic generation continues |

The prosody head projects backbone hidden states to 2049 logits (codes plus PAD).
During free running, its predictions are fed into the next step through the same
FSQ decode and input projection used during training. The initial delayed
predictions are forced to PAD. Text EOS (or explicitly ending text input) sets
the exact code count: PAD is suppressed in valid text-code slots and forced
after the last slot, while acoustic generation continues. Prosody uses the inference temperature/top-k
settings and the conditional hidden state with CFG; both CFG branches receive the
same prosody history, as they do for acoustic history.

FSQ vectors have a straight-through gradient and a learned input projection, so
acoustic prediction gradients reach the target encoder. Integer IDs are used as
detached prediction targets. Each real input code is independently replaced by a
separate learned MASK embedding with probability 0.5 during training; targets
stay clean. PAD inputs retain their learned embedding. The total loss adds
`prosody.loss_scale * prosody_loss + prosody.alignment_loss_scale * alignment_loss`.
These are logged separately as `train/prosody_loss` and
`train/prosody_alignment_loss`, with corresponding `val/` metrics.

Prosody is optional and disabled in existing configurations. It currently requires
streaming modes with acoustic delay at least `lead_steps`. Training and validation
require `audio` and `audio_lens` at the acoustic codec's input sample rate; dataset
construction automatically disables cached-code-only loading when enabled. Other
callers of `process_batch` must supply the waveforms even if they supply cached
acoustic codes. Inference needs no target waveform, encoder, or MAS pass.

Run the focused tests, including five toy optimizer steps and streaming feedback:

```bash
PYTHONPATH=. python -m pytest tests/collections/tts/models/nemotron_tts/v2/test_prosody.py -v -s
```

The toy run uses a tiny randomly initialized Nemotron TTS v2 backbone/codec and a
16-channel version of the same prosody encoder. It verifies finite losses and
gradients, actual encoder updates, both MAS orientations, pooling before FSQ,
detached text gradients, exact text/code counts, code/PAD feedback, delay
alignment, and frame stacking. It is a wiring and optimization smoke test;
speech quality, alignment quality, and prosody disentanglement require training
on real speech. The recipe keeps its `1e-4` starting learning rate.
