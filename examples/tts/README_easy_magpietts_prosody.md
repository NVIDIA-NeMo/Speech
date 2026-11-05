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
  dropout: 0.5
  loss_scale: 1.0
  lead_steps: 1
```

The target waveform is resampled to 16 kHz and passed through a trainable
`MultiResolutionSTFTEncoder`: STFT `[1280, 320, 1280]`, 512 channels, and three
stride-2 downsampling blocks. This is the semantic codec encoder with one extra
downsampling block, producing 6.25 Hz codes. A single FSQ codebook with levels
`[8, 8, 8, 4]` has 2048 entries. The final partial frame is zero-padded and retained.
The encoder starts from random weights and learns jointly with TTS.

The compact code sequence is predicted one code per backbone step. It is not
upsampled or repeated to match acoustic duration. After its last code, the target
is a learned PAD token (ID 2048) through acoustic EOS. These PAD targets contribute
to cross-entropy; only batch padding beyond each utterance's acoustic EOS is
excluded. PAD does not end acoustic generation.

For acoustic delay 5 and `lead_steps: 1`, the first prosody prediction is at step 4.
Steps below are counted from the first target-text input, after context:

| Backbone step | Prosody input | Prosody target | Acoustic input / target |
| --- | --- | --- | --- |
| 0–3 | PAD | PAD | delayed |
| 4 | PAD | code 0 | delayed |
| 5 | code 0 | code 1 | BOS / acoustic frame 0 |
| 6 | code 1 | code 2 | frame 0 / frame 1 |
| after codes end | last code, then PAD | PAD | acoustic generation continues |

The prosody head projects backbone hidden states to 2049 logits (codes plus PAD).
During free running, its predictions are fed into the next step through the same
FSQ decode and input projection used during training. The initial delayed
predictions are forced to PAD. Prosody uses the inference temperature/top-k
settings and the conditional hidden state with CFG; both CFG branches receive the
same prosody history, as they do for acoustic history.

FSQ vectors have a straight-through gradient and a learned input projection, so
acoustic prediction gradients reach the target encoder. Integer IDs are used as
detached prediction targets. Each real input code is independently replaced by a
separate learned MASK embedding with probability 0.5 during training; targets
stay clean. PAD inputs retain their learned embedding. The total loss adds
`prosody.loss_scale * prosody_loss`, logged as `train/prosody_loss` and
`val/prosody_loss`.

Prosody is optional and disabled in existing configurations. It currently requires
streaming modes with acoustic delay at least `lead_steps`. Training and validation
require `audio` and `audio_lens` at the acoustic codec's input sample rate; dataset
construction automatically disables cached-code-only loading when enabled. Other
callers of `process_batch` must supply the waveforms even if they supply cached
acoustic codes. Inference needs no target waveform or encoder pass.

Run the focused tests, including five toy optimizer steps and streaming feedback:

```bash
PYTHONPATH=. python -m pytest tests/collections/tts/models/nemotron_tts/v2/test_prosody.py -v -s
```

The toy run uses a tiny randomly initialized Nemotron TTS v2 backbone/codec and a
16-channel version of the same prosody encoder. It verifies finite losses and
gradients, actual encoder updates, code/PAD feedback, delay alignment, and frame
stacking. It is a wiring and optimization smoke test; speech quality and prosody
disentanglement require training on real speech.

The full 512-channel encoder also passed five BF16 Lightning steps with the real
12 fps codec and v2 acoustic predictor, followed by free-running CFG synthesis.
At the recipe's `1e-4` learning rate, all five steps had finite, nonzero encoder
gradients. A synthetic preflight at `1e-3` produced zero encoder gradients after
two steps; use the recipe's learning rate when starting full-size experiments.
