# Enrollment-conditioned target-speaker RNN-T

This experimental recipe trains a cache-aware RNN-T to transcribe one enrolled
speaker from a mixture. It is not a diarization recipe: the model receives a
fixed Titanet speaker embedding and produces one target-activity posterior plus
one RNN-T transcript.

## Entry point

Use [`speech_to_text_target_speaker_rnnt_bpe.py`](speech_to_text_target_speaker_rnnt_bpe.py), not
[`../speech_to_text_finetune.py`](../speech_to_text_finetune.py). The generic
fine-tune script restores the original model class from the ASR checkpoint; it
therefore cannot construct the Sortformer branch or pass enrollment/activity
tensors through training.

Start from the cache-aware FastConformer RNN-T config that matches the current
domain ASR model. The command below shows the required overrides; retain the
optimizer, tokenizer, trainer, and experiment-manager settings from the
existing domain recipe.

```bash
python examples/asr/asr_transducer/speech_to_text_target_speaker_rnnt_bpe.py \
  init_from_nemo_model=/models/domain_cache_aware_rnnt.nemo \
  model.tokenizer.dir=/models/tokenizer \
  model.train_ds.use_lhotse=true \
  model.train_ds.cuts_path=/data/train_cuts.jsonl.gz \
  model.train_ds.background_cuts_path=/data/train_cuts.jsonl.gz \
  model.train_ds.enrollment_cuts_path=/data/train_cuts.jsonl.gz \
  model.train_ds.enrollment_embeddings_path=/data/titanet_large_by_cut_id.pt \
  model.train_ds.num_background_clips='[1,2]' \
  model.train_ds.background_snr='[-5,5]' \
  model.train_ds.absent_target_prob=0.1 \
  model.train_ds.noise_path=null \
  model.spk_kernel_type=ff \
  model.spk_kernel_layers='[0]' \
  model.add_bg_spk_kernel=true \
  model.target_speaker.enrollment_dim=192 \
  model.target_speaker.activity_loss_weight=0.2 \
  model.target_speaker.freeze_backbones_until_step=2000 \
  model.target_speaker.sortformer_model_path=/models/diar_streaming_sortformer_4spk-v2.nemo
```

`init_from_nemo_model` initializes the cache-aware ASR encoder, decoder, and
joint from the domain checkpoint. `sortformer_model_path` independently
initializes the Sortformer backbone. The target activity head, enrollment FiLM
projection, and speaker kernels are new parameters.

To use a ten-times lower learning rate for both pretrained encoder branches,
put the actual values for the chosen scheduler in the model config:

```yaml
model:
  optim_param_groups:
    encoder:
      lr: 1.0e-4
    target_sortformer:
      lr: 1.0e-4
  optim:
    lr: 1.0e-3
```

The values above are an example, not a replacement for the domain recipe's
learning-rate schedule. The remaining, newly created parameters use
`model.optim.lr`.

## Clean-cut data contract

`train_ds.cuts_path` is the **core target** CutSet. Every cut must be a clean,
single-speaker utterance with exactly one speaker identity across its
supervisions. The target transcript is constructed only from these core
supervisions.

Each core cut needs the following Lhotse information:

```text
cut.id:          unique clean-utterance ID
cut.duration:    duration of that utterance
supervision.text: target transcript
supervision.speaker: stable speaker ID
supervision.language: optional tokenizer language
```

For example, a Lhotse CutSet may contain a supervision equivalent to:

```json
{
  "id": "utt-000123",
  "recording_id": "rec-000123",
  "start": 0.0,
  "duration": 3.42,
  "text": "schedule a meeting tomorrow",
  "speaker": "speaker-071"
}
```

Use `background_cuts_path` for the pool of clean clips that can become
interferers. It may be the same CutSet as the core set. The dataset rejects any
background clip whose speaker equals the core target speaker, deep-copies every
selected background cut, and clears `supervisions` before Lhotse constructs the
mixture. Do not set the generic Lhotse `noise_path`: doing so creates a mixed
core before this dataset can enforce the target/background contract.

`enrollment_cuts_path` is the clean reference pool and can also be the same
CutSet. Each target speaker must have at least two distinct clean cuts: one
core utterance and one eligible reference utterance. The reference cut is never
mixed into its corresponding example.

With `absent_target_prob: 0.1`, one tenth of rows contain only different-speaker
background audio. Their activity target and transcript are empty; these rows
train the false-dictation failure mode. Before a full run, execute a one-batch
remote smoke test with the selected RNN-T loss backend to confirm empty-token
loss support.

## Enrollment embeddings

Precompute a frozen, L2-normalized 192-dimensional Titanet-Large embedding for
each clean enrollment cut. `enrollment_embeddings_path` must be a PyTorch file
whose top-level object is a non-empty mapping from **Lhotse `cut.id`** to a
vector:

```python
torch.save(
    {
        "utt-000123": torch.tensor([...]),  # 192 values
        "utt-000124": torch.tensor([...]),
    },
    "/data/titanet_large_by_cut_id.pt",
)
```

The dataset normalizes vectors again and rejects empty or inconsistent widths.
The existing
[`extract_speaker_embeddings.py`](../../speaker_tasks/recognition/extract_speaker_embeddings.py)
is useful for obtaining Titanet embeddings, but its output keys are derived
from audio paths rather than Lhotse cut IDs. Convert that output to the mapping
above, or use an extraction manifest that lets you retain the original cut ID
alongside every returned embedding.

## Compatibility checks and current scope

The target model rejects ASR and Sortformer configurations with different
sample rates, mel-feature counts, window strides, or encoder subsampling
factors. Use a streaming Sortformer checkpoint whose frontend matches the
cache-aware ASR recipe.

This initial implementation supports training and offline tensor-level
inference through `model.forward(..., enrollment_embeddings=embedding)`. It
does not yet add enrollment arguments to the high-level `transcribe()` API,
export a joint streaming graph, or integrate with Riva. Keep the original
cache-aware ASR chunk configuration while measuring accuracy; streaming export
is the next milestone after the offline prototype is validated.
