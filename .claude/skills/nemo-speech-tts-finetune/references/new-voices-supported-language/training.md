# Stage 3: Training, Checkpoint Selection, Export

The task's values (learning rates, `max_steps`, validation interval, expected load log) are in the task file
(`references/new-voices-supported-language/README.md`).

## Config

`examples/tts/conf/magpietts/easy_magpietts_finetuning.yaml` is a standalone config (no `defaults:` list) for
fine-tuning the released checkpoint: the checkpoint's architecture, its training settings,
Lhotse Shar training data and one validation set. The model section must match the
checkpoint; do not edit it. The config also sets checkpointing and early stopping. Per-run values stay `???` and are set on the command line;
so are the learning rates and the constant-LR schedule: the config keeps the pretraining defaults there (`lr` 1e-4,
LR × 0.998 per epoch).

TODO: the config is added with the model release.

## Train (one-voice example; values from `references/new-voices-supported-language/README.md`)

```bash
python examples/tts/easy_magpietts.py --config-name=easy_magpietts_finetuning \
  name=<voice>_ft +pretrained_model=<checkpoint>.nemo \
  model.codecmodel_path=<codec>.nemo model.phoneme_tokenizer.tokenizer_path=<ipa_tokenizer>.json \
  model.train_ds.dataset.tts_data.input_cfg=<voice>_input_cfg.yaml model.train_ds.dataset.tts_data.batch_duration=100 \
  model.validation_ds.dataset.dataset_meta.val.manifest_path=<voice>/val.jsonl \
  model.validation_ds.dataset.dataset_meta.val.language=<code from the language table, references/new-voices-supported-language/data.md> \
  model.use_speaker_encoder=true \
  model.optim.lr=5e-6 '~model.optim.sched' \
  '+model.optim_param_group_rules.speaker_encoder.param_names=[speaker_encoder.*]' \
  '+model.optim_param_group_rules.speaker_encoder.lr=5e-5' \
  trainer.max_steps=20000 trainer.val_check_interval=50 \
  exp_manager.explicit_log_dir=<work>/runs/<voice>_ft
```

Weights load from `pretrained_model` (partial load: modules absent from the checkpoint, such as a new speaker encoder,
keep their random initialisation). `+` adds a key absent from the config (`pretrained_model`, `optim_param_group_rules`), and `~` removes one
(`model.optim.sched`).

| Key | Rule |
| --- | --- |
| `+pretrained_model` | the checkpoint the user has (`.nemo`); check the restored-tensor count. Without it the model trains from random weights with no error: the restore line is then missing from the load log |
| `model.optim.lr`, `optim_param_group_rules.<module>.lr` | from the task file (per-module LRs: `references/new-voices-supported-language/setup.md`); without `model.optim.lr` the config's 1e-4 applies |
| `~model.optim.sched` | removes the config's LR schedule, so the LR stays constant; without it the LR is multiplied by 0.998 once per epoch (`ExponentialLR` steps per epoch; an epoch is `trainer.limit_train_batches` = 1000 steps) |
| `model.train_ds.dataset.tts_data.batch_duration` | 100 (seconds of audio per batch) on an 80 GB GPU; GPUs with less memory are untested |
| `trainer.max_steps` | an upper bound; early stopping ends the run |
| `trainer.val_check_interval` | from the task file |
| checkpointing (config) | validation loss (min), early stopping after 5 checks without improvement, 2 lowest-loss checkpoints kept plus a `-last` copy, weights only, file names `step=<N>-val_loss=<L>.ckpt` (the checkpoint choice reads them) |
| `exp_manager.explicit_log_dir` | a new directory per run: the config does not resume (weights-only checkpoints), and a second run in the same directory writes its checkpoints next to the first run's. A run that dies cannot be continued: evaluate its kept checkpoints, or restart from the release in a new directory. Write your own files (job log, notes) elsewhere, e.g. `<work>/logs/`: files already in the directory when training starts are moved into `run_0/` |

Give the job a time limit of 2–3× the task file's run-time estimate, within the cluster's limit; a run that hits
`trainer.max_steps` or the time limit without early stopping still leaves usable kept checkpoints. A job killed at its
time limit is reported as failed (SLURM `TIMEOUT`), so a next step chained with `afterok` never starts: chain with
`afterany` and start the next step if kept checkpoints exist. Checks run after training in the same job must not turn a
finished run into a failure (e.g. a `grep` that finds nothing exits non-zero under `set -e`).

Validation utterances longer than 20 s are dropped by the validation loader without a message (`max_duration` of
`validation_ds` in the config): keep validation utterances ≤ 20 s, or report how many are
dropped.

### Load-log checks (first minutes)

- restored-tensor count as the task file expects (new voice: `994 / 1001 ... (994 exact, 0 partial, 0 skipped due to incompatible shape)`);
- with per-module LRs: `Optimizer config = AdamW (` followed by `Parameter Group 0` (`lr: 5e-06`) and
  `Parameter Group 1` (`lr: 5e-05`); with the task file's one-learning-rate alternative, which drops the
  `optim_param_group_rules` overrides, only `Parameter Group 0`;
- `val_loss` logged at the first validation and decreasing over the first checks.

## Checkpoint choice between the 2 kept checkpoints

The run directory holds `hparams.yaml` and `checkpoints/step=<N>-val_loss=<L>.ckpt` (weights only, no `.nemo`); the
2 kept checkpoints are the two with the lowest `val_loss` in the file name (ignore the `*-last.ckpt` copy). The
evaluator reads them directly, so no export is needed to choose: evaluate both on the validation set in checkpoint mode
with 3 repeats (`references/new-voices-supported-language/evaluation.md`; one comma-separated list per flag):

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python examples/tts/magpietts_inference.py <evaluation flags> --num_repeats 3 \
  --hparams_files <run>/hparams.yaml,<run>/hparams.yaml \
  --checkpoint_files <run>/checkpoints/<ckpt_1>.ckpt,<run>/checkpoints/<ckpt_2>.ckpt \
  --out_dir <work>/eval/<voice>_val
```

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is needed because Lightning checkpoints store the config as an OmegaConf object,
which recent PyTorch refuses to load by default; use it only for your own checkpoints. Then, over the 3 repeats
(`<set>_metrics_<i>.json`, `<set>_filewise_metrics_<i>.json`, i = 0, 1, 2), keep:

1. drop checkpoints with any **broken output** (as in `references/new-voices-supported-language/evaluation.md`, "Metrics"): empty (`pred_text`
   empty), runaway (output ≥ 90% of the length limit, ≈ 72 s with `--max_decoder_steps 1000`), silent (peak amplitude
   below 0.01) or far too long (output > 2 × its real recording + 1 s); for English also `eou_cutoff_rate` > 0.02
   (mean over the 3 repeats).
   The ASR hears only the first 30 s of an output (Whisper), so runaways can have a low CER: read the durations from
   the wav files. If none is left, keep those with the fewest broken outputs (with several voices: the total over
   all voices);
2. lowest **cumulative CER** (`cer_cumulative`, mean over the 3 repeats) among them; not `cer_filewise_avg`, where one
   garbled sentence (CER 4 or 20 on its own) decides the mean alone. Only if the cumulative CERs are exactly equal,
   the lower `val_loss` (ties → earlier step). One repeat is not enough: on ~100
   utterances a single dropped word moves CER by a few tenths of a point.

SSIM (`ssim_pred_context_avg`) is recorded in the ledger for each checkpoint but does not decide.

```python
import glob, json, numpy as np, soundfile as sf
def summary(model_dir, set_name, limit_s=72.0):  # one model's folder of the evaluator's output
    items = [it for f in sorted(glob.glob(f"{model_dir}/{set_name}_filewise_metrics_*.json")) for it in json.load(open(f))]
    broken = 0
    for it in items:
        wav, sr = sf.read(it["pred_audio_filepath"])
        out_s = len(wav) / sr
        broken += (not it["pred_text"].strip() or out_s >= limit_s or np.abs(wav).max(initial=0.0) < 0.01
                   or out_s > 2 * sf.info(it["gt_audio_filepath"]).duration + 1)
    return {"broken": broken}  # CER: the mean of cer_cumulative in <set>_metrics_<i>.json
```

The two checkpoints come from the same run, a few hundred steps apart, so this choice may use Whisper CER (the
evaluation rule against deciding on Whisper CER alone is for different models). Languages other than English have no
automatic cut-off check; a cut-off last word raises the CER, which step 2 already uses. Export the chosen checkpoint
without waiting for a listening step.

## Export

Convert the chosen checkpoint to a `.nemo` (training config from `hparams.yaml` + trained weights). Run in the training
container, 1 CPU node is enough (~40 GB RAM):

```python
import sys, torch
from omegaconf import OmegaConf, open_dict
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

hparams, ckpt, out = sys.argv[1:4]  # <run>/hparams.yaml, <run>/checkpoints/<chosen>.ckpt, <voice>.nemo
cfg = OmegaConf.load(hparams)
cfg = cfg.cfg if "cfg" in cfg else cfg
with open_dict(cfg):  # no data loaders in the exported model
    cfg.train_ds = None
    cfg.validation_ds = None
    cfg.run_val_inference = False
model = EasyMagpieTTSInferenceModel(cfg=cfg)
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])  # strict
model.save_to(out)
```

`load_state_dict` is strict: it fails if the checkpoint and the config disagree. The `.nemo` holds the same tensors as
the checkpoint (new voice: 1001, of which the speaker encoder's are new). Evaluate the `.nemo` on the test set
(`--nemo_files`); this `.nemo` is the voice's model.

The `.nemo` stores the codec and phoneme-tokenizer paths of the training machine (as the release does): ship both files
with it, and on another machine load it with their new paths (the evaluator does the same with `--codecmodel_path` and
`--phoneme_tokenizer_path`):

```python
from nemo.collections.tts.modules.magpietts_inference import ModelLoadConfig, load_easy_magpie_model
model, _ = load_easy_magpie_model(ModelLoadConfig(nemo_file="<voice>.nemo", codecmodel_path="<codec>.nemo",
                                                  phoneme_tokenizer_path="<ipa_tokenizer>.json"))
```

## Several voices / several validation sets

Training data: one Lhotse group for all the voices of a language, or one group per voice when the user asks for equal
shares (`references/new-voices-supported-language/data.md`). Validation: one validation manifest with every voice's validation lines (each line keeps its own `speaker`
field and its voice's reference clip), with the same number of lines per voice so that each voice weighs the same in
`val_loss`; it is set with the usual two `dataset_meta.val` overrides. Several languages: one validation set per
language; `val` stays the first language's set, each other language adds its own (all four keys, `+` because they are
new; quote the list):

```bash
  model.validation_ds.dataset.dataset_meta.val.manifest_path=<lang_1>/val.jsonl \
  model.validation_ds.dataset.dataset_meta.val.language=<code_1> \
  +model.validation_ds.dataset.dataset_meta.val_<lang_2>.manifest_path=<lang_2>/val.jsonl \
  +model.validation_ds.dataset.dataset_meta.val_<lang_2>.audio_dir=/ \
  '+model.validation_ds.dataset.dataset_meta.val_<lang_2>.tokenizer_names=[nemotron_nano_30b]' \
  +model.validation_ds.dataset.dataset_meta.val_<lang_2>.language=<code_2> \
```

`val_loss` is computed over all sets together. Choose between the 2 kept checkpoints with the standalone evaluation of
each voice on its own validation lines: split the validation manifest by voice into one evalset entry per voice (do not
pool the voices): a broken output in any voice drops the checkpoint (if both have one, keep the one with fewer broken
outputs in total over all voices); then the lowest mean over voices of their cumulative CERs, each voice with equal
weight. Task-specific advice: task file (`references/new-voices-supported-language/README.md`).
