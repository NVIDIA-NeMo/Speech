# Background
Magpie-TTS uses special tokens like AUDIO_BOS and AUDIO_EOS for its operation. The indices of these tokens are after the audio codec tokens, at the end of the embedding table.

In April 2025 we changed the layout of the embedding table in a non-backwards compatible way:

## Old Layout (until April 16)
With the most common codec configuration (2016 codes), the layout used to look like this:
```
| Index   | Token Description    | Comments                                                                                                  |
|---------|----------------------|-----------------------------------------------------------------------------------------------------------|
| [0]     | Codec Token 0        |                                                                                                           |
| [1]     | Codec Token 1        |                                                                                                           |
| [2]     | Codec Token 2        |                                                                                                           |
| ...     | ...                  |                                                                                                           |
| [2015]  | Codec Token 2015     |                                                                                                           |
| [2016]  | <Unused>             |                                                                                                           |
| [2017]  | <Unused>             |                                                                                                           |
| [2018]  | <Unused>             |                                                                                                           |
| ...     |                      |                                                                                                           |
| [2044]  | Context Audio BOS    | if model_type == `decoder_context_tts`                                                                    |
| [2045]  | Context Audio EOS    | if model_type == `decoder_context_tts`                                                                    |
| [2046]  | Audio BOS            | also used for Context Audio BOS if model_type == `multi_encoder_context_tts` or `single_encoder_sv_tts`   |
| [2047]  | Audio EOS            | also used for Context Audio EOS if model_type == `multi_encoder_context_tts` or `single_encoder_sv_tts`   |
```

## New Layout```
The new layout for the same codec configuration is:
```
| Index   | Token Description    | Comments  |
---------------------------------------------|
| [0]     | Codec Token 0        |           |
| [1]     | Codec Token 1        |           |
| [2]     | Codec Token 2        |           |
| ...     | ...                  |           |
| [2015]  | Codec Token 2015     |           |
| [2016]  | Audio BOS            |           |
| [2017]  | Audio EOS            |           |
| [2018]  | Context Audio BOS    |           |
| [2019]  | Context Audio EOS    |           |
| [2020]  | MASK token (MaskGit) |           |
| [2021]  | RESERVED_1           |           |
| [2022]  | RESERVED_2           |           |
| [2023]  | RESERVED_3           |           |
```

# How to Train and Load a New Checkpoint
For new trainings and inference all configuration is automatic:
* The number of codebooks, codec codebooks size, and codec downsampling rate are all read from the codec checkpoint rather than configured in Magpie.
* The embedding table size is automatically set to codec_codebook_size + number_of_special_tokens (currently 2016+8=2024). There is no risk of accidentally stepping on codec tokens since the table sizes gets automatically sized with enough room for the special tokens.

# How to Load Old Checkpoints
For checkpoints created before the change you can force legacy codebook layout in one of these ways:

## If using `infer_and_evaluate.py`
Just set the `--legacy_codebooks` command line option. No need to update your YAML file – The script will automatically add the overrides.

## If using a Hydra command line
This scenario would happen when either finetuning with an old checkpoint or doing data generation with an old checkpoint.

You have two options:
### Add these to your command line
```
# decoder context model
+model.forced_num_all_tokens_per_codebook=2048 +model.forced_audio_eos_id=2047 +model.forced_audio_bos_id=2046  +model.forced_context_audio_eos_id=2045 +model.forced_context_audio_bos_id=2044

# multi encoder context and any other model type
+model.forced_num_all_tokens_per_codebook=2048 +model.forced_audio_eos_id=2047 +model.forced_audio_bos_id=2046  +model.forced_context_audio_eos_id=2047 +model.forced_context_audio_bos_id=2046
```
# Or, add these overrides to your YAML file
```
forced_num_all_tokens_per_codebook: 2048
forced_audio_eos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -1}           # 2047
forced_audio_bos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -2}           # 2046

# Depending on the old model type, the context_audio_bos_id and context_audio_eos_id will be different (choose one of the pairs below)

# For `multi_encoder_context_tts`, `single_encoder_sv_tts`:
#forced_context_audio_eos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -1}  # 2047
#forced_context_audio_bos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -2}  # 2046

# For `decoder_context_tts` models:
#forced_context_audio_eos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -3}   # 2045
#forced_context_audio_bos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -4}   # 2044
```

# Additional Details
Over the last few weeks we have gone through a few embedding table layouts. When using an old checkpoint it's important to know which layout your checkpoint was trained with and configuring the system accordingly.

* Layout 1: used until April 16 (described in the table above). Add `--legacy-codebooks` to the `infer_and_evaluate.py` command line to inference using this layout.

* Layout 2: after the [config changes](https://github.com/blisc/NeMo/commit/7e2cdca74a866ecefdbe01c0076ad9b5d140ac61): 2018 tokens with special tokens at the end 2017, 2016, 2015, 2014 (the last two being overwrites of codec tokens). This is an invalid layout and these checkpoints should not be used.

* Layout 3: after the [bugfix](https://github.com/blisc/NeMo/commit/23e299a0bd14b666543b4bbcc7783f783acb0bd3) but before the [refactoring](https://github.com/blisc/NeMo/commit/8ba55061a0ebb161abff4b329e402d5307f4af98): 2024 tokens with special tokens at the end (2023, 2022, 2021, 2020). There are no automatic options provided for using this layout but it can be manually configured by updating the `hparams.yaml` file with the `forced_*` options. Set `forced_num_all_tokens_per_codebook` to `2024` and set the rest of the overrides as defined under section `# Or, add these overrides to your YAML file` above.

* Layout 4: The new layout, [from this commit onwards](https://github.com/blisc/NeMo/commit/8ba55061a0ebb161abff4b329e402d5307f4af98): 2024 tokens but with special tokens immediately after codec tokens (2016, 2017, 2018, 2019). Training and inference with the latest version of the code automatically use this layout.
