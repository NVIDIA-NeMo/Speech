# MagpieTTS Inference and Evaluation

To evaluate any MagpieTTS checkpoint you trained follow the steps as shown below (INTERNAL ONLY):

1) Mount the EOS cluster path `/lustre/fsw/llmservice_nemo_speechlm/data/TTS:/Data`

All the needed manifests are here: `/lustre/fsw/llmservice_nemo_speechlm/data/TTS/evaluation_manifests`

2) Run the following command:
```
CKPT=<Path to .ckpt file for MagpieTTS that you want to evaluate>
HPARAM=<Path to .yaml file for MagpieTTS that you want to evaluate>
CODEC=<Path to .nemo audio codec file for MagpieTTS >
OUT_DIR=<Path to output directory for evaluation >

python scripts/magpietts/infer_and_evaluate.py \
--checkpoint_files ${CKPT} \
--hparams_files ${HPARAM} \
--codecmodel_path ${CODEC} \
--out_dir ${OUT_DIR} \
--use_cfg \
--apply_attention_prior

# If you want streaming
DATASET=<dataset_name e.g. local_longer_1>
python scripts/magpietts/infer_and_evaluate_streaming.py \
--checkpoint_files ${CKPT} \
--hparams_files ${HPARAM} \
--codecmodel_path ${CODEC} \
--out_dir ${OUT_DIR} \
--datasets ${DATASET} \
--use_cfg \
--disable_fcd \
--apply_attention_prior
```

**Test Sets**
The Datasets that we evaluate on are:

- LibriTTS test clean
- LibriTTS seen
- VCTK
- RIVA Hard examples

**Evaluation Metrics**

- ASR of the generated speech is done using `nvidia/parakeet-tdt-1.1b` and then CER/WER is computed.
- Speaker Similarity using `titanet`



# Using Lhotse Datasets in MagpieTTS

Refer to [this file](./README_lhotse.md) for more information about using Lhotse Dataset of MagpieTTS.

# Preference Alignment of MagpieTTS

Refer to [this file](./README_magpie_po.md) for more information about preference alignment of MagpieTTS.