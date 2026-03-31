.. _magpie-tts-finetuning:

======================
Magpie-TTS Finetuning
======================

Finetuning a pretrained Magpie-TTS checkpoint lets you adapt the model to new voices or new languages without training from scratch. The pretrained model has already learned general speech patterns, prosody, and acoustic modeling, so finetuning requires far less data and compute than pretraining. This guide covers two common finetuning scenarios:

- **Adding new speakers in an existing language** — adapt the model to speak in voices not seen during pretraining, using a small dataset of target-speaker audio.
- **Adding a new language** — extend the model to synthesize speech in a language absent from the pretraining data, using a multilingual dataset configuration.

For preference optimization (DPO/GRPO) on top of a finetuned checkpoint, see :doc:`Magpie-TTS Preference Optimization <magpietts-po>`.


Prerequisites
#############

Before finetuning, you will need:

- A pretrained Magpie-TTS checkpoint (``pretrained.ckpt`` or ``pretrained.nemo``). Public checkpoints (``https://huggingface.co/nvidia/magpie_tts_multilingual_357m``) are available on Hugging Face.
- The audio codec model (``https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps``), available on Hugging Face alongside the TTS checkpoint.
- A prepared dataset. For faster finetuning audio codec tokens must be pre-extracted from your audio files. See the *Dataset Preparation* section below.
- NeMo installed from source or via the NeMo container. See the `NeMo GitHub page <https://github.com/NVIDIA/NeMo>`_ for installation instructions.


Dataset Preparation
-------------------

Training uses ``MagpieTTSDataset`` with ``dataset_meta`` entries (see ``DatasetMeta`` in ``nemo/collections/tts/data/text_to_speech_dataset.py``). Each line in ``manifest_path`` file is one training example.

**Optional cached codec codes.** If each line includes ``target_audio_codes_path`` and ``context_audio_codes_path`` (paths to saved tensors) and ``model.load_cached_codes_if_available=true``, the dataloader can skip on-the-fly codec encoding. If those keys are absent, the codec runs during training and loads waveform from ``audio_filepath`` and ``context_audio_filepath`` (slower but no separate extraction step).

**Minimum fields** (paths relative to ``audio_dir`` / ``feature_dir`` unless you use absolute paths):

.. code-block:: json

    {
      "audio_filepath": "relative/path/to/audio.wav",
      "text": "transcript of the utterance",
      "duration": 5.2,
      "context_audio_filepath": "relative/path/to/context.wav",
      "context_text": "transcript of the context audio",
      "target_audio_codes_path": "/optional/path/to/target_codes.pt",
      "context_audio_codes_path": "/optional/path/to/context_codes.pt"
    }

The ``context_audio_filepath`` is the reference audio used for voice cloning during training. It should come from the same speaker as ``audio_filepath``. A minimum context duration of about 3 seconds and a high speaker similarity (for example ≥ 0.6 with TitaNet) are recommended for best results.

**Registering datasets in config.** For each named split in ``train_ds_meta`` and ``val_ds_meta``, set ``manifest_path``, ``audio_dir``, ``feature_dir``, ``sample_weight`` (training), and ``tokenizer_names``: a list of keys that exist under ``model.text_tokenizers`` in the config. The dataloader picks the tokenizer for each sample from that list (see ``DatasetMeta``).


.. _magpie-tts-new-speaker:

Adding New Speakers in an Existing Language
###########################################

This scenario adapts a pretrained checkpoint to new speakers in a language the model already supports (for example adding new English speakers to a checkpoint trained on English data). You are teaching new voice characteristics while keeping the same text tokenizer. Mixing in some public Magpie-TTS data can reduce regression; see the `Magpie-TTS dataset <https://huggingface.co/nvidia/magpie_tts_multilingual_357m#training-dataset>`_ on Hugging Face.

Key training choices:

- **Low learning rate** (``5e-6``): the pretrained model is already well-converged; a high LR can destroy learned representations.
- **Disable alignment prior** (``alignment_loss_scale=0.0``, ``prior_scaling_factor=null``): the prior helps pretraining but can over-constrain finetuning.
- **Tokenizer**: use ``tokenizer_names: [english_phoneme]`` (or the tokenizer that matches your transcripts) on each ``train_ds_meta`` / ``val_ds_meta`` entry.

``magpietts.yaml`` trains with ``max_epochs`` and top-level ``batch_size``. Validation mixes all ``val_ds_meta`` entries in a single dataloader (joint validation metrics).

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-path=examples/tts/conf/magpietts \
        --config-name=magpietts \
        +init_from_ptl_ckpt=/path/to/pretrained.ckpt \
        exp_manager.exp_dir=/path/to/output \
        +train_ds_meta.en_sft.manifest_path=/path/to/train.json \
        +train_ds_meta.en_sft.audio_dir=/path/to/audio \
        +train_ds_meta.en_sft.feature_dir=/path/to/features \
        +train_ds_meta.en_sft.sample_weight=1.0 \
        "+train_ds_meta.en_sft.tokenizer_names=[english_phoneme]" \
        +val_ds_meta.en_val.manifest_path=/path/to/val.json \
        +val_ds_meta.en_val.audio_dir=/path/to/audio \
        +val_ds_meta.en_val.feature_dir=/path/to/audio \
        +val_ds_meta.en_val.sample_weight=1.0 \
        "+val_ds_meta.en_val.tokenizer_names=[english_phoneme]" \
        model.codecmodel_path=nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
        model.context_duration_min=5.0 \
        model.context_duration_max=5.0 \
        model.alignment_loss_scale=0.0 \
        model.prior_scaling_factor=null \
        model.optim.lr=5e-6 \
        ~model.optim.sched \
        model.load_cached_codes_if_available=true \
        trainer.precision=32 \
        trainer.devices=8 \
        trainer.num_nodes=1 \
        batch_size=16 \
        max_epochs=500

The ``+init_from_ptl_ckpt`` flag loads the pretrained checkpoint weights before training begins. The ``+`` prefix is required because this key is not present in the base config.

``~model.optim.sched`` removes the learning rate schedule so the LR stays constant during finetuning.

``trainer.precision=32`` is recommended for finetuning stability. Mixed precision (``bf16`` or ``16``) can cause loss instability on small datasets.


.. _magpie-tts-new-language:

Adding a New Language
#####################

This scenario extends the model to one or more languages not present in the pretraining data. Use the same ``magpietts`` config and combine multiple manifests with per-language ``sample_weight``.

**Tokenizers**

- Define each new tokenizer under ``model.text_tokenizers`` (for example an ``AutoTokenizer`` with ``google/byt5-small`` for scripts outside the IPA vocabulary).
- **How it is applied:** each ``train_ds_meta`` / ``val_ds_meta`` entry lists ``tokenizer_names`` (keys under ``model.text_tokenizers``). The dataloader uses those names to select which tokenizer encodes each sample’s transcript (see ``DatasetMeta`` in ``nemo/collections/tts/data/text_to_speech_dataset.py``).

**Per-language entries**

Each language is a separate key under ``train_ds_meta`` / ``val_ds_meta`` with ``manifest_path``, ``audio_dir``, ``feature_dir``, ``sample_weight``, and ``tokenizer_names``.

**Sample weights**

Upsample low-resource languages with a higher ``sample_weight`` so they are not drowned out by high-resource languages.

Align transcript format with the tokenizer you choose (IPA phonemes for ``english_phoneme`` / IPA-style tokenizers, raw text for byte-level models, and so on). Audio codes can be cached as in *Dataset Preparation*.

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-name=magpietts \
        +init_from_ptl_ckpt=/path/to/pretrained.ckpt \
        exp_manager.exp_dir=/path/to/output \
        +model.text_tokenizers.your_language_chartokenizer._target_=AutoTokenizer \
        +model.text_tokenizers.your_language_chartokenizer.pretrained_model="google/byt5-small" \
        +train_ds_meta.your_language.manifest_path=/path/to/your_lang_train.json \
        +train_ds_meta.your_language.audio_dir=/path/to/your_lang_audio \
        +train_ds_meta.your_language.feature_dir=/path/to/your_lang_audio \
        +train_ds_meta.your_language.sample_weight=1.0 \
        "+train_ds_meta.your_language.tokenizer_names=[your_language_chartokenizer]" \
        +val_ds_meta.your_language_dev.manifest_path=/path/to/your_lang_val.json \
        +val_ds_meta.your_language_dev.audio_dir=/path/to/your_lang_audio \
        +val_ds_meta.your_language_dev.feature_dir=/path/to/your_lang_audio \
        +val_ds_meta.your_language_dev.sample_weight=1.0 \
        "+val_ds_meta.your_language_dev.tokenizer_names=[your_language_chartokenizer]" \
        model.codecmodel_path=nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
        model.context_duration_min=5.0 \
        model.context_duration_max=5.0 \
        model.alignment_loss_scale=0.0 \
        model.prior_scaling_factor=null \
        model.optim.lr=1e-5 \
        ~model.optim.sched \
        model.load_cached_codes_if_available=true \
        trainer.precision=32 \
        trainer.devices=8 \
        trainer.num_nodes=1 \
        max_epochs=500


Mixing Multiple Languages
--------------------------

Add one ``train_ds_meta`` entry per language. Increase ``sample_weight`` for low-resource languages. You can mix public Magpie-TTS data with your own; see the `Magpie-TTS dataset <https://huggingface.co/nvidia/magpie_tts_multilingual_357m#training-dataset>`_ on Hugging Face.

.. code-block:: bash

        # High-resource languages — standard weight
        +train_ds_meta.spanish.manifest_path=/path/to/spanish_train.json \
        +train_ds_meta.spanish.audio_dir=/path/to/spanish_audio \
        +train_ds_meta.spanish.feature_dir=/path/to/spanish_audio \
        +train_ds_meta.spanish.sample_weight=1.0 \
        "+train_ds_meta.spanish.tokenizer_names=[spanish_phoneme_or_chartokenizer]" \
        +train_ds_meta.french.manifest_path=/path/to/french_train.json \
        +train_ds_meta.french.audio_dir=/path/to/french_audio \
        +train_ds_meta.french.feature_dir=/path/to/french_audio \
        +train_ds_meta.french.sample_weight=1.0 \
        "+train_ds_meta.french.tokenizer_names=[french_chartokenizer]" \
        # Low-resource language — upsampled 10x
        +train_ds_meta.low_resource_lang.manifest_path=/path/to/low_resource_train.json \
        +train_ds_meta.low_resource_lang.audio_dir=/path/to/low_resource_audio \
        +train_ds_meta.low_resource_lang.feature_dir=/path/to/low_resource_audio \
        +train_ds_meta.low_resource_lang.sample_weight=5.0 \
        "+train_ds_meta.low_resource_lang.tokenizer_names=[low_resource_chartokenizer]"

With ``model.load_cached_codes_if_available=true``, precomputed ``target_audio_codes_path`` / ``context_audio_codes_path`` in the manifest avoid recomputing codec codes at train time.


Preference Optimization After Finetuning
#########################################

After supervised finetuning, you can further improve quality with GRPO. For commands and hyperparameters, see :doc:`Magpie-TTS Preference Optimization <magpietts-po>` (the GRPO example uses ``--config-name=magpietts`` with ``+mode=onlinepo_train``).


Key Hyperparameter Reference
#############################

.. list-table::
   :widths: 35 25 40
   :header-rows: 1

   * - Parameter
     - Typical Value
     - Notes
   * - ``model.optim.lr``
     - ``5e-6`` (same-language speakers), ``1e-5`` (multilingual)
     - Much lower than pretraining LR to preserve learned features
   * - ``max_epochs``
     - tens to hundreds
     - Shorter runs for small datasets; monitor validation loss
   * - ``model.alignment_loss_scale``
     - ``0.0``
     - Disable alignment prior during finetuning
   * - ``model.prior_scaling_factor``
     - ``null``
     - Disable alignment prior during finetuning
   * - ``trainer.precision``
     - ``32``
     - Recommended for finetuning stability
   * - ``model.cfg_unconditional_prob``
     - ``0.1``
     - Classifier-free guidance dropout rate during training
