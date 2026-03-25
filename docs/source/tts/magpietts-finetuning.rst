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

For faster finetuning, Magpie-TTS audio codec tokens can be pre-computed and stored alongside each audio file. Run the codec model over your audio dataset to generate and cache the audio codes before launching any training job:

.. code-block:: bash

    python scripts/magpietts/extract_audio_codes.py \
        --manifest_path /path/to/your_manifest.json \
        --audio_dir /path/to/audio \
        --codecmodel_path nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
        --output_manifest /path/to/your_manifest_withAudioCodes.json

Each manifest entry should be a JSON line with at minimum:

.. code-block:: json

    {
      "audio_filepath": "relative/path/to/audio.wav",
      "text": "transcript of the utterance",
      "duration": 5.2,
      "context_audio_filepath": "relative/path/to/context.wav",
      "context_text": "transcript of the context audio"
    }

The ``context_audio_filepath`` is the reference audio that the model uses for voice cloning during training. It should come from the same speaker as ``audio_filepath``. A minimum context duration of 3 seconds and a speaker similarity score of at least 0.6 (measured by TitaNet) are recommended for best results.

For Lhotse-style dataset loading (used in the English SFT scenario), manifest entries are organized into a YAML ``input_cfg`` file instead of being passed directly. See the Lhotse configuration examples below.


.. _magpie-tts-new-speaker:

Adding New Speakers in an Existing Language
###########################################

This scenario adapts a pretrained checkpoint to a set of target speakers in a specific language already present in the pretrained checkpoint. The model already knows the language; you are teaching it new voice characteristics. You should consider mixing the new data with some of the publicly available existing data to prevent degradation of the model. You can find the publicly available data in the `Magpie-TTS dataset <https://huggingface.co/nvidia/magpie_tts_multilingual_357m#training-dataset>`_ on Hugging Face.

Key training choices for speaker finetuning:

- **Low learning rate** (``5e-6``): the pretrained model is already well-converged. A high LR will destroy the learned representations.
- **Disable alignment prior** (``alignment_loss_scale=0.0``, ``prior_scaling_factor=null``): the alignment prior is beneficial during pretraining to enforce monotonicity, but during finetuning it can over-constrain adaptation and hurt quality.
- **Context audio filtering**: ensure each training sample has a high-quality context audio from the same speaker. Use ``min_context_speaker_similarity: 0.6`` in your Lhotse manifest to filter low-quality pairs.

The dataset uses Lhotse's ``input_cfg`` YAML format, which supports bucketed batching by duration for efficient GPU utilization:

.. code-block:: yaml

    # train_input_cfg.yaml
    - input_path: /path/to/your_dataset_shards/
      type: nemo_tarred
      weight: 1.0

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-path=examples/tts/conf/magpietts \
        --config-name=magpietts_en_v2_lhotse \
        +init_from_ptl_ckpt=/path/to/pretrained.ckpt \
        +exp_manager.explicit_log_dir=/path/to/output \
        model.codecmodel_path=nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
        model.train_ds.dataset.input_cfg=/path/to/train_input_cfg.yaml \
        model.train_ds.dataset.batch_duration=500 \
        "+model.train_ds.dataset.bucket_duration_bins=[4.96,5.92,6.8,7.6,8.4,9.2,10.0,10.72,11.46,12.24,13.07,13.92,14.82,15.79,16.8,17.92,18.96,19.6,19.92]" \
        model.validation_ds.dataset.input_cfg=/path/to/val_input_cfg.yaml \
        model.validation_ds.dataset.batch_duration=300 \
        model.optim.lr=5e-6 \
        ~model.optim.sched \
        model.alignment_loss_scale=0.0 \
        model.prior_scaling_factor=null \
        trainer.max_steps=15000 \
        trainer.precision=32 \
        trainer.devices=8 \
        trainer.num_nodes=1

The ``+init_from_ptl_ckpt`` flag loads the pretrained checkpoint weights before training begins. The ``+`` prefix is required because this key is not present in the base config.

``~model.optim.sched`` removes the learning rate schedule so the LR stays constant throughout finetuning. For short finetuning runs a fixed low LR is more stable than a decaying schedule.

``trainer.precision=32`` is recommended for finetuning stability. Mixed precision (``bf16`` or ``16``) can cause loss instability when adapting to small datasets.


.. _magpie-tts-new-language:

Adding a New Language
#####################

This scenario extends the model to synthesize speech in one or more languages not present in the pretraining data. The multilingual finetuning config uses the non-Lhotse ``train_ds_meta`` dataset format, which is better suited for combining multiple language-specific manifests with per-language sample weights. You should consider mixing the new data with some of the publicly available existing data to prevent degradation of the model. You can find the publicly available data in the `Magpie-TTS dataset <https://huggingface.co/nvidia/magpie_tts_multilingual_357m#training-dataset>`_ on Hugging Face.

Key differences from the English SFT scenario:

- **Byte-level tokenizer** (``google/byt5-small``): when adding languages whose characters fall outside the pretraining vocabulary, use a byte-level tokenizer. This avoids out-of-vocabulary tokens for new scripts or phoneme sets and lets the model learn new language representations without modifying the vocabulary.
- **Per-language dataset entries** (``train_ds_meta.<lang_name>``): each language is registered as a separate entry with its own manifest path, audio directory, and ``sample_weight``. This makes it straightforward to control the language balance during training.
- **Sample weight upsampling**: languages with less data can be upsampled via ``sample_weight``. For instance, a language with only 30 minutes of training data might use ``sample_weight=10.0`` alongside a resource-rich language at ``sample_weight=1.0`` to prevent the larger language from dominating training.

Dataset manifest entries for multilingual finetuning use the IPA phoneme representation. Ensure your manifests include IPA-transcribed text fields. Audio codes must be pre-extracted as described in the *Dataset Preparation* section.

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-name=magpietts_multilingual_v1 \
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

To train on several languages simultaneously, add one ``train_ds_meta`` block per language. Languages with less data should use a higher ``sample_weight`` to compensate:
You should consider mixing the new data with some of the publicly available existing data to prevent degradation of the model. You can find the publicly available data in the `Magpie-TTS dataset <https://huggingface.co/nvidia/magpie_tts_multilingual_357m#training-dataset>`_ on Hugging Face.

.. code-block:: bash

        # High-resource languages — standard weight
        +train_ds_meta.spanish.manifest_path=/path/to/spanish_train.json \
        +train_ds_meta.spanish.audio_dir=/path/to/spanish_audio \
        +train_ds_meta.spanish.feature_dir=/path/to/spanish_audio \
        +train_ds_meta.spanish.sample_weight=1.0 \
        "+train_ds_meta.spanish.tokenizer_names=[chartokenizer]" \
        +train_ds_meta.french.manifest_path=/path/to/french_train.json \
        +train_ds_meta.french.audio_dir=/path/to/french_audio \
        +train_ds_meta.french.feature_dir=/path/to/french_audio \
        +train_ds_meta.french.sample_weight=1.0 \
        "+train_ds_meta.french.tokenizer_names=[chartokenizer]" \
        # Low-resource language — upsampled 10x
        +train_ds_meta.low_resource_lang.manifest_path=/path/to/low_resource_train.json \
        +train_ds_meta.low_resource_lang.audio_dir=/path/to/low_resource_audio \
        +train_ds_meta.low_resource_lang.feature_dir=/path/to/low_resource_audio \
        +train_ds_meta.low_resource_lang.sample_weight=10.0 \
        "+train_ds_meta.low_resource_lang.tokenizer_names=[chartokenizer]"

The ``model.load_cached_codes_if_available=true`` flag skips re-computing audio codes at training time when they are already stored in the manifest. This can substantially reduce data loading overhead when audio codes are pre-extracted.


Preference Optimization After Finetuning
#########################################

After supervised finetuning, you can further improve output quality with GRPO (Group Relative Policy Optimization). GRPO generates multiple candidate outputs per item online, scores them with automatic metrics (CER, SSIM, PESQ), and trains the model to prefer higher-scoring outputs.

To run GRPO starting from a finetuned checkpoint:

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-name=magpietts_multilingual_v1 \
        +init_from_ptl_ckpt=/path/to/finetuned.ckpt \
        +mode=onlinepo_train \
        +model.loss_type=grpo \
        +model.num_generations_per_item=12 \
        +model.cer_reward_weight=0.45 \
        +model.ssim_reward_weight=0.45 \
        +model.pesq_reward_weight=0.1 \
        +model.use_pesq=true \
        +model.reward_asr_model=whisper \
        model.optim.lr=2e-7 \
        ~model.optim.sched \
        trainer.precision=32 \
        trainer.devices=8 \
        trainer.num_nodes=1

For full GRPO configuration options and the complete pipeline, see :doc:`Magpie-TTS Preference Optimization <magpietts-po>`.


Key Hyperparameter Reference
#############################

.. list-table::
   :widths: 35 25 40
   :header-rows: 1

   * - Parameter
     - Typical Value
     - Notes
   * - ``model.optim.lr``
     - ``5e-6`` (English SFT), ``1e-5`` (multilingual)
     - Much lower than pretraining LR to preserve learned features
   * - ``trainer.max_steps``
     - ``10000``–``15000``
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

