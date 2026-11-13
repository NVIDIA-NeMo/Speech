.. _nemotron-tts-finetuning:

======================
NemotronTTS Finetuning
======================

Finetuning a pretrained NemotronTTS checkpoint adapts the model to new speakers in a language it already speaks, without training from scratch. The pretrained model has already learned general speech patterns, prosody, and acoustic modeling, so finetuning requires far less data and compute than pretraining. Finetuning initializes a new speaker encoder and trains it together with the rest of the model. This guide covers data preparation and training for one or several new speakers in one model.


Prerequisites
#############

Before finetuning, you will need:

- A pretrained NemotronTTS checkpoint (``.nemo``).
- The audio codec (``.nemo``) and the IPA phoneme tokenizer (``.json``) released with the checkpoint.

  .. TODO: add the download links of the checkpoint, the audio codec and the IPA phoneme tokenizer once the final
     model is released.

- A NeMo Speech version that includes commit ``d2a01114d`` (per-module learning rates with ``model.optim_param_group_rules``, PR #16426), recommended. Older versions silently ignore the key and train every parameter at ``model.optim.lr``; at ``5e-6`` this is the one-learning-rate alternative described under the training choices.
- `eSpeak NG <https://github.com/espeak-ng/espeak-ng>`_ **1.52.0** for data preparation. Other versions produce phonemes the model was not trained on.
- One GPU with 80 GB of VRAM (e.g. A100 or H100 80 GB) is enough for finetuning data under about 50 hours; more GPUs are optional.
- NeMo installed from source or in the NeMo Speech container. See :ref:`installation` for installation instructions.

.. note::

   Languages ``en``, ``hi``, ``de``, ``es``, ``fr``, ``it``, ``pt`` and ``ar`` need an ``ipa`` field in the training manifest (``pt`` is Brazilian Portuguese; eSpeak NG voice ``en-us`` for English, ``pt-br`` for Brazilian Portuguese, otherwise the language code). Languages ``vi``, ``zh``, ``ja`` and ``ko`` are read from the written text, without ``ipa``. The checkpoint's ``ignore_phoneme_languages`` sets this routing.


Dataset Preparation
-------------------

**How much data.** At least one hour of clean speech per speaker; more data mainly improves long and difficult text.

**Short utterances.** If the dataset contains few or no short utterances, you should add around 100–200 short samples to the training data. Each sample should contain only 1–2 words, totaling approximately 3–4 minutes of audio, such as yes, no, thanks, names, numbers, or other short responses commonly used by the product.

**Audio and text checks, before building the manifest.** Build the manifest from the final audio files, so every ``duration`` is measured on the audio that is trained on (the Lhotse Shar script cuts each clip to the manifest's duration).

- Every clip must end with at least ~0.2 s of silence; otherwise fade out the last 5 ms, append 200 ms of zeros and write a new file under a unique name. A model trained on abrupt endings may learn to cut off the last word.
- Write transcripts in spoken form: numbers, symbols and abbreviations written out as they are read. Check how eSpeak NG reads initialisms: if it reads one as a word (for example LED or HVAC) while the speaker spells it, write the letters separately (``L E D``).

**Reference ("context") clips.** The ``context_audio_filepath`` is the reference audio used for voice cloning during training. It should be a different clip of the same speaker as ``audio_filepath``, taken from the training split. A minimum context duration of about 3 seconds and a high speaker similarity (for example ≥ 0.6 with TitaNet) are recommended for best results.

**Manifest.** One NeMo JSONL line per utterance:

.. code-block:: json

    {
      "audio_filepath": "/data/voice/wavs/0001.wav",
      "duration": 2.6,
      "text": "The weather is lovely today.",
      "language": "en",
      "speaker": "| Language:en Dataset:my_voice Speaker:voice1 |",
      "context_audio_filepath": "/data/voice/wavs/0420.wav",
      "context_audio_duration": 5.3,
      "ipa": "ðə wˈɛðɚɹ ɪz lˈʌvli tədˈeɪ"
    }

The ``ipa`` field is the eSpeak NG output for the text, with whitespace collapsed to single spaces. Use the output as written: it has no punctuation other than what eSpeak NG writes itself (for example hyphens in French). Example for English (voice ``en-us``; other languages use their language code):

.. code-block:: console

    $ echo "The weather is lovely today." | espeak-ng -v en-us --ipa -q
    ðə wˈɛðɚɹ ɪz lˈʌvli tədˈeɪ

**Lhotse Shar and codec codes.** Convert the training manifest to Lhotse Shar and pre-compute codec codes with the checkpoint's codec:

.. code-block:: bash

    python scripts/magpietts/create_lhotse_shar_from_nemo_manifest.py \
        --manifest-path /data/voice/train.jsonl --audio-base-dir / --output-dir /data/voice/shar \
        --num-jobs 32 --processing-chunk-size 256 --audio-format flac --shuffle --shuffle-seed 42
    python scripts/magpietts/extend_lhotse_shards_with_audio_codes.py \
        --cuts-dir /data/voice/shar/cuts --target-audio-dir /data/voice/shar/target_audio \
        --context-audio-dir /data/voice/shar/context_audio --output-dir /data/voice/shar \
        --codec-model-name codes_25fps --codec-model-path /path/to/codec.nemo --codec-frame-rate 25 \
        --devices 1 --num-nodes 1 --batch-size 32

The training input config lists the shards: the range ends at the number of shards − 1, written with six digits (``{000000..000012}`` for 13 shards):

.. code-block:: yaml

    - type: group
      weight: 1.0
      tags: {task: tts, lang: en}
      input_cfg:
      - type: lhotse_shar
        weight: 1.0
        shar_path:
          cuts: /data/voice/shar/cuts/cuts.{000000..000012}.jsonl.gz
          target_audio: /data/voice/shar/target_audio/recording.{000000..000012}.tar
          context_audio: /data/voice/shar/context_audio/recording.{000000..000012}.tar
          target_codes: /data/voice/shar/codes_25fps/target_codes/codes.{000000..000012}.tar
          context_codes: /data/voice/shar/codes_25fps/context_codes/codes.{000000..000012}.tar
        tags:
          tokenizer_names: [nemotron_nano_30b]


.. _nemotron-tts-new-speakers:

New Speakers in an Existing Language
####################################

.. TODO: add examples/tts/conf/magpietts/easy_magpietts_finetuning.yaml once the final model is released.

``examples/tts/conf/magpietts/easy_magpietts_finetuning.yaml`` is a standalone config for finetuning: the checkpoint's architecture, Lhotse Shar training data, one validation set, and validation-loss checkpointing with early stopping. Per-run values, the learning rates and their constant schedule are set on the command line (the config keeps the pretraining defaults for these):

.. code-block:: bash

    python examples/tts/easy_magpietts.py \
        --config-name=easy_magpietts_finetuning \
        name=my_voice \
        +pretrained_model=/path/to/pretrained.nemo \
        model.codecmodel_path=/path/to/codec.nemo \
        model.phoneme_tokenizer.tokenizer_path=/path/to/ipa_tokenizer.json \
        model.train_ds.dataset.tts_data.input_cfg=/data/voice/input_cfg.yaml \
        model.train_ds.dataset.tts_data.batch_duration=100 \
        model.validation_ds.dataset.dataset_meta.val.manifest_path=/data/voice/val.jsonl \
        model.validation_ds.dataset.dataset_meta.val.language=en \
        model.use_speaker_encoder=true \
        model.optim.lr=5e-6 \
        '+model.optim_param_group_rules.speaker_encoder.param_names=[speaker_encoder.*]' \
        '+model.optim_param_group_rules.speaker_encoder.lr=5e-5' \
        '~model.optim.sched' \
        trainer.max_steps=20000 \
        trainer.val_check_interval=50 \
        exp_manager.explicit_log_dir=/results/my_voice

``model.validation_ds.dataset.dataset_meta.val.language`` is the speakers' language code (``en`` in this example), the same as the ``language`` field of the manifest.

Key training choices:

- **Two learning rates** (default): ``5e-6`` for the pretrained weights (``model.optim.lr``) and ``5e-5`` for the new speaker encoder: the rule ``speaker_encoder`` of ``model.optim_param_group_rules`` selects the parameters whose names match ``speaker_encoder.*`` and sets their ``lr``. The ``+`` prefix is needed because ``optim_param_group_rules`` is not in the config. The training log then shows two optimizer parameter groups (``Parameter Group 0`` with ``lr: 5e-06``, ``Parameter Group 1`` with ``lr: 5e-05``).
- **One learning rate** (alternative, when per-module learning rates are not available): train the whole model at ``5e-6`` and drop the ``optim_param_group_rules`` overrides. Quality is close to two learning rates, but one-word inputs read slightly worse and speaker similarity can be slightly lower.
- ``+pretrained_model`` loads the pretrained weights. The ``+`` prefix is required because the key is not in the base config.
- ``~model.optim.sched`` removes the learning rate schedule so the LR stays constant during finetuning.
- **New speaker encoder** (``model.use_speaker_encoder=true``): created and randomly initialized; ``+pretrained_model`` loads every other weight (without it, training starts from random weights with no error).


Key Hyperparameter Reference
############################

.. list-table::
   :widths: 40 20 40
   :header-rows: 1

   * - Parameter
     - Value
     - Notes
   * - ``model.optim.lr``
     - ``5e-6``
     - Much lower than pretraining LR to preserve learned features
   * - ``model.optim_param_group_rules.speaker_encoder.lr``
     - ``5e-5``
     - Higher than ``model.optim.lr`` because the speaker encoder is initialized from scratch
   * - ``model.use_speaker_encoder``
     - ``true``
     - New speaker encoder
   * - ``model.train_ds.dataset.tts_data.batch_duration``
     - ``100``
     - In seconds. Adjust based on your GPU memory
   * - ``trainer.max_steps``
     - ``20000`` (upper bound)
     - Early stopping on validation loss ends the run, usually after a few hundred to a few thousand steps; more data runs longer
   * - ``trainer.val_check_interval``
     - ``50``–``500``
     - Fewer steps between validations for small datasets, more for large datasets or several speakers; early stopping after 5 validations without improvement
   * - ``use_cfg`` at inference
     - ``False``
     - Classifier-free guidance off
