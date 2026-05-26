.. _asr-fine-tuning:

===========
Fine-Tuning
===========

This page covers how to fine-tune pretrained ASR models in NeMo.


When to Fine-Tune
-----------------

Fine-tuning is recommended when:

* You have domain-specific data (medical, legal, call center, etc.) and want to improve accuracy on that domain.
* You need to adapt to a new accent, speaking style, or acoustic environment.
* You want to add support for a new language using a pretrained multilingual model.

If you have a large, diverse dataset and want to train from scratch, see :doc:`Configuration Files <./configs>` for full training setup.


Fine-Tuning Script
------------------

Use the ``speech_to_text_finetune.py`` script with the default config at
``examples/asr/conf/asr_finetune/speech_to_text_finetune.yaml``:

.. code-block:: bash

    python examples/asr/speech_to_text_finetune.py \
        --config-path=../conf/asr_finetune \
        --config-name=speech_to_text_finetune \
        init_from_pretrained_model="nvidia/parakeet-tdt-0.6b-v2" \
        model.train_ds.manifest_filepath=/path/to/train_manifest.json \
        model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
        trainer.devices=1 \
        trainer.max_epochs=50

You must specify either ``init_from_pretrained_model`` (NGC/HuggingFace name) or ``init_from_nemo_model`` (local ``.nemo`` path) to load the pretrained weights.


Initialization Options
-----------------------

NeMo supports several ways to initialize a model for fine-tuning:

**From a pretrained model (NGC/HuggingFace):**

.. code-block:: yaml

    init_from_pretrained_model: "nvidia/parakeet-tdt-0.6b-v2"

**From a local .nemo checkpoint:**

.. code-block:: yaml

    init_from_nemo_model: "/path/to/checkpoint.nemo"

**Partial loading (selective layers):**

You can include or exclude specific model components using ``include`` and ``exclude`` lists:

.. code-block:: yaml

    init_from_nemo_model: "/path/to/checkpoint.nemo"
    init_from_nemo_model_include:
      - encoder
      - preprocessor
    init_from_nemo_model_exclude:
      - decoder

This is useful when changing the decoder architecture or tokenizer while keeping the pretrained encoder.


Tokenizer Changes
------------------

**Same tokenizer (same vocabulary):**

No special handling needed — fine-tune directly.

**New tokenizer (different vocabulary):**

When changing the tokenizer (e.g., for a new language or domain), you need to:

1. Provide the new tokenizer directory in the config.
2. Exclude the decoder/joint from initialization (for Transducer models) or exclude the final linear layer (for CTC models).

.. code-block:: yaml

    model:
      tokenizer:
        dir: /path/to/new/tokenizer
        type: bpe

    init_from_nemo_model: "/path/to/pretrained.nemo"
    init_from_nemo_model_exclude:
      - decoder
      - joint

**Enforcing a single language after fine-tuning:**

When fine-tuning a multilingual ``EncDecMultiTaskModel`` (e.g., Canary) on a single language, the model may still exhibit phonetic drift — switching languages mid-utterance at inference time. To enforce a specific language during decoding, explicitly set ``source_lang`` and ``target_lang`` to the same language:

.. code-block:: python

    results = model.transcribe(
        audio=["audio.wav"],
        source_lang="de",
        target_lang="de",
    )

See :ref:`Enforcing a Single Language <asr-enforcing-single-language>` in the Inference documentation for more details.


Fine-Tuning with HuggingFace Datasets
---------------------------------------

NeMo supports loading datasets directly from HuggingFace:

.. note::
   HuggingFace dataset loading is not currently supported with the Lhotse dataloader.

.. code-block:: bash

    python examples/asr/speech_to_text_finetune_with_hf.py \
        --config-path=<path to config directory> \
        --config-name=<config name> \
        model.train_ds.hf_data_cfg.path="mozilla-foundation/common_voice_11_0" \
        model.train_ds.hf_data_cfg.name="en" \
        model.train_ds.hf_data_cfg.split="train" \
        model.validation_ds.hf_data_cfg.path="mozilla-foundation/common_voice_11_0" \
        model.validation_ds.hf_data_cfg.name="en" \
        model.validation_ds.hf_data_cfg.split="validation"


Key Configuration Parameters
-----------------------------

The most important parameters for fine-tuning:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``trainer.max_epochs``
     - Number of fine-tuning epochs (typically 50-100 for domain adaptation)
   * - ``model.optim.lr``
     - Learning rate (use lower than training from scratch, e.g., 1e-4 to 1e-5)
   * - ``model.train_ds.manifest_filepath``     
     - Path to training manifest (NeMo JSON format)
   * - ``model.train_ds.batch_size``
     - Batch size per GPU
   * - ``init_from_pretrained_model``
     - NGC/HF model name to initialize from
   * - ``init_from_nemo_model``
     - Local .nemo file to initialize from

For the complete configuration reference, see :doc:`Configuration Files <./configs>`.


Tips
----

1. **Start with a low learning rate** — fine-tuning with too high a learning rate can destroy pretrained features. Typical fine-tuning LRs are 1e-4 to 1e-5. If your pretrained config uses the Noam (warmup + decay) scheduler, override it with a constant or cosine-annealing schedule to avoid the warmup phase resetting to a high LR.
2. **Use Lhotse dataloading** for efficient training with dynamic batching. See :doc:`Lhotse Dataloading </dataloaders>`.
3. **Use spec augmentation** during fine-tuning to improve robustness. See :ref:`Augmentation Configurations <asr-configs-augmentation-configurations>`.
4. **For multilingual fine-tuning**, use a multilingual tokenizer. NeMo supports two approaches: a **unified multilingual SentencePiece tokenizer** — a single BPE model trained on all target languages (as used by Canary v2/Flash), and an ``AggregateTokenizer`` that combines separate monolingual tokenizers with per-language routing (see :doc:`Configs <./configs>` for the ``agg`` tokenizer setup). For prompt-conditioned multilingual models, see the :ref:`Hybrid model with prompt conditioning <Hybrid-Transducer-CTC-Prompt_model__Config>`.
