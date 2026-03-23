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

Use the ``speech_to_text_finetune.py`` script:

.. code-block:: bash

    python examples/asr/speech_to_text_finetune.py \
        --config-path=<path to config directory> \
        --config-name=<config name> \
        model.train_ds.manifest_filepath=<path to train manifest> \
        model.validation_ds.manifest_filepath=<path to val manifest> \
        trainer.devices=1 \
        trainer.max_epochs=50

The script handles model initialization from a pretrained checkpoint using the ``init_from_nemo_model`` or ``init_from_pretrained_model`` config options.


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


Fine-Tuning with HuggingFace Datasets
---------------------------------------

NeMo supports loading datasets directly from HuggingFace:

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


Execution Flow
--------------

The fine-tuning execution flow for CTC and Transducer models is documented in:

* `CTC Fine-tuning README <https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf/asr_finetune>`_
* `Transducer Fine-tuning README <https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf/asr_finetune>`_


Tips
----

1. **Start with a low learning rate** — fine-tuning with too high a learning rate can destroy pretrained features.
2. **Use Lhotse dataloading** for efficient training with dynamic batching. See :doc:`Lhotse Dataloading </dataloaders>`.
3. **Monitor validation WER** closely — fine-tuning can overfit quickly on small datasets.
4. **Use spec augmentation** during fine-tuning to improve robustness.
5. **For multilingual fine-tuning**, consider using ``AggregateTokenizer`` and the Hybrid model with prompt conditioning.
