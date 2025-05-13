.. _megatron_finetuning:

NeMo Megatron
=============

Megatron-LM :cite:`nlp-megatron-shoeybi2019megatron` is a large, powerful transformer developed by the Applied Deep Learning Research 
team at NVIDIA. Currently NeMo Megatron supports 3 types of models:

* GPT-style models (decoder only)
* T5/BART-style models (encoder-decoder)
* BERT-style models (encoder only)

.. note::
    We recommend using `NeMo Megatron containers <https://developer.nvidia.com/nemo-megatron-early-access>`_ for pre-training, tuning and running inference with large (1B and above) Megatrons.


Model Parallelism
-----------------

`Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ is a highly optimized and efficient library for training large language models.
With Megatron model parallelism, language models can be trained with billions of weights and then used in NeMo for downstream tasks.

NeMo handles pretrained model parallel checkpoints from Megatron-LM automatically and model parallel models in NeMo have the all 
the same features as other NeMo Models.

.. note::

    Currently, NeMo only supports tensor model parallelism.

Training
^^^^^^^^

All of the necessary logic to train model parallel models in NeMo with PyTorch Lightning is contained in the ``NLPDDPStrategy``. 
The ``NLPDDPStrategy`` subclasses the PyTorch Lightning strategy type ``DDPStrategy``.
See `strategies <https://pytorch-lightning.readthedocs.io/en/latest/extensions/strategy.html>`_ for more information on PyTorch Lightning Strategies

To enable model parallel training in NeMo:

.. code-block:: python

    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

Megatron-LM checkpoints have a specific format. One checkpoint is saved for each model parallel rank:

.. code-block:: bash

    iter_0080000/
    ├── mp_rank_00
    │   └── model_optim_rng.pt
    └── mp_rank_01
        └── model_optim_rng.pt


To start fine-tuning from a Megatron-LM checkpoint, simply pass the path to the Megatron-LM checkpoint 
via the language model config:

.. code-block:: bash 

    model.language_model.lm_checkpoint=/raid/megatron/bert/iter_0080000 \

We also need to input the model configuration. This can be done via json:

.. code-block:: json

    {
    "hidden-size": 1024, 
    "num-attention-heads": 16, 
    "num-layers": 24, 
    "max-seq-length": 512
    }

And input via command line:

.. code-block:: bash

    model.language_model.config_file=/raid/data/megatron/bert/config.json \

Or the model configuration can be input via YAML:

.. code-block:: YAML

    model:
        language_model:
            config:
                hidden_size: 1024
                num_attention_heads: 16
                num_layers: 24
                max_position_embeddings: 512

Additionally, Megatron-LM requires a vocab file:

.. code-block:: bash

    model.tokenizer.vocab_file=/path/to/vocab.txt

If using the Megatron-LM default tokenizer for training BERT the vocab file can be omitted:

.. code-block:: bash

    # uncased model
    model.tokenizer.tokenizer_name=megatron-bert-uncased

.. code-block:: bash

    # cased model 
    model.tokenizer.tokenizer_name=megatron-bert-uncased

Auto-Resume
^^^^^^^^^^^

Resuming training with NeMo experiment manager and PyTorch Lightning works exactly the same as other NeMo models.
While training with PTL, model parallel checkpoint will be saved and loaded properly.

.. code-block:: bash

    checkpoints/
    ├── mp_rank_00
    │   ├── mp_autoresume-last.ckpt
    │   ├── mp_autoresume---val_loss=0.35-epoch=0.ckpt
    │   ├── mp_autoresume---val_loss=0.38-epoch=1.ckpt
    │   └── mp_autoresume---val_loss=0.39-epoch=2.ckpt
    └── mp_rank_01
        ├── mp_autoresume-last.ckpt
        ├── mp_autoresume---val_loss=0.35-epoch=0.ckpt
        ├── mp_autoresume---val_loss=0.38-epoch=1.ckpt
        └── mp_autoresume---val_loss=0.39-epoch=2.ckpt

Save and Restore
^^^^^^^^^^^^^^^^

Model parallel .nemo files behave the same as all other .nemo files. Calling ``.save_to`` will save 
a checkpoint for each model parallel rank inside the .nemo file:

.. code-block:: bash

    text_class_350m
    ├── megatron-bert-uncased_encoder_config.json
    ├── megatron_checkpoint_version.json
    ├── model_config.yaml
    ├── mp_rank_00
    │   └── model_weights.ckpt
    ├── mp_rank_01
    │   └── model_weights.ckpt
    ├── tokenizer_vocab_dict.json
    └── tokenizer.vocab_file

When restoring a model parallel .nemo file, we must pass in the ``Trainer`` as model parallel requires DDP:

.. code-block:: python

    model = TokenClassificationModel.restore_from(cfg.pretrained_model, trainer=trainer)

Evaluation
^^^^^^^^^^

Since model parallel models always require more than one GPU, the ``Trainer`` is needed for evaluation:

.. code-block:: python

    trainer = pl.Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    model = TextClassificationModel.restore_from(cfg.model.nemo_path, trainer=trainer)
    model.setup_test_data(test_data_config=cfg.model.test_ds)

    trainer.test(model=model, ckpt_path=None)

BioMegatron
-----------

BioMegatron has the same network architecture as the Megatron-LM, but is pretrained on a different dataset - PubMed,
a large biomedical text corpus, which achieves better performance in biomedical downstream tasks than the original Megatron-LM.

Examples of using BioMegatron on biomedical downstream tasks can be found at (can be executed with `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_): 
`NeMo/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb>`__ and `NeMo/tutorials/nlp/Token_Classification-BioMegatron.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Token_Classification-BioMegatron.ipynb>`__.


References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-MEGATRON
    :keyprefix: nlp-megatron-
