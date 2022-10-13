.. _nlp_model:

Model NLP
=========

The config file for NLP models contain three main sections:

    - ``trainer``: contains the configs for PTL training. For more information, refer to :doc:`../core/core` and `PTL Trainer class API <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api>`.
    - ``exp_manager``: the configs of the experiment manager. For more information, refer to :doc:`../core/core`.
    - ``model``: contains the configs of the datasets, model architecture, tokenizer, optimizer, scheduler, etc.

The following sub-sections of the model section are shared among most of the NLP models.

    - ``tokenizer``: specifies the tokenizer
    - ``language_model``: specifies the underlying model to be used as the encoder
    - ``optim``: the configs of the optimizer and scheduler :doc:`../core/core`

The ``tokenizer`` and ``language_model`` sections have the following parameters:

+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                                  | **Data Type**   |  **Description**                                                                                             |
+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.tokenizer.tokenizer_name**             | string          | Tokenizer name will be filled automatically based on ``model.language_model.pretrained_model_name``.         |
+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.tokenizer.vocab_file**                 | string          | Path to tokenizer vocabulary.                                                                                |
+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.tokenizer.tokenizer_model**            | string          | Path to tokenizer model (only for sentencepiece tokenizer).                                                  |
+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.language_model.pretrained_model_name** | string          | Pre-trained language model name, for example: ``bert-base-cased`` or ``bert-base-uncased``.                  |
+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.language_model.lm_checkpoint**         | string          | Path to the pre-trained language model checkpoint.                                                           |
+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.language_model.config_file**           | string          | Path to the pre-trained language model config file.                                                          |
+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.language_model.config**                | dictionary      | Config of the pre-trained language model.                                                                    |
+------------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

The parameter **model.language_model.pretrained_model_name** can be one of the following:

    - ``megatron-bert-345m-uncased``, ``megatron-bert-345m-cased``, ``biomegatron-bert-345m-uncased``, ``biomegatron-bert-345m-cased``, ``bert-base-uncased``, ``bert-large-uncased``, ``bert-base-cased``, ``bert-large-cased``
    - ``distilbert-base-uncased``, ``distilbert-base-cased``
    - ``roberta-base``, ``roberta-large``, ``distilroberta-base``
    - ``albert-base-v1``, ``albert-large-v1``, ``albert-xlarge-v1``, ``albert-xxlarge-v1``, ``albert-base-v2``, ``albert-large-v2``, ``albert-xlarge-v2``, ``albert-xxlarge-v2``
