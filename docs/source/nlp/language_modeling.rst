.. _language_modeling:

Language Modeling
=================

A language model (LM) estimates the joint probability of a given text corpus :math:`(x_1,\dots,x_T)` by factorizing it with a chain rule :math:`P(x_1,\dots,x_T) = \prod_{t=1}^T P(x_t|x_1,\dots,x_{t-1})` and sequentially modeling each conditional term in the product. To simplify modeling, it is often assumed that the context size (a number of preceding words) necessary to predict each word :math:`x_t` in the corpus is limited to :math:`N:\;P(x_t|x_1,\dots,x_{t-1}) \approx P(x_t|x_{t-N},\dots,x_{t-1})`. This approximation is commonly referred to as N-gram LM.

Currently, we mainly support sentence-level LMs which do not consider long-term dependencies and model all sentences independently of each other. Our models are based on the Transformer sequence-to-sequence architecture :cite:`nlp-language_modeling-vaswani2017attention`.

| An example script on how to train the model can be found here: `NeMo/examples/nlp/language_modeling/transformer_lm.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/language_modeling/transformer_lm.py>`_.
| The default configuration file for the model can be found at: `NeMo/examples/nlp/language_modeling/conf/transformer_lm_config.yaml <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/language_modeling/conf/transformer_lm_config.yaml>`_.


Data Format
-----------

Unsupervised LMs require the corpus which comprises many examples of sentences from a particular domain (Wikipedia, news, Pubmed abstracts, etc). We assume that the data is formatted as a text file where each line corresponds to a separate sentence:

.. list-table::
   :widths: 100
   :header-rows: 1

   * - Sentence-level LM coprus
   * - in a silver cake basket as the panins had at their party
   * - let us pretermit that long comparison
   * - poverty contempt and sickness treading on my heels i easily resolve not to be affrighted
   
It is common practice to apply data cleaning, normalization, and tokenization to the data prior to training LM and 
NeMo expects already cleaned, normalized, and tokenized data. The only data pre-processing NeMo does is subword tokenization with BPE :cite:`nlp-language_modeling-sennrich2015neural`.

.. note::
    If LM is intended to be used in a conjunction with another model (e.g. :ref:`re-scoring of ASR <neural_rescoring>`, shallow fusion with NMT), make sure that the training data is preprocessed accordingly (lower-case no punctuation for ASR, Moses tokenization/normalization for NMT). Otherwise, it might introduce inadequate LM scores.


Tokenizer Training
------------------

Our LMs support all tokenizers available in NeMo, but require special beginning-of-string ``<bos>`` and end-of-string ``<eos>`` tokens.

Below is the example of training `YouTokenToMe <https://github.com/VKCOM/YouTokenToMe>`__ BPE tokenizer:

.. code-block:: python

    import youtokentome as yttm
    data = # string, path to file with training data
    model = # string, path to where the trained model will be saved
    vocab_size = # int, number of tokens in the final vocabulary
    yttm.BPE.train(data, model, vocab_size)


Sentence Dataset Construction
-----------------------------

Given BPE tokenizer and a cleaned sentence-level text corpus, the following steps are applied to create a `SentenceDataset <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/nlp/data/language_modeling/sentence_dataset.py#L34>`__ object.

#. Text to IDs - Performs tokenization with the specified tokenizer model on an input sentence and maps it to a sequence of tokens.

#. Bucketing - Sentences vary in length and when creating minibatches, we'd like sentences in them to have roughly the same length to minimize the number of ``<pad>`` tokens and to maximize computational efficiency. This step groups sentences of roughly the same length into buckets.

#. Batching and padding - Creates minibatches with a maximum number of tokens specified by ``model.{train_ds,validation_ds,test_ds}.tokens_in_batch`` from buckets and pads, so they can be packed into a tensor.

To use ``SentenceDataset``, specify path to the training data in ``file_name`` in the experiment config file. Below is the list of all available configuration options:

+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **Parameter**                                               | **Data Type**   |   **Default**  | **Description**                                                                                                      |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.file_name**        | str             | ``null``       | Path to the file with sentences.                                                                                     |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.tokens_in_batch**  | int             | ``512``        | Maximum number of tokens per minibatch.                                                                              |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.max_seq_length**   | int             | ``512``        | Maximum sequence length, to be used with the ``clean`` argument below.                                               |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.clean**            | bool            | ``true``       | Whether to clean the dataset by discarding examples that are greater than ``max_seq_length``.                        |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.shuffle**          | bool            | ``true``       | Whether to shuffle minibatches in the PyTorch DataLoader.                                                            |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.num_samples**      | int             | ``-1``         | Number of samples to use. ``-1`` for the entire dataset.                                                             |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.pin_memory**       | bool            | ``false``      | Whether to pin memory in the PyTorch DataLoader.                                                                     |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.num_workers**      | int             | ``8``          | Number of workers for the PyTorch DataLoader.                                                                        |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+


Model Configuration and Training
--------------------------------

The overall model consists of an encoder and a classification head with the following configuration options:

.. list-table:: *Transformer Encoder Network*
   :widths: 30 5 5 60
   :header-rows: 1

   * - Parameter
     - Data Type
     - Default
     - Description
   * - **model.encoder.max_sequence_length**
     - int
     - ``512``
     - Maximum allowed sequence length.
   * - **model.encoder.learn_positional_encodings**
     - bool
     - ``false``
     - If ``true``, this is a regular learnable embedding layer. If ``false``, fixes position encodings to sinusoidal.
   * - **model.encoder.hidden_size**
     - int
     - ``512``
     - Size of the transformer hidden states.
   * - **model.encoder.num_layers**
     - int
     - ``6``
     - Number of transformer layers.
   * - **model.encoder.inner_size**
     - int
     - ``2048``
     - Size of the hidden states within the feedforward layers.
   * - **model.encoder.num_attention_heads**
     - int
     - ``8``
     - Number of attention heads.
   * - **model.encoder.embedding_dropout**
     - float
     - ``0.1``
     - Dropout probability of the embedding layer.
   * - **model.encoder.ffn_dropout**
     - float
     - ``0.1``
     - Dropout probability within the feedforward layers.
   * - **model.encoder.attn_score_dropout**
     - float
     - ``0.1``
     - Dropout probability of the attention scores before softmax normalization.
   * - **model.encoder.attn_layer_dropout**
     - float
     - ``0.1``
     - Dropout probability of the attention query, key, and value projection activations.
   * - **model.encoder.hidden_act**
     - str
     - ``relu``
     - Activation function throughout the network.
   * - **model.encoder.mask_future**
     - bool
     - ``true``
     - Whether to mask future timesteps for attention. Defaults to ``true`` for the standard left-to-right LM.
   * - **model.encoder.pre_ln**
     - bool
     - ``false``
     - Whether to apply layer-normalization before (``true``) or after (``false``) a sub-layer.

.. list-table:: *Head Network (multilayer perceptron)*
   :widths: 30 5 5 60
   :header-rows: 1

   * - Parameter
     - Data Type
     - Default
     - Description
   * - **model.head.num_layers**
     - int
     - ``1``
     - Number of layers in the head network.
   * - **model.head.activation**
     - str
     - ``relu``
     - Activation function used after each layer.
   * - **model.head.log_softmax**
     - bool
     - ``true``
     - Whether to apply ``log_softmax`` to the final layer output.
   * - **model.head.dropout**
     - float
     - ``0.0``
     - Dropout probability after each layer.  


Our pre-trained models are optimized with Adam, with a maximum learning of 0.001, beta of (0.9, 0.98), and inverse square root learning rate schedule from. The **model.optim** section sets the optimization parameters.

The following script trains 6-layer Transformer LM:

.. code ::

    python examples/nlp/language_modeling/transformer_lm.py \
      -cn transformer_lm_config \
      trainer.devices=2 \
      trainer.accelerator='gpu' \
      +exp_manager.exp_dir=/path/to/store/results \
      +exp_manager.create_checkpoint_callback=True \
      +exp_manager.checkpoint_callback_params.monitor=val_PPL \
      +exp_manager.checkpoint_callback_params.mode=min \
      +exp_manager.checkpoint_callback_params.save_top_k=5 \
      model.train_ds.file_name=/path/to/train.txt \
      model.validation_ds.file_name=/path/to/valid.txt \
      model.tokenizer.tokenizer_model=/path/to/yttm_tokenizer_model

The trainer keeps track of the LM perplexity (PPL) on the provided validation set and saves the checkpoints that have the top 5 (by default) PPL. At the end of training, a ``.nemo`` file is written to the result directory which allows to run inference on a test set.


Tarred Datasets for Large Corpora
---------------------------------

When training with ``DistributedDataParallel``, each process has its own copy of the dataset. For large datasets, this may not always fit in CPU memory. `Webdatasets <https://github.com/tmbdev/webdataset>`__ circumvents this problem by efficiently iterating over tar files stored on disk. Each tar file can contain hundreds to thousands of pickle files, each containing a single minibatch. We recommend using this method when working with the datasets of more than 5 million sentences.

To use an existing ``TarredSentenceDataset`` instead of a non-tarred ``SentenceDataset``, set ``is_tarred: true`` in
the experiment config file. Then, pass in the path to the metadata file in ``metadata_file`` and paths to all of the text tarballs in ``tar_files``, either as a list
of filepaths, e.g. ``['/data/shard1.tar', '/data/shard2.tar']``, or in a single brace-expandable string, e.g.
``'/data/shard_{1..64}.tar'`` or ``'/data/shard__OP_1..64_CL_'`` (recommended, see note below).

.. note::
  For brace expansion, there may be cases where ``{x..y}`` syntax cannot be used due to shell interference. This occurs most commonly 
  inside SLURM scripts. Therefore, we provide a few equivalent replacements. Supported opening braces (equivalent to ``{``) are ``(``, 
  ``[``, ``<`` and the special tag ``_OP_``. Supported closing braces (equivalent to ``}``) are ``)``, ``]``, ``>`` and the special 
  tag ``_CL_``. For SLURM based tasks, we suggest the use of the special tags for ease of use.

Tarred datasets for sentence-level LMs can be created with the following script:

.. code::

   python examples/nlp/machine_translation/create_tarred_monolingual_dataset.py \
     --pkl_file_prefix lm \
     --tokenizer_model /path/to/tokenizer_model \
     --fname /path/to/training_data \
     --out_dir /path/to/tarred_dataset \
     --tokens_in_batch 2048 \
     --num_batches_per_tarfile 250

For example, if your dataset contains 10000 batches, the script above will create 40 tarballs and the output directory will look similar to the following:

.. code::

  /path/to/tarred_dataset
  ├── lm-batches.tokens.2048.1.tar
  ├── lm-batches.tokens.2048.2.tar
  ├── ...
  ├── lm-batches.tokens.2048.40.tar
  └── metadata.json
  
To train the model on this dataset, the following parameters have to be specified in the **model.train_ds** section:

.. code::

  use_tarred_dataset: true
  tar_files: /path/to/tarred_dataset/lm-batches.2048._OP_1..40_CL_
  metadata_fiel: /path/to/tarred_dataset/metadata.json

Below is the full list of available configuration options for ``TarredSentenceDataset``:

.. list-table::
   :widths: 30 5 5 60
   :header-rows: 1

   * - Parameter
     - Data Type
     - Default
     - Description
   * - **model.{train_ds,validation_ds,test_ds}.use_tarred_dataset**
     - bool
     - ``false``
     - Whether to use tarred datasets.
   * - **model.{train_ds,validation_ds,test_ds}.tar_files**
     - str
     - ``null``
     - Path to all tar files. Either a list or a single brace-expandable string.
   * - **model.{train_ds,validation_ds,test_ds}.metadata_file**
     - str
     - ``null``
     - Path to JSON metadata file that contains only a single entry for the total number of batches in the dataset.
   * - **model.{train_ds,validation_ds,test_ds}.tar_shuffle_n**
     - int
     - ``100``
     - How many samples to look ahead and load to be shuffled.
   * - **model.{train_ds,validation_ds,test_ds}.shard_strategy**
     - str
     - ``scatter``
     - How the shards are distributed between multiple workers. Either ``scatter`` (each node gets a unique set of shards) or ``replicate`` (each node gets all of the set of shards available in the tarred dataset).

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: nlp-language_modeling
    :keyprefix: nlp-language_modeling-
