.. _machine_translation:

Machine Translation Models
==========================
Machine translation is the task of translating text from one language to another. For example, from English to Spanish. Models are 
based on the Transformer sequence-to-sequence architecture :cite:`nlp-machine_translation-vaswani2017attention`.

An example script on how to train the model can be found here: `NeMo/examples/nlp/machine_translation/enc_dec_nmt.py <https://github.com/NVIDIA/NeMo/blob/v1.0.2/examples/nlp/machine_translation/enc_dec_nmt.py>`__.
The default configuration file for the model can be found at: `NeMo/examples/nlp/machine_translation/conf/aayn_base.yaml <https://github.com/NVIDIA/NeMo/blob/v1.0.2/examples/nlp/machine_translation/conf/aayn_base.yaml>`__.

Quick Start Guide
-----------------

.. code-block:: python

    from nemo.collections.nlp.models import MTEncDecModel

    # To get the list of pre-trained models
    MTEncDecModel.list_available_models()

    # Download and load the a pre-trained to translate from English to Spanish
    model = MTEncDecModel.from_pretrained("nmt_en_es_transformer24x6")

    # Translate a sentence or list of sentences
    translations = model.translate(["Hello!"], source_lang="en", target_lang="es")

Available Models
^^^^^^^^^^^^^^^^

.. list-table:: *Pretrained Models*
   :widths: 5 10
   :header-rows: 1

   * - Model
     - Pretrained Checkpoint
   * - *New Checkppoints*
     - 
   * - English -> German
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_de_transformer24x6
   * - German -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_de_en_transformer24x6
   * - English -> Spanish
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_es_transformer24x6
   * - Spanish -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_es_en_transformer24x6
   * - English -> French
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_fr_transformer24x6
   * - French -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_fr_en_transformer24x6
   * - English -> Russian
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_ru_transformer24x6
   * - Russian -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_ru_en_transformer24x6
   * - English -> Chinese
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_zh_transformer24x6
   * - Chinese -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_zh_en_transformer24x6
   * - *Old Checkppoints*
     -
   * - English -> German
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_de_transformer12x2
   * - German -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_de_en_transformer12x2
   * - English -> Spanish
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_es_transformer12x2
   * - Spanish -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_es_en_transformer12x2
   * - English -> French
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_fr_transformer12x2
   * - French -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_fr_en_transformer12x2
   * - English -> Russian
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_ru_transformer6x6
   * - Russian -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_ru_en_transformer6x6
   * - English -> Chinese
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_zh_transformer6x6
   * - Chinese -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_zh_en_transformer6x6

Data Format
-----------

Supervised machine translation models require parallel corpora which comprises many examples of sentences in a source language and 
their corresponding translation in a target language. We use parallel data formatted as separate text files for source and target 
languages where sentences in corresponding files are aligned like in the table below.

.. list-table:: *Parallel Coprus*
   :widths: 10 10
   :header-rows: 1

   * - train.english.txt
     - train.spanish.txt
   * - Hello .
     - Hola .
   * - Thank you .
     - Gracias .
   * - You can now translate from English to Spanish in NeMo .
     - Ahora puedes traducir del inglés al español en NeMo .

It is common practice to apply data cleaning, normalization, and tokenization to the data prior to training a translation model and 
NeMo expects already cleaned, normalized, and tokenized data. The only data pre-processing NeMo does is subword tokenization with BPE 
:cite:`nlp-machine_translation-sennrich2015neural`.

Data Cleaning, Normalization & Tokenization
-------------------------------------------

We recommend applying the following steps to clean, normalize, and tokenize your data. All pre-trained models released, apply these data pre-processing steps.

#. Please take a look at a detailed notebook on best practices to pre-process and clean your datasets - NeMo/tutorials/nlp/Data_Preprocessing_and_Cleaning_for_NMT.ipynb

#. Language ID filtering - This step filters out examples from your training dataset that aren't in the correct language. For example, 
   many datasets contain examples where source and target sentences are in the same language. You can use a pre-trained language ID 
   classifier from `fastText <https://fasttext.cc/docs/en/language-identification.html>`__. Install fastText and then you can then run our script using the 
   ``lid.176.bin`` model downloaded from the fastText website.

   .. code ::

       python NeMo/scripts/neural_machine_translation/filter_langs_nmt.py \
         --input-src train.en \
         --input-tgt train.es \
         --output-src train_lang_filtered.en \
         --output-tgt train_lang_filtered.es \
         --source-lang en \
         --target-lang es \
         --removed-src train_noise.en \
         --removed-tgt train_noise.es \
         --fasttext-model lid.176.bin

#. Length filtering - We filter out sentences from the data that are below a minimum length (1) or exceed a maximum length (250). We 
   also filter out sentences where the ratio between source and target lengths exceeds 1.3 except for English <-> Chinese models.
   `Moses <https://github.com/moses-smt/mosesdecoder>`__ is a statistical machine translation toolkit that contains many useful 
   pre-processing scripts.

   .. code ::

       perl mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.3 train en es train.filter 1 250

#. Data cleaning - While language ID filtering can sometimes help with filtering out noisy sentences that contain too many punctuations, 
   it does not help in cases where the translations are potentially incorrect, disfluent,  or incomplete. We use `bicleaner <https://github.com/bitextor/bicleaner>`__ 
   a tool to identify such sentences. It trains a classifier based on many features included pre-trained language model fluency, word 
   alignment scores from a word-alignment model like `Giza++ <https://github.com/moses-smt/giza-pp>`__ etc. We use their available 
   pre-trained models wherever possible and train models ourselves using their framework for remaining languages. The following script 
   applies a pre-trained bicleaner model to the data and pick sentences that are clean with probability > 0.5.

   .. code ::

       awk '{print "-\t-"}' train.en \
       | paste -d "\t" - train.filter.en train.filter.es \
       | bicleaner-classify - - </path/to/bicleaner.yaml> > train.en-es.bicleaner.score

#. Data deduplication - We use `bifixer <https://github.com/bitextor/bifixer>`__ (which uses xxHash) to hash the source and target 
   sentences based on which we remove duplicate entries from the file. You may want to do something similar to remove training examples 
   that are in the test dataset.

   .. code ::

       cat train.en-es.bicleaner.score \
         | parallel -j 25 --pipe -k -l 30000 python bifixer.py --ignore-segmentation -q - - en es \
         > train.en-es.bifixer.score
    
       awk -F awk -F "\t" '!seen[$6]++' train.en-es.bifixer.score > train.en-es.bifixer.dedup.score

#. Filter out data that bifixer assigns probability < 0.5 to.

   .. code ::

       awk -F "\t" '{ if ($5>0.5) {print $3}}' train.en-es.bifixer.dedup.score > train.cleaned.en
       awk -F "\t" '{ if ($5>0.5) {print $4}}' train.en-es.bifixer.dedup.score > train.cleaned.es

#. Punctuation Normalization - Punctuation, especially things like quotes can be written in different ways.
   It's often useful to normalize the way they appear in text. We use the moses punctuation normalizer on all languages except Chinese.

   .. code ::

       perl mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l es < train.cleaned.es > train.normalized.es
       perl mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < train.cleaned.en > train.normalized.en

   For example:

   .. code ::

       Before - Aquí se encuentran joyerías como Tiffany`s entre negocios tradicionales suizos como la confitería Sprüngli.
       After  - Aquí se encuentran joyerías como Tiffany's entre negocios tradicionales suizos como la confitería Sprüngli.

#. Tokenization and word segmentation for Chinese - Naturally written text often contains punctuation markers like commas, full-stops 
   and apostrophes that are attached to words. Tokenization by just splitting a string on spaces will result in separate token IDs for 
   very similar items like ``NeMo`` and ``NeMo.``. Tokenization splits punctuation from the word to create two separate tokens. In the 
   previous example ``NeMo.`` becomes ``NeMo .`` which when split by space, results in two tokens and addresses the earlier problem. 
   
   For example:

   .. code ::

       Before - Especialmente porque se enfrentará "a Mathieu (Debuchy), Yohan (Cabaye) y Adil (Rami) ", recuerda.
       After  - Especialmente porque se enfrentará " a Mathieu ( Debuchy ) , Yohan ( Cabaye ) y Adil ( Rami ) " , recuerda .

   We use the Moses tokenizer for all languages except Chinese.

   .. code ::

       perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l es -no-escape < train.normalized.es > train.tokenized.es
       perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -no-escape < train.normalized.en > train.tokenized.en

   For languages like Chinese where there is no explicit marker like spaces that separate words, we use `Jieba <https://github.com/fxsjy/jieba>`__ to segment a string into words that are space separated. 
   
   For example:

   .. code ::

       Before - 同时，卫生局认为有必要接种的其他人员，包括公共部门，卫生局将主动联络有关机构取得名单后由卫生中心安排接种。
       After  - 同时 ， 卫生局 认为 有 必要 接种 的 其他 人员 ， 包括 公共部门 ， 卫生局 将 主动 联络 有关 机构 取得 名单 后 由 卫生 中心 安排 接种 。

Training a BPE Tokenization
---------------------------

Byte-pair encoding (BPE) :cite:`nlp-machine_translation-sennrich2015neural` is a sub-word tokenization algorithm that is commonly used 
to reduce the large vocabulary size of datasets by splitting words into frequently occuring sub-words. Currently, Machine translation 
only supports the `YouTokenToMe <https://github.com/VKCOM/YouTokenToMe>`__ BPE tokenizer. One can set the tokenization configuration 
as follows:

+-----------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------+
| **Parameter**                                                   | **Data Type**   |   **Default**  | **Description**                                                                                    |
+-----------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------+
| **model.{encoder_tokenizer,decoder_tokenizer}.tokenizer_name**  | str             | ``yttm``       | BPE library name. Only supports ``yttm`` for now.                                                  |
+-----------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------+
| **model.{encoder_tokenizer,decoder_tokenizer}.tokenizer_model** | str             | ``null``       | Path to an existing YTTM BPE model. If ``null``, will train one from scratch on the provided data. |
+-----------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------+
| **model.{encoder_tokenizer,decoder_tokenizer}.vocab_size**      | int             | ``null``       | Desired vocabulary size after BPE tokenization.                                                    |
+-----------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------+
| **model.{encoder_tokenizer,decoder_tokenizer}.bpe_dropout**     | float           | ``null``       | BPE dropout probability. :cite:`nlp-machine_translation-provilkov2019bpe`.                         |   
+-----------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------+
| **model.{encoder_tokenizer,decoder_tokenizer}.vocab_file**      | str             | ``null``       | Path to pre-computed vocab file if exists.                                                         |
+-----------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------+
| **model.shared_tokenizer**                                      | bool            | ``True``       | Whether to share the tokenizer between the encoder and decoder.                                    |
+-----------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------+


Applying BPE Tokenization, Batching, Bucketing and Padding
----------------------------------------------------------

Given BPE tokenizers, and a cleaned parallel corpus, the following steps are applied to create a `TranslationDataset <https://github.com/NVIDIA/NeMo/blob/v1.0.2/nemo/collections/nlp/data/machine_translation/machine_translation_dataset.py#L64>`__ object.

#. Text to IDs - This performs subword tokenization with the BPE model on an input string and maps it to a sequence of tokens for the 
   source and target text.

#. Bucketing - Sentences vary in length and when creating minibatches, we'd like sentences in them to have roughly the same length to 
   minimize the number of ``<pad>`` tokens and to maximize computational efficiency. This step groups sentences roughly the same length 
   into buckets.

#. Batching and padding - Creates minibatches with a maximum number of tokens specified by ``model.{train_ds,validation_ds,test_ds}.tokens_in_batch`` 
   from buckets and pads, so they can be packed into a tensor.

Datasets can be configured as follows:

+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **Parameter**                                               | **Data Type**   |   **Default**  | **Description**                                                                                                      |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.src_file_name**    | str             | ``null``       | Path to the source language file.                                                                                    |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.tgt_file_name**    | str             | ``null``       | Path to the target language file.                                                                                    |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.tokens_in_batch**  | int             | ``512``        | Maximum number of tokens per minibatch.                                                                              |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.clean**            | bool            | ``true``       | Whether to clean the dataset by discarding examples that are greater than ``max_seq_length``.                        |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.max_seq_length**   | int             | ``512``        | Maximum sequence to be used with the ``clean`` argument above.                                                       |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.shuffle**          | bool            | ``true``       | Whether to shuffle minibatches in the PyTorch DataLoader.                                                            |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.num_samples**      | int             | ``-1``         | Number of samples to use. ``-1`` for the entire dataset.                                                             |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.drop_last**        | bool            | ``false``      | Drop last minibatch if it is not of equal size to the others.                                                        |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.pin_memory**       | bool            | ``false``      | Whether to pin memory in the PyTorch DataLoader.                                                                     |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.num_workers**      | int             | ``8``          | Number of workers for the PyTorch DataLoader.                                                                        |
+-------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------------+


Tarred Datasets for Large Corpora
---------------------------------

When training with ``DistributedDataParallel``, each process has its own copy of the dataset. For large datasets, this may not always 
fit in CPU memory. `Webdatasets <https://github.com/tmbdev/webdataset>`__ circumvents this problem by efficiently iterating over 
tar files stored on disk. Each tar file can contain hundreds to thousands of pickle files, each containing a single minibatch.

We recommend using this method when working with datasets with > 1 million sentence pairs.

Tarred datasets can be configured as follows:

+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **Parameter**                                                         | **Data Type**   |   **Default**  | **Description**                                                                                                |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.use_tarred_dataset**         | bool            | ``false``      | Whether to use tarred datasets.                                                                                |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.tar_files**                  | str             | ``null``       | String specifying path to all tar files. Example with 100 tarfiles ``/path/to/tarfiles._OP_1..100_CL_.tar``.   |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.metadata_file**              | str             | ``null``       | Path to JSON metadata file that contains only a single entry for the total number of batches in the dataset.   |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.lines_per_dataset_fragment** | int             | ``1000000``    | Number of lines to consider for bucketing and padding.                                                         |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.num_batches_per_tarfile**    | int             | ``100``        | Number of batches (pickle files) within each tarfile.                                                          |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.tar_shuffle_n**              | int             | ``100``        | How many samples to look ahead and load to be shuffled.                                                        |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **model.{train_ds,validation_ds,test_ds}.shard_strategy**             | str             | ``scatter``    | How the shards are distributed between multiple workers.                                                       |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+
| **model.preproc_out_dir**                                             | str             | ``null``       | Path to folder that contains processed tar files or directory where new tar files are written.                 |
+-----------------------------------------------------------------------+-----------------+----------------+----------------------------------------------------------------------------------------------------------------+

Tarred datasets can be created in two ways:

#. Using the Hydra config and `training script <https://github.com/NVIDIA/NeMo/blob/v1.0.2/examples/nlp/machine_translation/enc_dec_nmt.py>`__.

   For example:

   .. code ::

       python examples/nlp/machine_translation/enc_dec_nmt.py \
         -cn aayn_base \
         do_training=false \
         model.preproc_out_dir=/path/to/preproc_dir \
         model.train_ds.use_tarred_dataset=true \
         model.train_ds.lines_per_dataset_fragment=1000000 \
         model.train_ds.num_batches_per_tarfile=200 \
         model.train_ds.src_file_name=train.tokenized.en \
         model.train_ds.tgt_file_name=train.tokenized.es \
         model.validation_ds.src_file_name=validation.tokenized.en \
         model.validation_ds.tgt_file_name=validation.tokenized.es \
         model.encoder_tokenizer.vocab_size=32000 \
         model.decoder_tokenizer.vocab_size=32000 \
         ~model.test_ds \
         trainer.devices=[0,1,2,3] \
         trainer.accelerator='gpu' \
         +trainer.fast_dev_run=true \
         exp_manager=null \

   The above script processes the parallel tokenized text files into tarred datasets that are written to ``/path/to/preproc_dir``. Since 
   ``do_training`` is set to ``False``, the above script only creates tarred datasets and then exits. If ``do_training`` is set ``True``, 
   then one of two things happen:

   (a) If no tar files are present in ``model.preproc_out_dir``, the script first creates those files and then commences training. 
   (b) If tar files are already present in ``model.preproc_out_dir``, the script starts training from the provided tar files.

#. Using a separate script without Hydra. 

   Tarred datasets for parallel corpora can also be created with a script that doesn't require specifying a configs via Hydra and 
   just uses Python argparse.

   For example:

   .. code ::

       python examples/nlp/machine_translation/create_tarred_parallel_dataset.py \
         --shared_tokenizer \
         --clean \
         --bpe_dropout 0.1 \
         --src_fname train.tokenized.en \
         --tgt_fname train.tokenized.es \
         --out_dir /path/to/preproc_dir \
         --vocab_size 32000 \
         --max_seq_length 512 \
         --min_seq_length 1 \
         --tokens_in_batch 8192 \
         --lines_per_dataset_fragment 1000000 \
        --num_batches_per_tarfile 200

  You can then set `model.preproc_out_dir=/path/to/preproc_dir` and `model.train_ds.use_tarred_dataset=true` to train with this data.

Model Configuration and Training
--------------------------------

The overall model consists of an encoder, decoder, and classification head. Encoders and decoders have the following configuration 
options:

+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **Parameter**                                                     | **Data Type**   |   **Default**         | **Description**                                                                                                 |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.max_sequence_length**                   | int             | ``512``               | Maximum sequence length of positional encodings.                                                                |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.embedding_dropout**                     | float           | ``0.1``               | Path to JSON metadata file that contains only a single entry for the total number of batches in the dataset.    |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.learn_positional_encodings**            | bool            | ``false``             | If ``True``, this is a regular learnable embedding layer. If ``False``, fixes position encodings to sinusoidal. |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.hidden_size**                           | int             | ``512``               | Size of the transformer hidden states.                                                                          |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.num_layers**                            | int             | ``6``                 | Number of transformer layers.                                                                                   |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.inner_size**                            | int             | ``2048``              | Size of the hidden states within the feedforward layers.                                                        |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.num_attention_heads**                   | int             | ``8``                 | Number of attention heads.                                                                                      |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.ffn_dropout**                           | float           | ``0.1``               | Dropout probability within the feedforward layers.                                                              |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.attn_score_dropout**                    | float           | ``0.1``               | Dropout probability of the attention scores before softmax normalization.                                       |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.attn_layer_dropout**                    | float           | ``0.1``               | Dropout probability of the attention query, key, and value projection activations.                              |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.hidden_act**                            | str             | ``relu``              | Activation function throughout the network.                                                                     |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.mask_future**                           | bool            | ``false``, ``true``   | Whether to mask future timesteps for attention. Defaults to ``True`` for decoder and ``False`` for encoder.     |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+
| **model.{encoder,decoder}.pre_ln**                                | bool            | ``false``             | Whether to apply layer-normalization before (``true``) or after (``false``) a sub-layer.                        |
+-------------------------------------------------------------------+-----------------+-----------------------+-----------------------------------------------------------------------------------------------------------------+

Our pre-trained models are optimized with Adam, with a maximum learning of 0.0004, beta of (0.9, 0.98), and inverse square root learning 
rate schedule from :cite:`nlp-machine_translation-vaswani2017attention`. The **model.optim** section sets the optimization parameters.

The following script creates tarred datasets based on the provided parallel corpus and trains a model based on the ``base`` configuration 
from :cite:`nlp-machine_translation-vaswani2017attention`.

.. code ::

    python examples/nlp/machine_translation/enc_dec_nmt.py \
      -cn aayn_base \
      do_training=true \
      trainer.devices=8 \
      trainer.accelerator='gpu' \
      ~trainer.max_epochs \
      +trainer.max_steps=100000 \
      +trainer.val_check_interval=1000 \
      +exp_manager.exp_dir=/path/to/store/results \
      +exp_manager.create_checkpoint_callback=True \
      +exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
      +exp_manager.checkpoint_callback_params.mode=max \
      +exp_manager.checkpoint_callback_params.save_top_k=5 \
      model.preproc_out_dir=/path/to/preproc_dir \
      model.train_ds.use_tarred_dataset=true \
      model.train_ds.lines_per_dataset_fragment=1000000 \
      model.train_ds.num_batches_per_tarfile=200 \
      model.train_ds.src_file_name=train.tokenized.en \
      model.train_ds.tgt_file_name=train.tokenized.es \
      model.validation_ds.src_file_name=validation.tokenized.en \
      model.validation_ds.tgt_file_name=validation.tokenized.es \
      model.encoder_tokenizer.vocab_size=32000 \
      model.decoder_tokenizer.vocab_size=32000 \
      ~model.test_ds \

The trainer keeps track of the sacreBLEU score :cite:`nlp-machine_translation-post2018call` on the provided validation set and saves 
the checkpoints that have the top 5 (by default) sacreBLEU scores.

At the end of training, a ``.nemo`` file is written to the result directory which allows to run inference on a test set.

Multi-Validation
----------------

To run validation on multiple datasets, specify ``validation_ds.src_file_name`` and ``validation_ds.tgt_file_name`` with a list of file paths:

.. code-block:: bash

  model.validation_ds.src_file_name=[/data/wmt13-en-de.src,/data/wmt14-en-de.src] \
  model.validation_ds.tgt_file_name=[/data/wmt13-en-de.ref,/data/wmt14-en-de.ref] \

When using ``val_loss`` or ``val_sacreBLEU`` for the ``exp_manager.checkpoint_callback_params.monitor`` 
then the 0th indexed dataset will be used as the monitor. 

To use other indexes, append the index:

.. code-block:: bash

    exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU_dl_index_1
  
Multiple test datasets work exactly the same way as validation datasets, simply replace ``validation_ds`` by ``test_ds`` in the above examples.

Bottleneck Models and Latent Variable Models (VAE, MIM)
-------------------------------------------------------

NMT with bottleneck encoder architecture is also supported (i.e., fixed size bottleneck), along with the training of Latent Variable Models (currently VAE, and MIM).

1. Supported  learning frameworks (**model.model_type**):
    * NLL - Conditional cross entropy (the usual NMT loss)
    * VAE - Variational Auto-Encoder (`paper <https://arxiv.org/pdf/1312.6114.pdf>`__)
    * MIM - Mutual Information Machine (`paper <https://arxiv.org/pdf/2003.02645.pdf>`__)
2. Supported encoder architectures (**model.encoder.arch**):
    * seq2seq - the usual transformer encoder without a bottleneck
    * bridge - attention bridge bottleneck (`paper <https://arxiv.org/pdf/1703.03130.pdf>`__)
    * perceiver -  Perceiver bottleneck (`paper <https://arxiv.org/pdf/2103.03206.pdf>`__)


+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **Parameter**                          | **Data Type**  | **Default**  | **Description**                                                                                       |
+========================================+================+==============+=======================================================================================================+
| **model.model_type**                   | str            | ``nll``      | Learning (i.e., loss) type: nll (i.e., cross-entropy/auto-encoder), mim, vae (see description above)  |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **model.min_logv**                     | float          | ``-6``       | Minimal allowed log variance for mim                                                                  |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **model.latent_size**                  | int            | ``-1``       | Dimension of latent (projected from hidden) -1 will take value of hidden size                         |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **model. non_recon_warmup_batches**    | bool           | ``200000``   | Warm-up steps for mim, and vae losses (anneals non-reconstruction part)                               |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **model. recon_per_token**             | bool           | ``true``     | When false reconstruction is computed per sample, not per token                                       |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **model.encoder.arch**                 | str            | ``seq2seq``  | Supported architectures: ``seq2seq``, ``bridge``, ``perceiver`` (see description above).              |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **model.encoder.hidden_steps**         | int            | ``32``       | Fixed number of hidden steps                                                                          |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **model.encoder.hidden_blocks**        | int            | ``1``        | Number of repeat blocks (see classes for description)                                                 |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+
| **model.encoder. hidden_init_method**  | str            | ``default``  | See classes for available values                                                                      |
+----------------------------------------+----------------+--------------+-------------------------------------------------------------------------------------------------------+


Detailed description of config parameters:

* **model.encoder.arch=seq2seq**
    * *model.encoder.hidden_steps is ignored*
    * *model.encoder.hidden_blocks is ignored*
    * *model.encoder.hidden_init_method is ignored*
* **model.encoder.arch=bridge**
    * *model.encoder.hidden_steps:* input is projected to the specified fixed steps
    * *model.encoder.hidden_blocks:* number of encoder blocks to repeat after attention bridge projection
    * *model.encoder.hidden_init_method:*
         *  enc_shared (default) - apply encoder to inputs, than attention bridge, followed by hidden_blocks number of the same encoder (pre and post encoders share parameters)
         * identity - apply attention bridge to inputs, followed by hidden_blocks number of the same encoder
         * enc - similar to enc_shared but the initial encoder has independent parameters
* **model.encoder.arch=perceiver**
    * *model.encoder.hidden_steps:* input is projected to the specified fixed steps
    * *model.encoder.hidden_blocks:* number of cross-attention + self-attention blocks to repeat after initialization block (all self-attention and cross-attention share parameters)
    * *model.encoder.hidden_init_method:*
         * params (default) - hidden state is initialized with learned parameters followed by cross-attention with independent parameters
         * bridge - hidden state is initialized with an attention bridge


Training requires the use of the following script (instead of ``enc_dec_nmt.py``):

.. code ::

    python -- examples/nlp/machine_translation/enc_dec_nmt-bottleneck.py \
          --config-path=conf \
          --config-name=aayn_bottleneck \
          ...
          model.model_type=nll \
          model.non_recon_warmup_batches=7500 \
          model.encoder.arch=perceiver \
          model.encoder.hidden_steps=32 \
          model.encoder.hidden_blocks=2 \
          model.encoder.hidden_init_method=params \
          ...


Model Inference
---------------

To generate translations on a test set and compute sacreBLEU scores, run the inference script:

.. code ::

    python examples/nlp/machine_translation/nmt_transformer_infer.py \
      --model /path/to/model.nemo \
      --srctext test.en \
      --tgtout test.en-es.translations \
      --batch_size 128 \
      --source_lang en \
      --target_lang es

The ``--srctext`` file must be provided before tokenization and normalization. The resulting ``--tgtout`` file is detokenized and 
can be used to compute sacreBLEU scores.

.. code ::

    cat test.en-es.translations | sacrebleu test.es

Inference Improvements
----------------------

In practice, there are a few commonly used techniques at inference to improve translation quality. NeMo implements: 

1) Model Ensembling
2) Shallow Fusion decoding with transformer language models :cite:`nlp-machine_translation-gulcehre2015using`
3) Noisy-channel re-ranking :cite:`nlp-machine_translation-yee2019simple`

(a) Model Ensembling - Given many models trained with the same encoder and decoder tokenizer, it is possible to ensemble their predictions (by averaging probabilities at each step) to generate better translations.

.. math::

  P(y_t|y_{<t},x;\theta_{1} \ldots \theta_{k}) = \frac{1}{k} \sum_{i=1}^k P(y_t|y_{<t},x;\theta_{i})


*NOTE*: It is important to make sure that all models being ensembled are trained with the same tokenizer.

The inference script will ensemble all models provided via the `--model` argument as a comma separated string pointing to multiple model paths.

For example, to ensemble three models /path/to/model1.nemo, /path/to/model2.nemo, /path/to/model3.nemo, run:

.. code::

    python examples/nlp/machine_translation/nmt_transformer_infer.py \
      --model /path/to/model1.nemo,/path/to/model2.nemo,/path/to/model3.nemo \
      --srctext test.en \
      --tgtout test.en-es.translations \
      --batch_size 128 \
      --source_lang en \
      --target_lang es

(b) Shallow Fusion Decoding with Transformer Language Models - Given a translation model or an ensemble ot translation models, it possible to combine the scores provided by the translation model(s) and a target-side language model.

At each decoding step, the score for a particular hypothesis on the beam is given by the weighted sum of the translation model log-probabilities and lanuage model log-probabilities.

.. math::
   \mathcal{S}(y_{1\ldots n}|x;\theta_{s \rightarrow t},\theta_{t}) = \mathcal{S}(y_{1\ldots n - 1}|x;\theta_{s \rightarrow t},\theta_{t}) + \log P(y_{n}|y_{<n},x;\theta_{s \rightarrow t}) + \lambda_{sf} \log P(y_{n}|y_{<n};\theta_{t})

Lambda controls the weight assigned to the language model. For now, the only family of language models supported are transformer language models trained in NeMo.

*NOTE*: The transformer language model needs to be trained using the same tokenizer as the decoder tokenizer in the NMT system.

For example, to ensemble three models /path/to/model1.nemo, /path/to/model2.nemo, /path/to/model3.nemo, with shallow fusion using an LM /path/to/lm.nemo

.. code::

    python examples/nlp/machine_translation/nmt_transformer_infer.py \
      --model /path/to/model1.nemo,/path/to/model2.nemo,/path/to/model3.nemo \
      --lm_model /path/to/lm.nemo \
      --fusion_coef 0.05 \
      --srctext test.en \
      --tgtout test.en-es.translations \
      --batch_size 128 \
      --source_lang en \
      --target_lang es

(c) Noisy Channel Re-ranking - Unlike ensembling and shallow fusion, noisy channel re-ranking only re-ranks the final candidates produced by beam search. It does so based on three scores 

1) Forward (source to target) translation model(s) log-probabilities
2) Reverse (target to source) translation model(s) log-probabilities
3) Language Model (target) log-probabilities

.. math::
  \argmax_{i} \mathcal{S}(y_i|x) = \log P(y_i|x;\theta_{s \rightarrow t}^{ens}) + \lambda_{ncr} \big( \log P(x|y_i;\theta_{t \rightarrow s}) + \log P(y_i;\theta_{t}) \big)


To perform noisy-channel re-ranking, first generate a `.scores` file that contains log-proabilities from the forward translation model for each hypothesis on the beam.

.. code::  bash

  python examples/nlp/machine_translation/nmt_transformer_infer.py \
    --model /path/to/model1.nemo,/path/to/model2.nemo,/path/to/model3.nemo \
    --lm_model /path/to/lm.nemo \
    --write_scores \
    --fusion_coef 0.05 \
    --srctext test.en \
    --tgtout test.en-es.translations \
    --batch_size 128 \
    --source_lang en \
    --target_lang es

This will generate a scores file test.en-es.translations.scores, which is provided as input to NeMo/examples/nlp/machine_translation/noisy_channel_reranking.py

This script also requires a reverse (target to source) translation model and a target language model.

.. code:: bash

    python noisy_channel_reranking.py \
        --reverse_model=/path/to/reverse_model1.nemo,/path/to/reverse_model2.nemo \
        --language_model=/path/to/lm.nemo \
        --srctext=test.en-es.translations.scores \
        --tgtout=test-en-es.ncr.translations \
        --forward_model_coef=1.0 \
        --reverse_model_coef=0.7 \
        --target_lm_coef=0.05 \

Pretrained Encoders
-------------------

Pretrained BERT encoders from either `HuggingFace Transformers <https://huggingface.co/models>`__ 
or `Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`__ 
can be used to to train NeMo NMT models.

The ``library`` flag takes values: ``huggingface``, ``megatron``, and ``nemo``.

The ``model_name`` flag is used to indicate a *named* model architecture.
For example, we can use ``bert_base_cased`` from HuggingFace or ``megatron-bert-345m-cased`` from Megatron-LM.

The ``pretrained`` flag indicates whether or not to download the pretrained weights (``pretrained=True``) or 
instantiate the same model architecture with random weights (``pretrained=False``).

To use a custom model architecture from a specific library, use ``model_name=null`` and then add the 
custom configuration under the ``encoder`` configuration.

HuggingFace
^^^^^^^^^^^

We have provided a `HuggingFace config file <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/machine_translation/conf/huggingface.yaml>`__
to use with HuggingFace encoders. 

To use the config file from CLI:

.. code ::

  --config-path=conf \
  --config-name=huggingface \

As an example, we can configure the NeMo NMT encoder to use ``bert-base-cased`` from HuggingFace 
by using the ``huggingface`` config file and setting

.. code ::

  model.encoder.pretrained=true \
  model.encoder.model_name=bert-base-cased \

To use a custom architecture from HuggingFace we can use

.. code ::

  +model.encoder._target_=transformers.BertConfig \
  +model.encoder.hidden_size=1536 \

Note the ``+`` symbol is needed if we're not adding the arguments to the YAML config file.

Megatron
^^^^^^^^

We have provided a `Megatron config file <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/machine_translation/conf/megatron.yaml>`__
to use with Megatron encoders. 

To use the config file from CLI:

.. code ::

  --config-path=conf \
  --config-name=megatron \

The ``checkpoint_file`` should be the path to Megatron-LM checkpoint:

.. code ::

  /path/to/your/megatron/checkpoint/model_optim_rng.pt

In case your megatron model requires model parallelism, then ``checkpoint_file`` should point to the directory containing the
standard Megatron-LM checkpoint format:

.. code ::

  3.9b_bert_no_rng
  ├── mp_rank_00
  │   └── model_optim_rng.pt
  ├── mp_rank_01
  │   └── model_optim_rng.pt
  ├── mp_rank_02
  │   └── model_optim_rng.pt
  └── mp_rank_03
      └── model_optim_rng.pt

As an example, to train a NeMo NMT model with a 3.9B Megatron BERT encoder,
we would use the following encoder configuration:

.. code ::

  model.encoder.checkpoint_file=/path/to/megatron/checkpoint/3.9b_bert_no_rng \
  model.encoder.hidden_size=2560 \
  model.encoder.num_attention_heads=40 \
  model.encoder.num_layers=48 \
  model.encoder.max_position_embeddings=512 \

To train a Megatron 345M BERT, we would use

.. code ::

  model.encoder.model_name=megatron-bert-cased \
  model.encoder.checkpoint_file=/path/to/your/megatron/checkpoint/model_optim_rng.pt \
  model.encoder.hidden_size=1024 \
  model.encoder.num_attention_heads=16 \
  model.encoder.num_layers=24 \
  model.encoder.max_position_embeddings=512 \

If the pretrained megatron model used a custom vocab file, then set:

.. code::

  model.encoder_tokenizer.vocab_file=/path/to/your/megatron/vocab_file.txt
  model.encoder.vocab_file=/path/to/your/megatron/vocab_file.txt


Use ``encoder.model_name=megatron_bert_uncased`` for uncased models with custom vocabularies and
use ``encoder.model_name=megatron_bert_cased`` for cased models with custom vocabularies.


References
----------

.. bibliography:: ../nlp_all.bib
    :style: plain
    :labelprefix: nlp-machine_translation
    :keyprefix: nlp-machine_translation-
