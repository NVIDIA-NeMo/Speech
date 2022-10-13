.. _punctuation_and_capitalization_lexical_audio:

Punctuation and Capitalization Lexical Audio Model
==================================================

Sometimes punctuation and capitalization cannot be restored based only on text. In this case we can use audio to improve model's accuracy.

Like in these examples:

.. code::
  
  Oh yeah? or Oh yeah.

  We need to go? or We need to go.

  Yeah, they make you work. Yeah, over there you walk a lot? or Yeah, they make you work. Yeah, over there you walk a lot.

You can find more details on text only punctuation and capitalization in `Punctuation And Capitalization's page <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/punctuation_and_capitalization.html>`_. In this document, we focus on model changes needed to use acoustic features.

Quick Start Guide
-----------------

.. code-block:: python

    from nemo.collections.nlp.models import PunctuationCapitalizationLexicalAudioModel

    # to get the list of pre-trained models
    PunctuationCapitalizationLexicalAudioModel.list_available_models()

    # Download and load the pre-trained model
    model = PunctuationCapitalizationLexicalAudioModel.from_pretrained("<PATH to .nemo file>")

    # try the model on a few examples
    model.add_punctuation_capitalization(['how are you', 'great how about you'], audio_queries=['/path/to/1.wav', '/path/to/2.wav'], target_sr=16000)

Model Description
-----------------
In addition to `Punctuation And Capitalization model <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/punctuation_and_capitalization.html>`_ we add audio encoder (e.g. Conformer's encoder) and attention based fusion of lexical and audio features.
This model architecture is based on `Multimodal Semi-supervised Learning Framework for Punctuation Prediction in Conversational Speech <https://arxiv.org/pdf/2008.00702.pdf>`__ :cite:`nlp-punct-sunkara20_interspeech`.

.. note::

    An example script on how to train and evaluate the model can be found at: `NeMo/examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py>`__.

    The default configuration file for the model can be found at: `NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml>`__.

    The script for inference can be found at: `NeMo/examples/nlp/token_classification/punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`__.

.. _raw_data_format_punct:

Raw Data Format
---------------
In addition to `Punctuation And Capitalization Raw Data Format <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/punctuation_and_capitalization.html#raw-data-format>`_ this model also requires audio data.
You have to provide ``audio_train.txt`` and ``audio_dev.txt`` (and optionally ``audio_test.txt``) which contain one valid path to audio per row.

Example of the ``audio_train.txt``/``audio_dev.txt`` file:

.. code::

    /path/to/1.wav
    /path/to/2.wav
    ....

In this case ``source_data_dir`` structure should look similar to the following:

.. code::

   .
   |--sourced_data_dir
     |-- dev.txt
     |-- train.txt
     |-- audio_train.txt
     |-- audio_dev.txt

.. _nemo-data-format-label:

Tarred dataset
--------------

It is recommended to use tarred dataset for training with large amount of data (>500 hours) due to large amount of RAM consumed by loading whole audio data into memory and CPU usage.

For creating of tarred dataset with audio you will need data in NeMo format:

.. code::

    python examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py \
        --text <PATH/TO/LOWERCASED/TEXT/WITHOUT/PUNCTUATION> \
        --labels <PATH/TO/LABELS/IN/NEMO/FORMAT> \
        --output_dir <PATH/TO/DIRECTORY/WITH/OUTPUT/TARRED/DATASET> \
        --num_batches_per_tarfile 100 \
        --use_audio \
        --audio_file <PATH/TO/AUDIO/PATHS/FILE> \
        --sample_rate 16000 

.. note::
  You can change sample rate to any positive integer. It will be used in constructor of :class:`~nemo.collections.asr.parts.preprocessing.AudioSegment`. It is recomended to set ``sample_rate`` to the same value as data which was used during training of ASR model.


Training Punctuation and Capitalization Model
---------------------------------------------

The audio encoder is initialized with pretrained ASR model. You can use any of ``list_available_models()`` of ``EncDecCTCModel`` or your own checkpoints, either one should be provided in ``model.audio_encoder.pretrained_model``.
You can freeze audio encoder during training and add additional ``ConformerLayer`` on top of encoder to reduce compute with ``model.audio_encoder.freeze``. You can also add `Adapters <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/core/adapters/components.html>`_ to reduce compute with ``model.audio_encoder.adapter``. Parameters of fusion module are stored in ``model.audio_encoder.fusion``.
An example of a model configuration file for training the model can be found at:
`NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml>`__.

Configs
^^^^^^^^^^^^
.. note::
  This page contains only parameters specific to lexical and audio model. Others parameters can be found in `Punctuation And Capitalization's page <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/punctuation_and_capitalization.html>`_.

Model config
^^^^^^^^^^^^

A configuration of
:class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_lexical_audio_model.PunctuationCapitalizationLexicalAudioModel`
model.

.. list-table:: Model config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **audio_encoder** 
     - :ref:`audio encoder config<audio-encoder-config-label>`
     - :ref:`audio encoder config<audio-encoder-config-label>`
     - A configuration for audio encoder.


Data config
^^^^^^^^^^^

.. list-table:: Location of data configs in parent configs
   :widths: 5 5
   :header-rows: 1

   * - **Parent config**
     - **Keys in parent config**
   * - :ref:`Run config<run-config-label>`
     - ``model.train_ds``, ``model.validation_ds``, ``model.test_ds``
   * - :ref:`Model config<model-config-label>`
     - ``train_ds``, ``validation_ds``, ``test_ds``

.. _regular-dataset-parameters-label:

.. list-table:: Parameters for regular (:class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset`) dataset
   :widths: 5 5 5 30
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **use_audio**
     - bool
     - ``false``
     - If set to ``true`` dataset will return audio as well as text.
   * - **audio_file**
     - string
     - ``null``
     - A path to file with audio paths.
   * - **sample_rate**
     - int
     - ``null``
     - Target sample rate of audios. Can be used for up sampling or down sampling of audio.
   * - **use_bucketing**
     - bool
     - ``true``
     - If set to True will sort samples based on their audio length and assamble batches more efficently (less padding in batch). If set to False dataset will return ``batch_size`` batches instead of ``number_of_tokens`` tokens. 
   * - **preload_audios**
     - bool
     - ``true``
     - If set to True batches will include waveforms, if set to False will store audio_filepaths instead and load audios during ``collate_fn`` call.
    

.. _audio-encoder-config-label:

Audio Encoder config
^^^^^^^^^^^^^^^^^^^^

.. list-table:: Audio Encoder Config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **pretrained_model**
     - string
     - ``stt_en_conformer_ctc_medium``
     - Pretrained model name or path to ``.nemo``` file to take audio encoder from.
   * - **freeze**
     - :ref:`freeze config<freeze-config-label>`
     - :ref:`freeze config<freeze-config-label>`
     - Configuration for freezing audio encoder's weights.
   * - **adapter**
     - :ref:`adapter config<adapter-config-label>`
     - :ref:`adapter config<adapter-config-label>`
     - Configuration for adapter.
   * - **fusion**
     - :ref:`fusion config<fusion-config-label>`
     - :ref:`fusion config<fusion-config-label>`
     - Configuration for fusion.


.. _freeze-config-label:

.. list-table:: Freeze Config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **is_enabled**
     - bool
     - ``false``
     - If set to ``true`` encoder's weights will not be updated during training and aditional ``ConformerLayer`` layers will be added.
   * - **d_model**
     - int
     - ``256``
     - Input dimension of ``MultiheadAttentionMechanism`` and ``PositionwiseFeedForward`` of additional ``ConformerLayer`` layers.
   * - **d_ff**
     - int
     - ``1024``
     - Hidden dimension of ``PositionwiseFeedForward`` of additional ``ConformerLayer`` layers.
   * - **num_layers**
     - int
     - ``4``
     - Number of additional ``ConformerLayer`` layers.


.. _adapter-config-label:

.. list-table:: Adapter Config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **enable**
     - bool
     - ``false``
     - If set to ``true`` will enable adapters for audio encoder.
   * - **config**
     - ``LinearAdapterConfig``
     - ``null``
     - For more details see `nemo.collections.common.parts.LinearAdapterConfig <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/common/parts/adapter_modules.py#L141>`_ class.


.. _fusion-config-label:

.. list-table:: Fusion Config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **num_layers**
     - int
     - ``4``
     - Number of layers to use in fusion.
   * - **num_attention_heads**
     - int
     - ``4``
     - Number of attention heads to use in fusion.
   * - **inner_size**
     - int
     - ``2048``
     - Fusion inner size.



Model training
^^^^^^^^^^^^^^

For more information, refer to the :ref:`nlp_model` section.

To train the model from scratch, run:

.. code::

      python examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py \
             model.train_ds.ds_item=<PATH/TO/TRAIN/DATA_DIR> \
             model.train_ds.text_file=<NAME_OF_TRAIN_INPUT_TEXT_FILE> \
             model.train_ds.labels_file=<NAME_OF_TRAIN_LABELS_FILE> \
             model.validation_ds.ds_item=<PATH/TO/DEV/DATA_DIR> \
             model.validation_ds.text_file=<NAME_OF_DEV_TEXT_FILE> \
             model.validation_ds.labels_file=<NAME_OF_DEV_LABELS_FILE> \
             trainer.devices=[0,1] \
             trainer.accelerator='gpu' \
             optim.name=adam \
             optim.lr=0.0001 \
             model.train_ds.audio_file=<NAME_OF_TRAIN_AUDIO_FILE> \
             model.validation_ds.audio_file=<NAME_OF_DEV_AUDIO_FILE>

The above command will start model training on GPUs 0 and 1 with Adam optimizer and learning rate of 0.0001; and the
trained model is stored in the ``nemo_experiments/Punctuation_and_Capitalization`` folder.

To train from the pre-trained model, run:

.. code::

      python examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py \
             model.train_ds.ds_item=<PATH/TO/TRAIN/DATA_DIR> \
             model.train_ds.text_file=<NAME_OF_TRAIN_INPUT_TEXT_FILE> \
             model.train_ds.labels_file=<NAME_OF_TRAIN_LABELS_FILE> \
             model.validation_ds.ds_item=<PATH/TO/DEV/DATA/DIR> \
             model.validation_ds.text_file=<NAME_OF_DEV_TEXT_FILE> \
             model.validation_ds.labels_file=<NAME_OF_DEV_LABELS_FILE> \
             model.train_ds.audio_file=<NAME_OF_TRAIN_AUDIO_FILE> \
             model.validation_ds.audio_file=<NAME_OF_DEV_AUDIO_FILE> \
             pretrained_model=<PATH/TO/SAVE/.nemo>


.. note::

    All parameters defined in the configuration file can be changed with command arguments. For example, the sample
    config file mentioned above has :code:`train_ds.tokens_in_batch` set to ``2048``. However, if you see that
    the GPU utilization can be optimized further by using a larger batch size, you may override to the desired value
    by adding the field :code:`train_ds.tokens_in_batch=4096` over the command-line. You can repeat this with
    any of the parameters defined in the sample configuration file.

Inference
---------

Inference is performed by a script `examples/nlp/token_classification/punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`_

.. code::

    python punctuate_capitalize_infer.py \
        --input_manifest <PATH/TO/INPUT/MANIFEST> \
        --output_manifest <PATH/TO/OUTPUT/MANIFEST> \
        --pretrained_name <PATH to .nemo file> \
        --max_seq_length 64 \
        --margin 16 \
        --step 8 \
        --use_audio

Long audios are split just like in text only case, audio sequences treated the same as text seqences except :code:`max_seq_length` for audio equals :code:`max_seq_length*4000`.

Model Evaluation
----------------

Model evaluation is performed by the same script
`examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py
<https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py>`_
as training.

Use :ref`config<run-config-lab>` parameter ``do_training=false`` to disable training and parameter ``do_testing=true``
to enable testing. If both parameters ``do_training`` and ``do_testing`` are ``true``, then model is trained and then
tested.

To start evaluation of the pre-trained model, run:

.. code::

    python punctuation_capitalization_lexical_audio_train_evaluate.py \
           +model.do_training=false \
           +model.to_testing=true \
           model.test_ds.ds_item=<PATH/TO/TEST/DATA/DIR>  \
           pretrained_model=<PATH to .nemo file> \
           model.test_ds.text_file=<NAME_OF_TEST_INPUT_TEXT_FILE> \
           model.test_ds.labels_file=<NAME_OF_TEST_LABELS_FILE> \
           model.test_ds.audio_file=<NAME_OF_TEST_AUDIO_FILE>


Required Arguments
^^^^^^^^^^^^^^^^^^

- :code:`pretrained_model`: pretrained Punctuation and Capitalization Lexical Audio model from ``list_available_models()`` or path to a ``.nemo``
  file. For example: ``your_model.nemo``.
- :code:`model.test_ds.ds_item`: path to the directory that contains :code:`model.test_ds.text_file`, :code:`model.test_ds.labels_file` and :code:`model.test_ds.audio_file`

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-PUNCT
    :keyprefix: nlp-punct-

