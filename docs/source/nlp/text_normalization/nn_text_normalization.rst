.. _nn_text_normalization:

Neural Text Normalization Models
================================
Text normalization is the task of converting a written text into its spoken form. For example,
``$123`` should be verbalized as ``one hundred twenty three dollars``, while ``123 King Ave``
should be verbalized as ``one twenty three King Avenue``. At the same time, the inverse problem
is about converting a spoken sequence (e.g., an ASR output) into its written form.

NeMo has an implementation that allows you to build a neural-based system that is able to do
both text normalization (TN) and also inverse text normalization (ITN). At a high level, the
system consists of two individual components:

- `DuplexTaggerModel <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/nlp/models/duplex_text_normalization/duplex_tagger.py/>`__ - a Transformer-based tagger for identifying "semiotic" spans in the input (e.g., spans that are about times, dates, or monetary amounts).
- `DuplexDecoderModel <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/nlp/models/duplex_text_normalization/duplex_decoder.py/>`__ - a Transformer-based seq2seq model for decoding the semiotic spans into their appropriate forms (e.g., spoken forms for TN and written forms for ITN).

The typical workflow is to first train a DuplexTaggerModel and also a DuplexDecoderModel. An example training script
is provided: `duplex_text_normalization_train.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py>`__.
After that, the two trained models can be used to initialize a `DuplexTextNormalizationModel <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/nlp/models/duplex_text_normalization/duplex_tn.py/>`__ that can be used for end-to-end inference.
An example script for evaluation is provided here: `duplex_text_normalization_test.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/duplex_text_normalization_test.py>`__.
The script for inference of the full pipeline is provided here: `duplex_text_normalization_infer.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/duplex_text_normalization_infer.py>`__.
This script runs inference from a raw text file or an interactive terminal. 
The term *duplex* refers to the fact that our system can be trained to do both TN and ITN. However, you can also specifically train the system for only one of the tasks.


Quick Start Guide
-----------------

To run the pretrained models interactively see :ref:`inference_text_normalization_nn`.

Available models
^^^^^^^^^^^^^^^^

.. list-table:: *Pretrained Models*
   :widths: 5 10
   :header-rows: 1

   * - Model
     - Pretrained Checkpoint
   * - neural_text_normalization_t5
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:neural_text_normalization_t5



Data Format
-----------
Both the DuplexTaggerModel model and the DuplexDecoderModel model use the same text format as the `Google text normalization dataset <https://www.kaggle.com/google-nlu/text-normalization>`__.
The data needs to be stored in TAB separated files (``.tsv``) with three columns.
The first of which is the "semiotic class" (e.g.,  numbers, times, dates) , the second is the token
in written form, and the third is the spoken form. An example sentence in the dataset is shown below.
In the example, ``self`` denotes that the spoken form is the same as the written form. 

.. code::

    PLAIN	The	<self>
    PLAIN	company 's	<self>
    PLAIN	revenues	<self>
    PLAIN	grew	<self>
    PLAIN	four	<self>
    PLAIN	fold	<self>
    PLAIN	between	<self>
    DATE	2005	two thousand five
    PLAIN	and	<self>
    DATE	2008	two thousand eight
    PUNCT	.	<self>
    <eos>	<eos>


More information about the Google text normalization dataset can be found in the paper `RNN Approaches to Text Normalization: A Challenge <https://arxiv.org/ftp/arxiv/papers/1611/1611.00068.pdf>`__ :cite:`nlp-textnorm-sproat2016rnn`.
The script for splitting the Google text normalization data files into `train`, `dev`, `test` can be found here: 
`data/data_split.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/data/data_split.py>`__.

Data preprocessing
------------------

Processing scripts can be found in the same folder. Right now we only provide scripts for English text normalization, see `data/en/data_preprocessing.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/data/en/data_preprocessing.py>`__.
The details can be found at the top of the scripts.
The purpose of the preprocessing scripts is to standardize the format in order to help with model training.
We also changed punctuation class `PUNCT` to be treated like a plain token ( label changed from `<sil> to ``<self>`), since we want to preserve punctuation even after normalization. 
For text normalization it is crucial to avoid unrecoverable errors, which are linguistically coherent and not semantic preserving. 
We noticed that due to data scarcity the model struggles verbalizing long numbers correctly, so we changed the ground truth for long numbers to digit by digit verbalization.
We also ignore certain semiotic classes from neural verbalization, e.g. `ELECTRONIC` or `WHITELIST` -- `VERBATIM` and `LETTER` in the original dataset. Instead we label urls/email addresses and abbreviations as plain tokens, and handle it separately with WFST-based grammars, see :ref:`inference_text_normalization_nn`.
This simplifies the task for the model and significantly reduces unrecoverable errors.


Data upsampling
---------------

Data upsampling is an effective way to increase the training data for better model performance, especially on the long tail of semiotic tokens.
We used upsampling for training an English text normalization model, see `data/en/upsampling.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/data/en/upsampling.py>`__.
Currently this script only upsamples a few classes, that are diverse in semiotic tokens but at the same time underrepresented in the training data.
Of all the input files in `train` folder created by `data/data_split.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/data/data_split.py>`__. this script takes the first file and detects the class patterns that occur in it.
For those that are underrepresented, quantitatively defined as lower than `min_number`, the other files are scanned for sentences that have the missing patterns. 
Those sentences are appended to the first file, which can then be used for training. 
Details can be found at the top of the script.

Tarred Dataset
--------------

When training with ``DistributedDataParallel``, each process has its own copy of the dataset. For large datasets, this may not always
fit in CPU memory. `Webdatasets <https://github.com/tmbdev/webdataset>`__ circumvents this problem by efficiently iterating over
tar files stored on disk. Each tar file can contain hundreds to thousands of pickle files, each containing a single minibatch.

Tarred datasets can be created as follows:

.. code::

    python examples/nlp/duplex_text_normalization/data/create_tarred_dataset.py \
        --input_files = "<trained_processed/output-00099-of-00100>" \
        --input_files = "<trained_processed/output-00098-of-00100>" \
        --batch_size = "<batch size>" \
        --out_dir= "<TARRED_DATA_OUTPUT_DIR>"


.. warning::
  The batch size used for creating the tarred dataset will be the batch size used in training regardless of what the user specifies in the configuration yaml file. 
  The number of shards should be divisible by the world size to ensure an even
  split among workers. If it is not divisible, logging will give a warning but training will proceed, but likely hang at the last epoch.
  

Model Training
--------------

An example training script is provided: `duplex_text_normalization_train.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py>`__.
The config file used for the example is at `duplex_tn_config.yaml <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization/conf/duplex_tn_config.yaml>`__.
You can change any of these parameters directly from the config file or update them with the command-line arguments.

The config file contains three main sections. The first section contains the configs for the tagger, the second section is about the decoder,
and the last section is about the dataset. Most arguments in the example config file are quite self-explanatory (e.g.,
*decoder_model.optim.lr* refers to the learning rate for training the decoder). We have set most of the hyper-parameters to
be the values that we found to be effective (for the English and the Russian subsets of the Google TN dataset).
Some arguments that you may want to modify are:

- *lang*: The language of the dataset.

- *mode*: ``tn``, ``itn`` or ``joint`` for text normalization, inverse text normalization or duplex mode

- *data.train_ds.data_path*: The path to the training file.

- *data.validation_ds.data_path*: The path to the validation file.

- *data.test_ds.data_path*: The path to the test file.

- *data.test_ds.data_path*: The path to the test file.

- *data.test_ds.errors_log_fp*: Path to the file for logging the errors for the test file.

- *tagger_pretrained_model*: pretrained model path or name (optional)

- *decoder_pretrained_model*: pretrained model path or name (optional)

- *tagger_model.nemo_path*: This is the path where the final trained tagger model will be saved to.

- *decoder_model.nemo_path*: This is the path where the final trained decoder model will be saved to.

- *tagger_model.transformer*: The huggingface transformer model used to initialize the tagger model weights 

- *decoder_model.transformer*: The huggingface transformer model used to initialize the decoder model weights 


Example of a training command:

.. code::

    python examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py \
        data.base_dir=<PATH_TO_DATASET_DIR> \
        mode={tn,itn,joint}

There are 3 different modes. "tn" mode is for training a system for TN only.
"itn" mode is for training a system for ITN. "joint" is for training a system
that can do both TN and ITN at the same time. Note that the above command will
first train a tagger and then train a decoder sequentially.

You can also train only a tagger (without training a decoder) by running the
following command:

.. code::

    python examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py \
        data.base_dir=PATH_TO_DATASET_DIR \
        mode={tn,itn,joint} \
        decoder_model.do_training=false

Or you can also train only a decoder (without training a tagger):

.. code::

    python examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py \
        data.base_dir=PATH_TO_DATASET_DIR \
        mode={tn,itn,joint} \
        tagger_model.do_training=false

To use the tarred version of the data with the decoder model, set `data.train_ds.use_tarred_dataset` to `True` and provide \
path to the `metadata.json` file. The metadata file is created during the tarred dataset construction and stored at `<TARRED_DATA_OUTPUT_DIR>`.
To enable training with the tarred dataset, add the following arguments:

.. code::

    data.train_ds.use_tarred_dataset=True \
    data.train_ds.tar_metadata_file=\PATH_TO\<TARRED_DATA_OUTPUT_DIR>\metadata.json

.. _inference_text_normalization_nn:

Model Inference
---------------

Run the full inference pipeline:

.. code-block:: bash

    cd NeMo/examples/nlp/duplex_text_normalization;

    # run inference in interactive mode using pretrained tagger and decoder models
    python duplex_text_normalization_infer.py \
        tagger_pretrained_model=neural_text_normalization_t5 \
        decoder_pretrained_model=neural_text_normalization_t5 \
        inference.from_file=False \
        lang=en \
        mode=tn

To run inference from a file adjust the previous command by

.. code-block:: bash

    inference.from_file=<path_to_file>
    inference.interactive=False

    


This pipeline consists of 
    
* WFST-based grammars to verbalize hard classes, such as urls and abbreviations.
* regex pre-preprocssing of the input, e.g.
    * adding space around `-` in alpha-numerical words, e.g. `2-car` -> `2 - car`
    * converting unicode fraction e.g. ½ to 1/2
    * normalizing greek letters and some special characters, e.g. `+` -> `plus`
* Moses :cite:`nlp-textnorm-koehnetal2007moses` tokenization/preprocessing of the input
* inference with neural tagger and decoder
* Moses postprocessing/ detokenization
* WFST-based grammars to verbalize some `VERBATIM`
* punctuation correction for TTS (to match  the output punctuation to the input form)

Model Architecture
------------------

The tagger model first uses a Transformer encoder (e.g., albert-base-v2) to build a
contextualized representation for each input token. It then uses a classification head
to predict the tag for each token (e.g., if a token should stay the same, its tag should
be ``SAME``). The decoder model then takes the semiotic spans identified by the tagger and
transform them into the appropriate forms (e.g., spoken forms for TN and written forms for ITN).
The decoder model is essentially a Transformer-based encoder-decoder seq2seq model (e.g., the example
training script uses the T5-base model by default). Overall, our design is partly inspired by the
RNN-based sliding window model proposed in the paper
`Neural Models of Text Normalization for Speech Applications <https://research.fb.com/wp-content/uploads/2019/03/Neural-Models-of-Text-Normalization-for-Speech-Applications.pdf>`__ :cite:`nlp-textnorm-zhang2019neural`.

We introduce a simple but effective technique to allow our model to be duplex. Depending on the
task the model is handling, we append the appropriate prefix to the input. For example, suppose
we want to transform the text ``I live in 123 King Ave`` to its spoken form (i.e., TN problem),
then we will simply append the prefix ``tn`` to it and so the final input to our models will actually
be ``tn I live in tn 123 King Ave``. Similarly, for the ITN problem, we just append the prefix ``itn``
to the input.

To improve the effectiveness and robustness of our models, we also experiment with some simple data
augmentation techniques during training.

Data Augmentation for Training DuplexTaggerModel (Set to be False by default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the Google English TN training data, about 93% of the tokens are not in any semiotic span. In other words, the ground-truth tags of most tokens are of trivial types (i.e., ``SAME`` and ``PUNCT``). To alleviate this class imbalance problem,
for each original instance with several semiotic spans, we create a new instance by simply concatenating all the semiotic spans together. For example, considering the following ITN instance:

Original instance: ``[The|SAME] [revenues|SAME] [grew|SAME] [a|SAME] [lot|SAME] [between|SAME] [two|B-TRANSFORM] [thousand|I-TRANSFORM] [two|I-TRANSFORM] [and|SAME] [two|B-TRANSFORM] [thousand|I-TRANSFORM] [five|I-TRANSFORM] [.|PUNCT]``

Augmented instance: ``[two|B-TRANSFORM] [thousand|I-TRANSFORM] [two|I-TRANSFORM] [two|B-TRANSFORM] [thousand|I-TRANSFORM] [five|I-TRANSFORM]``

The argument ``data.train_ds.tagger_data_augmentation`` in the config file controls whether this data augmentation will be enabled or not.

Data Augmentation for Training DuplexDecoderModel (Set to be True by default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since the tagger may not be perfect, the inputs to the decoder may not all be semiotic spans. Therefore, to make the decoder become more robust against the tagger's potential errors,
we train the decoder with not only semiotic spans but also with some other more "noisy" spans. This way even if the tagger makes some errors, there will still be some chance that the
final output is still correct.

The argument ``data.train_ds.decoder_data_augmentation`` in the config file controls whether this data augmentation will be enabled or not.

References
----------

.. bibliography:: tn_itn_all.bib ../nlp_all.bib
    :style: plain
    :labelprefix: NLP-TEXTNORM
    :keyprefix: nlp-textnorm-
