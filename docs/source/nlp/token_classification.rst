.. _token_classification:

Token Classification (Named Entity Recognition) Model
=====================================================

Token Classification model supports named entity recognition (NER) and other token level classification tasks, as long as the data 
follows the format specified below.

We're going to use NER task throughout this section. NER, also referred to as entity chunking, identification or extraction, is the 
task of detecting and classifying key information (entities) in text. In other words, a NER model takes a piece of text as input and 
for each word in the text, the model identifies a category the word belongs to. For example, in a sentence: ``Mary lives in Santa Clara 
and works at NVIDIA``, the model should detect that ``Mary`` is a person, ``Santa Clara`` is a location and ``NVIDIA`` is a company.

Quick Start Guide
-----------------

.. code-block:: python

    from nemo.collections.nlp.models import TokenClassificationModel

    # to get the list of pre-trained models
    TokenClassificationModel.list_available_models()

    # Download and load the pre-trained BERT-based model
    model = TokenClassificationModel.from_pretrained("ner_en_bert")

    # try the model on a few examples
    model.add_predictions(['we bought four shirts from the nvidia gear store in santa clara.', 'NVIDIA is a company.'])

.. note::

    We recommend you try this model in a Jupyter notebook (run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): 
    `NeMo/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb>`__.

    Connect to an instance with a GPU (**Runtime** -> **Change runtime type** -> select **GPU** for the hardware accelerator).

    An example script on how to train the model can be found here: `NeMo/examples/nlp/token_classification/token_classification_train.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/token_classification_train.py>`__.

    An example script on how to run evaluation and inference can be found here: `NeMo/examples/nlp/token_classification/token_classification_evaluate.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/token_classification_evaluate.py>`__.

    The default configuration file for the model can be found here: `NeMo/examples/nlp/token_classification/conf/token_classification_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/token_classification_config.yaml>`__.

.. _dataset_token_classification:

Data Input for Token Classification Model
-----------------------------------------

For pre-training or fine-tuning of the model, the data should be split into 2 files:

- ``text.txt``
- ``labels.txt``

Each line of the ``text.txt`` file contains text sequences, where words are separated with spaces, i.e.: ``[WORD] [SPACE] [WORD] [SPACE] [WORD]``.
The ``labels.txt`` file contains corresponding labels for each word in ``text.txt``, the labels are separated with spaces, i.e.: ``[LABEL] [SPACE] [LABEL] [SPACE] [LABEL]``.
Example of a ``text.txt`` file:

    Jennifer is from New York City .
    She likes ...
    ...

Corresponding ``labels.txt`` file:

    B-PER O O B-LOC I-LOC I-LOC O
    O O ...
    ...

Dataset Conversion
------------------

To convert an `IOB format <https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)>`__ (short for inside, outside, beginning) data to the format required for training, use
`examples/nlp/token_classification/data/import_from_iob_format.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/data/import_from_iob_format.py>`_.

.. code::

    # For conversion from IOB format, for example, for CoNLL-2003 dataset:
    python import_from_iob_format.py --data_file=<PATH/TO/THE/FILE/IN/IOB/FORMAT>

Convert Dataset Required Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :code:`--data_file`: path to the file to convert from IOB to NeMo format

After running the above command, the data directory, where the :code:`--data_file` is stored, should contain :code:`text_*.txt` and :code:`labels_*.txt` files.
The default names for the training and evaluation in the :code:`conf/token_classification_config.yaml` are the following:

.. code::

   .
   |--data_dir
     |-- labels_dev.txt
     |-- labels_train.txt
     |-- text_dev.txt
     |-- text_train.txt


Training The Token Classification model
---------------------------------------

In the Token Classification model, we are jointly training a classifier on top of a pre-trained language model, such as 
`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ :cite:`nlp-ner-devlin2018bert`.
Unless the user provides a pre-trained checkpoint for the language model, the language model is initialized with the pre-trained model 
from `HuggingFace Transformers <https://github.com/huggingface/transformers>`__.

Example of model configuration file for training the model can be found at: `NeMo/examples/nlp/token_classification/conf/token_classification_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/token_classification_config.yaml>`__.

The specification can be roughly grouped into three categories:

- Parameters that describe the training process: **trainer**
- Parameters that describe the datasets: **model.dataset**, **model.train_ds**, **model.validation_ds**
- Parameters that describe the model: **model**

More details about parameters in the spec file can be found below:

+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   | **Description**                                                                                              |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.dataset.data_dir**                    | string      | Path to the data converted to the specified above format.                                                    |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.head.num_fc_layers**                  | integer     | Number of fully connected layers.                                                                            |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.head.fc_dropout**                     | float       | Dropout to apply to the input hidden states.                                                                 |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.head.activation**                     | string      | Activation to use between fully connected layers.                                                            |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.punct_head.use_transrormer_init**     | bool        | Whether to initialize the weights of the classifier head with the same approach used in Transformer.         |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.text_file**                     | string      | Name of the text training file located at ``data_dir``.                                                      |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.labels_file**                   | string      | Name of the labels training file located at ``data_dir``.                                                    |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.num_samples**                   | integer     | Number of samples to use from the training dataset, ``-1`` - to use all.                                     |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.text_file**                   | string      | Name of the text file for evaluation, located at ``data_dir``.                                               |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.labels_file**                 | string      | Name of the labels dev file located at ``data_dir``.                                                         |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.num_samples**                 | integer     | Number of samples to use from the dev set, ``-1`` - to use all.                                              |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

For more information, see :ref:`nlp_model`.

Example of the command for training the model:

.. code::

    python token_classification_train.py \
           model.dataset.data_dir=<PATH_TO_DATA_DIR>  \
           trainer.max_epochs=<NUM_EPOCHS> \
           trainer.devices=[<CHANGE_TO_GPU(s)_YOU_WANT_TO_USE>] \
           trainer.accelerator='gpu'


Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :code:`model.dataset.data_dir`: path to the directory with pre-processed data.

.. note::

    While the arguments are defined in the spec file, if you want to override these parameter definitions in the spec file and 
    experiment with them, use the command-line to define the parameter. For example, the sample spec file mentioned above has 
    :code:`validation_ds.batch_size` set to ``64``. However, if the GPU utilization can be optimized further by
    using a larger batch size, override it to the desired value by adding the field :code:`validation_ds.batch_size=128` from
    the command-line. You can repeat this with any of the parameters defined in the sample spec file.

Inference
---------

An example script on how to run inference can be found at `examples/nlp/token_classification/token_classification_evaluate.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/token_classification_evaluate.py>`_.

To run inference with the pre-trained model, run:

.. code::

    python token_classification_evaluate.py \
           pretrained_model=<PRETRAINED_MODEL>

Required Arguments for Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :code:`pretrained_model`: pretrained Token Classification model from ``list_available_models()`` or path to a ``.nemo`` file. For example, ``ner_en_bert`` or ``your_model.nemo``

Model Evaluation
----------------

An example script on how to evaluate the pre-trained model can be found at `examples/nlp/token_classification/token_classification_evaluate.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/token_classification_evaluate.py>`_.

To start evaluation of the pre-trained model, run:

.. code::

    python token_classification_evaluate.py \
           model.dataset.data_dir=<PATH/TO/DATA/DIR>  \
           pretrained_model=ner_en_bert \
           model.test_ds.text_file=<text_*.txt> \
           model.test_ds.labels_file=<labels_*.txt> \
           model.dataset.max_seq_length=512


Required Arguments
^^^^^^^^^^^^^^^^^^

- :code:`pretrained_model`: pretrained Token Classification model from ``list_available_models()`` or path to a ``.nemo`` file. For example, ``ner_en_bert`` or ``your_model.nemo``
- :code:`model.dataset.data_dir`: path to the directory that containes :code:`model.test_ds.text_file` and :code:`model.test_ds.labels_file`

During evaluation of the :code:`test_ds`, the script generates a classification report that includes the following metrics:

- :code:`Precision`
- :code:`Recall`
- :code:`F1`

For more information, see `here <https://en.wikipedia.org/wiki/Precision_and_recall>`__.

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-NER
    :keyprefix: nlp-ner-
