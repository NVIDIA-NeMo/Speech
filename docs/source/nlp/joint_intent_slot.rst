.. _intent_slot:

Joint Intent and Slot Classification
====================================

Joint Intent and Slot classification is a NLU task for classifying an intent and detecting all
relevant slots (Entities) for the intent in a query. For example, in the query ``What is the weather in Santa Clara tomorrow morning?``,
we would like to classify the query as a ``weather intent``, detect ``Santa Clara`` as a `location slot`,
and ``tomorrow morning`` as a ``date_time slot``. Intent and Slot names are usually task-specific and
defined as labels in the training data. This is a fundamental step that is executed in any
task-driven conversational assistant.

Our BERT-based model implementation allows you to train and detect both of these tasks together.

.. note::

    We recommend you try the Joint Intent and Slot Classification model in a Jupyter notebook (can run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): `NeMo/tutorials/nlp/Joint_Intent_and_Slot_Classification.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Joint_Intent_and_Slot_Classification.ipynb>`__.

    Connect to an instance with a GPU (**Runtime** -> **Change runtime type** -> select **GPU** for the hardware accelerator).

    An example script on how to train the model can be found here: `NeMo/examples/nlp/intent_slot_classification <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/intent_slot_classification>`__.


NeMo Data Format
----------------

When training the model, the dataset should be first converted to the required data format, which requires the following files:

- :code:`dict.intents.csv` - A list of all intent names in the data. One line per an intent name. The index of the intent line
  (starting from ``0``) is used to identify the appropriate intent in ``train.tsv`` and ``test.tsv`` files.

.. code::

    weather
    alarm
    meeting
    ...

- :code:`dict.slots.csv` - A list of all slot names in the data. One line per slot name. The index of the slot line
  (starting from ``0``) is used to identify the appropriate slots in the queries in ``train_slot.tsv`` and ``test_slot.tsv`` files.
  In the last line of this dictionary ``O`` slot name is used to identify all ``out of scope`` slots, which are usually the majority of the tokens
  in the queries.

.. code::

    date
    time
    city
    ...
    O

- :code:`train.tsv/test.tsv` - A list of original queries, one per line, with the intent number
  separated by a tab (e.g. "what alarms do i have set right now <TAB> 0"). Intent numbers are
  set according to the intent line in the intent dictionary file (:code:`dict.intents.csv`),
  starting from ``0``. The first line in these files should contain the header line ``sentence
  <tab> label``.

- :code:`train_slot.tvs/test_slot.tsv` - A list that contains one line per query, when each word from the original text queries
  is replaced by a token number from the slots dictionary file (``dict.slots.csv``), counted starting from ``0``. All the words 
  which do not contain a relevant slot are replaced by ``out-of scope`` token number, which is also a part of the slot dictionary file,
  usually as the last entry there. For example a line from these files should look similar to: "54 0 0 54 54 12 12" (the numbers are 
  separated by a space). These files do not contain a header line.


Dataset Conversion
------------------

To convert to the format of the model data, use the ``import_datasets`` utility, which implements
the conversion for the Assistant dataset. Download the dataset `here <https://github.com/xliuhw/NLU-Evaluation-Data>`_ or you can 
write your own converter for the format that you are using for data annotation.

For a dataset that follows your own annotation format, we recommend using one text file for all
samples of the same intent, with the name of the file as the name of the intent. Use one line per
query, with brackets to define slot names. This is very similar to the assistant format, and you can
adapt this converter utility or your own format with small changes:

::

    did i set an alarm to [alarm_type : wake up] in the [timeofday : morning]

Run the ``dataset_converter`` command:

.. code::

    python examples/nlp/intent_slot_classification/data/import_datasets.py
        --source_data_dir=`source_data_dir` \
        --target_data_dir=`target_data_dir` \
        --dataset_name=['assistant'|'snips'|'atis']

- :code:`source_data_dir`: the directory location of the your dataset
- :code:`target_data_dir`: the directory location where the converted dataset should be saved
- :code:`dataset_name`: one of the implemented dataset names

After conversion, ``target_data_dir`` should contain the following files:

.. code::

   .
   |--target_data_dir
     |-- dict.intents.csv
     |-- dict.slots.csv
     |-- train.tsv
     |-- train_slots.tsv
     |-- test.tsv
     |-- test_slots.tsv

Model Training
--------------

This is a pretrained BERT based model with 2 linear classifier heads on the top of it, one for classifying an intent of the query and 
another for classifying slots for each token of the query. This model is trained with the combined loss function on the Intent and Slot 
classification task on the given dataset. The model architecture is based on the paper `BERT for Joint Intent Classification and Slot Filling <https://arxiv.org/pdf/1902.10909.pdf>`__:cite:`nlp-jis-chen2019bert`.

For each query, the model classifies it as one the intents from the intent dictionary and for each word of the query it will classify 
it as one of the slots from the slot dictionary, including out of scope slot for all the remaining words in the query which does not 
fall in another slot category. Out of scope slot (``O``) is a part of slot dictionary that the model is trained on.

Example of model configuration file for training the model can be found at: `NeMo/examples/nlp/intent_slot_classification/conf/intent_slot_classification.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/intent_slot_classification/conf/intent_slot_classification_config.yaml>`__.
In the configuration file, define the parameters of the training and the model, although most of the default values will work well.

The specification can be roughly grouped into three categories:

- Parameters that describe the training process: **trainer**
- Parameters that describe the model: **model**
- Parameters that describe the datasets: **model.train_ds**, **model.validation_ds**, **model.test_ds**,

More details about parameters in the spec file can be found below:

+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                                                    | **Description**                                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **model.data_dir**                        | string          | --                                                                               | The path of the data converted to the specified format.                                                      |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **model.class_balancing**                 | string          | ``null``                                                                         | Choose from ``[null, weighted_loss]``. The ``weighted_loss`` enables weighted class balancing of the loss.   |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **model.intent_loss_weight**              | float           | ``0.6``                                                                          | The elation of intent-to-slot loss in the total loss.                                                        |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **model.pad_label**                       | integer         | ``-1``                                                                           | A value to pad the inputs.                                                                                   |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **model.ignore_extra_tokens**             | boolean         | ``false``                                                                        | A flag that specifies whether to ignore extra tokens.                                                        |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **model.ignore_start_end**                | boolean         | ``true``                                                                         | A flag that specifies whether to not use the first and last token for slot training.                         |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **model.head.num_output_layers**          | integer         | ``2``                                                                            | The number of fully connected layers of the classifier on top of the BERT model.                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **model.head.fc_dropout**                 | float           | ``0.1``                                                                          | The dropout ratio of the fully connected layers.                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.prefix**                    | string          | ``train``                                                                        | A prefix for the training file names.                                                                        |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.prefix**                  | string          | ``dev``                                                                          | A prefix for the validation file names.                                                                      |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **test_ds.prefix**                        | string          | ``test``                                                                         | A prefix for the test file names.                                                                            |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+

For additional config parameters common to all NLP models, refer to the `nlp_model doc <https://github.com/NVIDIA/NeMo/blob/stable/docs/source/nlp/nlp_model.rst#model-nlp>`__.

The following is an example of the command for training the model:

.. code::

    python examples/nlp/intent_slot_classification/intent_slot_classification.py
           model.data_dir=<PATH_TO_DATA_DIR> \
           trainer.max_epochs=<NUM_EPOCHS> \
           trainer.devices=[<CHANGE_TO_GPU(s)_YOU_WANT_TO_USE>] \
           trainer.accelerator='gpu'


Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :code:`model.data_dir`: the dataset directory


Optional Arguments
^^^^^^^^^^^^^^^^^^

Most of the default parameters in the existing configuration file are already set appropriately, however, there are some parameters 
you may want to experiment with.

- ``trainer.max_epochs``: the number of training epochs (reasonable to be between 10 to 100)
- ``model.class_balancing``: value ``weighted_loss`` may help to train the model when there is unbalanced set of classes
- ``model.intent_loss_weight``: a number between 0 to 1 that defines a weight of the intent lost versus a slot loss during training. A default value 0.6 gives a slight preference for the intent lose optimization.

Training Procedure
^^^^^^^^^^^^^^^^^^

At the start of evaluation, NeMo will print out a log of the experiment specification, a summary of the training dataset, and the 
model architecture.

As the model starts training, you should see a progress bar per epoch. During training, after each epoch, NeMo will display accuracy 
metrics on the validation dataset for every intent and slot separately, as well as the total accuracy. You can expect these numbers 
to grow up to 50-100 epochs, depending on the size of the trained data. Since this is a joint iIntent and slot training, usually 
intent's accuracy will grow first for the initial 10-20 epochs, and after that, slot's accuracy will start improving as well.

At the end of training, NeMo saves the best checkpoint on the validation dataset at the path specified by the experiment spec file 
before finishing.

.. code::

  GPU available: True, used: True
  TPU available: None, using: 0 TPU cores
  LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2]
  [NeMo W 2021-01-28 14:52:19 exp_manager:299] There was no checkpoint folder at checkpoint_dir :results/checkpoints. Training from scratch.
  [NeMo I 2021-01-28 14:52:19 exp_manager:186] Experiments will be logged at results
  ...
    label                                                precision    recall       f1           support
    weather.weather (label_id: 0)                            0.00       0.00       0.00        128
    weather.temperature (label_id: 1)                        0.00       0.00       0.00          0
    weather.temperature_yes_no (label_id: 2)                 0.00       0.00       0.00          0
    weather.rainfall (label_id: 3)                           0.00       0.00       0.00          0
    weather.rainfall_yes_no (label_id: 4)                    0.00       0.00       0.00          0
    weather.snow (label_id: 5)                               0.00       0.00       0.00          0
    weather.snow_yes_no (label_id: 6)                        0.00       0.00       0.00          0
    weather.humidity (label_id: 7)                           0.00       0.00       0.00          0
    weather.humidity_yes_no (label_id: 8)                    0.00       0.00       0.00          0
    weather.windspeed (label_id: 9)                          0.00       0.00       0.00          0
    weather.sunny (label_id: 10)                             0.00       0.00       0.00          0
    weather.cloudy (label_id: 11)                            0.00       0.00       0.00          0
    weather.alert (label_id: 12)                             0.00       0.00       0.00          0
    context.weather (label_id: 13)                           0.00       0.00       0.00          0
    context.continue (label_id: 14)                          0.00       0.00       0.00          0
    context.navigation (label_id: 15)                        0.00       0.00       0.00          0
    context.rating (label_id: 16)                            0.00       0.00       0.00          0
    context.distance (label_id: 17)                          0.00       0.00       0.00          0
    -------------------
    micro avg                                                0.00       0.00       0.00        128
    macro avg                                                0.00       0.00       0.00        128
    weighted avg                                             0.00       0.00       0.00        128

Model Evaluation and Inference
------------------------------

There is no separate script for the evaluation and inference of this model in NeMo, however, inside of the example file `examples/nlp/intent_slot_classification/intent_slot_classification.py` 
after the training part is finished, you can see the code that evaluates the trained model on an evaluation test set and then an example of doing inference using a list of given queries.

For the deployment in the production environment, refer to `NVIDIA Riva <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html>`__ and `NVIDIA TLT documentation <https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/nlp/index.html>`__.

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-JIS
    :keyprefix: nlp-jis-
