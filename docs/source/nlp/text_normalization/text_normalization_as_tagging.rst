.. _text_normalization_as_tagging:

Thutmose Tagger: Single-pass Tagger-based ITN Model
===================================================
Inverse text normalization(ITN) converts text from spoken domain (e.g., an ASR output) into its written form: 

Input: ``on may third we paid one hundred and twenty three dollars``
Output: ``on may 3 we paid $123``

`ThutmoseTaggerModel <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/nlp/models/text_normalization_as_tagging/thutmose_tagger.py/>`__ is a single-pass tagger-based model mapping spoken-domain words to written-domain fragments.
Additionally this model predicts "semiotic" classes of the spoken words (e.g., words belonging to the spans that are about times, dates, or monetary amounts)

The typical workflow is to first prepare the dataset, which requires to find granular alignments between spoken-domain words and written-domain fragments.
An example bash-script for data preparation pipeline is provided: `prepare_dataset_en.sh <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/prepare_dataset_en.sh>`__.
After getting the dataset you can train the model. An example training script is provided: `normalization_as_tagging_train.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_train.py>`__.
The script for inference from a raw text file is provided here: `normalization_as_tagging_infer.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py>`__.
An example bash-script that runs inference and evaluation is provided here: `run_infer.sh <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/run_infer.sh>`__.


Quick Start Guide
-----------------

To run the pretrained models see :ref:`inference_text_normalization_tagging`.

Available models
^^^^^^^^^^^^^^^^

.. list-table:: *Pretrained Models*
   :widths: 5 10
   :header-rows: 1

   * - Model
     - Pretrained Checkpoint
   * - itn_en_thutmose_bert
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:itn_en_thutmose_bert


Initial Data
------------
The initial data from which the dataset is prepared is `Google text normalization dataset <https://www.kaggle.com/google-nlu/text-normalization>`__.
It is stored in TAB separated files (``.tsv``) with three columns.
The first column is the "semiotic class" (e.g.,  numbers, times, dates) , the second is the token
in written form, and the third is the spoken form. An example sentence in the dataset is shown below.
In the example, ``<self>`` denotes that the spoken form is the same as the written form. 

.. code::

    PLAIN	The	<self>
    PLAIN	company	<self>
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


More information about the Google Text Normalization Dataset can be found in the paper `RNN Approaches to Text Normalization: A Challenge <https://arxiv.org/ftp/arxiv/papers/1611/1611.00068.pdf>`__ :cite:`nlp-textnorm-tag-sproat2016rnn`.


Data preprocessing
------------------

Our preprocessing is rather complicated, because we need to find granular alignments for semiotic spans that are aligned at phrase-level in Google Text Normalization Dataset.
Right now we only provide data preparation scripts for English and Russian languages, see `prepare_dataset_en.sh <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/prepare_dataset_en.sh>`__ and `prepare_dataset_ru.sh <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/prepare_dataset_ru.sh>`__.
Data preparation includes running the GIZA++ automatic alignment tool, see `install_requirements.sh <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/install_requirements.sh>`__ for installation details.
The purpose of the preprocessing scripts is to build the training dataset for the tagging model.
The final dataset has a simple 3-column tsv format: 1) input sentence, 2) tags for input words, 3) coordinates of "semiotic" spans if any

.. code::

    this plan was first enacted in nineteen eighty four and continued to be followed for nineteen years    <SELF> <SELF> <SELF> <SELF> <SELF> <SELF> _19 8 4_ <SELF> <SELF> <SELF> <SELF> <SELF> <SELF> _19_ <SELF>    DATE 6 9;CARDINAL 15 16


Model Training
--------------

An example training script is provided: `normalization_as_tagging_train.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_train.py>`__.
The config file used by default is `thutmose_tagger_itn_config.yaml <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/text_normalization_as_tagging/conf/thutmose_tagger_itn_config.yaml>`__.
You can change any of the parameters directly from the config file or update them with the command-line arguments.

Most arguments in the example config file are quite self-explanatory (e.g., *model.optim.lr* refers to the learning rate for training the decoder). We have set most of the hyper-parameters to
be the values that we found to be effective (for the English and the Russian subsets of the Google TN dataset).
Some arguments that you may want to modify are:

- *lang*: The language of the dataset.

- *data.train_ds.data_path*: The path to the training file.

- *data.validation_ds.data_path*: The path to the validation file.

- *model.language_model.pretrained_model_name*: The huggingface transformer model used to initialize the model weights

- *model.label_map*: The path/.../label_map.txt. This is the dictionary of possible output tags that model may produce.

- *model.semiotic_classes*: The path/to/.../semiotic_classes.txt. This is the list of possible semiotic classes.


Example of a training command:

.. code::

    python examples/nlp/text_normalization_as_tagging/normalization_as_tagging_train.py \  
        lang=en \
        data.validation_ds.data_path=<PATH_TO_DATASET_DIR>/valid.tsv \
        data.train_ds.data_path=<PATH_TO_DATASET_DIR>/train.tsv \
        model.language_model.pretrained_model_name=bert-base-uncased \
        model.label_map=<PATH_TO_DATASET_DIR>/label_map.txt \
        model.semiotic_classes=<PATH_TO_DATASET_DIR>/semiotic_classes.txt \
        trainer.max_epochs=5



.. _inference_text_normalization_tagging:

Model Inference
---------------

Run the inference:

.. code::

    python examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
        pretrained_model=itn_en_thutmose_bert \
        inference.from_file=./test_sent.txt \
        inference.out_file=./output.tsv

The output tsv file consists of 5 columns:

    * Final output text - it is generated from predicted tags after some simple post-processing.
    * Input text.
    * Sequence of predicted tags - one tag for each input word.
    * Sequence of tags after post-processing (some swaps may be applied).
    * Sequence of predicted semiotic classes - one class for each input word.


Model Architecture
------------------

The model first uses a Transformer encoder (e.g., bert-base-uncased) to build a
contextualized representation for each input token. It then uses a classification head
to predict the tag for each token. Another classification head is used to predict a "semiotic" class label for each token.

Overall, our design is partly inspired by the LaserTagger approach proposed in the paper
`Encode, tag, realize: High-precision text editing <https://arxiv.org/abs/1909.01187>`__ :cite:`nlp-textnorm-tag-malmi2019encode`.

The LaserTagger method is not directly applicable to ITN because it can only regard the whole non-common fragment as a single
replacement tag, whereas spoken-to-written conversion, e.g. a date, needs to be aligned on a more granular level. Otherwise,
the tag vocabulary should include all possible numbers, dates etc. which is impossible. For example, given an example pair "over
four hundred thousand fish" - "over 400,000 fish", LaserTagger will need a single replacement "400,000" in the tag vocabulary.
To overcome this problem, we use another method of collecting the vocabulary of replacement tags, based on automatic alignment of spoken-domain words to small fragments of
written-domain text along with <SELF> and <DELETE> tags.


References
----------

.. bibliography:: tn_itn_all.bib
    :style: plain
    :labelprefix: NLP-TEXTNORM-TAG
    :keyprefix: nlp-textnorm-tag-
