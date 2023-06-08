.. _spellchecking_asr_customization:

SpellMapper (Spellchecking ASR Customization) Model
=====================================================

`SpellMapper <https://arxiv.org/abs/2306.02317>`__ :cite:`nlp-ner-antonova2023spellmapper` is a non-autoregressive model for postprocessing of ASR output. It gets as input a single ASR hypothesis (text) and a custom vocabulary and predicts which fragments in the ASR hypothesis should be replaced by which custom words/phrases if any. Unlike traditional spellchecking approaches, which aim to correct known words using language models, SpellMapper's goal is to correct highly specific user terms, out-of-vocabulary (OOV) words or spelling variations (e.g., "John Koehn", "Jon Cohen").

This model is an alternative to word boosting/shallow fusion approaches:

- does not require retraining ASR model;
- does not require beam-search/language model (LM);
- can be applied on top of any English ASR model output;

Model Architecture
------------------
Though SpellMapper is based on `BERT <https://arxiv.org/abs/1810.04805>`__ :cite:`nlp-ner-devlin2018bert` architecture, it uses some non-standard tricks that make it different from other BERT-based models:

- ten separators (``[SEP]`` tokens) are used to combine the ASR hypothesis and ten candidate phrases into a single input;
- the model works on character level;
- subword embeddings are concatenated to the embeddings of each character that belongs to this subword;
 
 .. code::

    Example input:   [CLS] a s t r o n o m e r s _ d i d i e _ s o m o n _ a n d _ t r i s t i a n _ g l l o [SEP] d i d i e r _ s a u m o n [SEP] a s t r o n o m i e [SEP] t r i s t a n _ g u i l l o t [SEP] ...
    Input segments:      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0     1 1 1 1 1 1 1 1 1 1 1 1 1 1     2 2 2 2 2 2 2 2 2 2 2     3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3     4      
    Example output:      0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 0     ...

The model calculates logits for each character x 11 labels: 

- ``0`` - character doesn't belong to any candidate,
- ``1..10`` - character belongs to candidate with this id.

At inference average pooling is applied to calculate replacement probability for the whole fragments.

Quick Start Guide
-----------------

We recommend you try this model in a Jupyter notebook (need GPU): 
`NeMo/tutorials/nlp/SpellMapper_English_ASR_Customization.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/SpellMapper_English_ASR_Customization.ipynb>`__.

A pretrained English checkpoint can be found at `HuggingFace <https://huggingface.co/bene-ges/spellmapper_asr_customization_en>`__. 

An example inference pipeline can be found here: `NeMo/examples/nlp/spellchecking_asr_customization/run_infer.sh <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/spellchecking_asr_customization/run_infer.sh>`__.

An example script on how to train the model can be found here: `NeMo/examples/nlp/spellchecking_asr_customization/run_training.sh <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/spellchecking_asr_customization/run_training.sh>`__.

An example script on how to train on large datasets can be found here: `NeMo/examples/nlp/spellchecking_asr_customization/run_training_tarred.sh <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/spellchecking_asr_customization/run_training_tarred.sh>`__.

The default configuration file for the model can be found here: `NeMo/examples/nlp/spellchecking_asr_customization/conf/spellchecking_asr_customization_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/spellchecking_asr_customization/conf/spellchecking_asr_customization_config.yaml>`__.

.. _dataset_spellchecking_asr_customization:

Input/Output Format at Inference stage
--------------------------------------
Here we describe input/output format of the SpellMapper model. 

.. note::

    If you use `inference pipeline <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/spellchecking_asr_customization/run_infer.sh>`__ this format will be hidden inside and you only need to provide an input manifest and user vocabulary and you will get a corrected manifest.

An input line should consist of 4 tab-separated columns:
    1. text of ASR-hypothesis
    2. texts of 10 candidates separated by semicolon
    3. 1-based ids of non-dummy candidates, separated by space
    4. approximate start/end coordinates of non-dummy candidates (correspond to ids in third column)

Example input (in one line):

.. code::

    t h e _ t a r a s i c _ o o r d a _ i s _ a _ p a r t _ o f _ t h e _ a o r t a _ l o c a t e d _ i n _ t h e _ t h o r a x	
    h e p a t i c _ c i r r h o s i s;u r a c i l;c a r d i a c _ a r r e s t;w e a n;a p g a r;p s y c h o m o t o r;t h o r a x;t h o r a c i c _ a o r t a;a v f;b l o c k a d e d
    1 2 6 7 8 9 10
    CUSTOM 6 23;CUSTOM 4 10;CUSTOM 4 15;CUSTOM 56 62;CUSTOM 5 19;CUSTOM 28 31;CUSTOM 39 48

Each line in SpellMapper output is tab-separated and consists of 4 columns:
    1. ASR-hypothesis (same as in input)
    2. 10 candidates separated by semicolon (same as in input)
    3. fragment predictions, separated by semicolon, each prediction is a tuple (start, end, candidate_id, probability)
    4. letter predictions - candidate_id predicted for each letter (this is only for debug purposes)

Example output (in one line):

.. code::

    t h e _ t a r a s i c _ o o r d a _ i s _ a _ p a r t _ o f _ t h e _ a o r t a _ l o c a t e d _ i n _ t h e _ t h o r a x
    h e p a t i c _ c i r r h o s i s;u r a c i l;c a r d i a c _ a r r e s t;w e a n;a p g a r;p s y c h o m o t o r;t h o r a x;t h o r a c i c _ a o r t a;a v f;b l o c k a d e d
    56 62 7 0.99998;4 20 8 0.95181;12 20 8 0.44829;4 17 8 0.99464;12 17 8 0.97645
    8 8 8 0 8 8 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7 7 7 7 7 7    

Training Data Format
--------------------

For training, the data should consist of 5 files:

- ``config.json`` - BERT config
- ``label_map.txt`` - labels from 0 to 10, do not change
- ``semiotic_classes.txt`` - currently there are only two classes: ``PLAIN`` and ``CUSTOM``, do not change
- ``train.tsv`` - training examples
- ``test.tsv`` - validation examples

Note that since all these examples are synthetic, we do not reserve a set for final testing. Instead, we run `inference pipeline <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/spellchecking_asr_customization/run_infer.sh>`__ and compare resulting word error rate (WER) to the WER of baseline ASR output. 

One (non-tarred) training example should consist of 4 tab-separated columns:
    1. text of ASR-hypothesis
    2. texts of 10 candidates separated by semicolon
    3. 1-based ids of correct candidates, separated by space, or 0 if none
    4. start/end coordinates of correct candidates (correspond to ids in third column)

Example (in one line):

.. code::

    a s t r o n o m e r s _ d i d i e _ s o m o n _ a n d _ t r i s t i a n _ g l l o
    d i d i e r _ s a u m o n;a s t r o n o m i e;t r i s t a n _ g u i l l o t;t r i s t e s s e;m o n a d e;c h r i s t i a n;a s t r o n o m e r;s o l o m o n;d i d i d i d i d i;m e r c y
    1 3
    CUSTOM 12 23;CUSTOM 28 41

For data preparation see `this script <https://github.com/bene-ges/nemo_compatible/blob/main/scripts/nlp/en_spellmapper/dataset_preparation/build_training_data.sh>`__


References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-NER
    :keyprefix: nlp-ner-
