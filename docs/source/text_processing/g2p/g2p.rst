.. _g2p:

Grapheme-to-Phoneme Models
==========================

Grapheme-to-phoneme conversion (G2P) is the task of transducing graphemes (i.e., orthographic symbols) to phonemes (i.e., units of the sound system of a language).
For example, for `International_Phonetic_Alphabet (IPA): <https://en.wikipedia.org/wiki/International_Phonetic_Alphabet>`__ ``"Swifts, flushed from chimneys …" → "ˈswɪfts, ˈfɫəʃt ˈfɹəm ˈtʃɪmniz …"``.

Modern text-to-speech (TTS) synthesis models can learn pronunciations from raw text input and its corresponding audio data,
but by relying on grapheme input during training, such models fail to provide a reliable way of correcting wrong pronunciations. As a result, many TTS systems use phonetic input
during training to directly access and correct pronunciations at inference time. G2P systems allow users to enforce the desired pronunciation by providing a phonetic transcript of the input.

G2P models convert out-of-vocabulary words (OOV), e.g. proper names and loaner words, as well as heteronyms in their phonetic form to improve the quality of the syntesized text.

*Heteronyms* represent words that have the same spelling but different pronunciations, e.g., “read” in “I will read the book.” vs. “She read her project last week.”  A single model that can handle OOVs and heteronyms and replace dictionary lookups can significantly simplify and improve the quality of synthesized speech.

We support the following G2P models:

* **ByT5 G2P** a text-to-text model that is based on ByT5 :cite:`g2p--xue2021byt5` neural network model that was originally proposed in :cite:`g2p--vrezavckova2021t5g2p` and :cite:`g2p--zhu2022byt5`.

* **G2P-Conformer** CTC model -  uses a Conformer encoder :cite:`g2p--ggulati2020conformer` followed by a linear decoder; the model is trained with CTC-loss. G2P-Conformer model has about 20 times fewer parameters than the ByT5 model and is a non-autoregressive model that makes it faster during inference.

The models can be trained using words or sentences as input.
If trained with sentence-level input, the models can handle out-of-vocabulary (OOV) and heteronyms along with unambiguous words in a single pass.
See :ref:`Sentence-level Dataset Preparation Pipeline <sentence_level_dataset_pipeline>` on how to label data for G2P model training.

Additionally, we support a purpose-built BERT-based classification model for heteronym disambiguation, see :ref:`this <bert_heteronym_cl>` for details.

Model Training, Evaluation and Inference
----------------------------------------

The section covers both ByT5 and G2P-Conformer models.

The models take input data in `.json` manifest format, and there should be separate training and validation manifests.
Each line of the manifest should be in the following format:

.. code::

  {"text_graphemes": "Swifts, flushed from chimneys.", "text": "ˈswɪfts, ˈfɫəʃt ˈfɹəm ˈtʃɪmniz."}

Manifest fields:

* ``text`` - name of the field in manifest_filepath for ground truth phonemes

* ``text_graphemes`` - name of the field in manifest_filepath for input grapheme text

The models can handle input with and without punctuation marks.

To train ByT5 G2P model and evaluate it after at the end of the training, run:

.. code::

    python examples/text_processing/g2p/g2p_train_and_evaluate.py \
        # (Optional: --config-path=<Path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<Path to manifest file>" \
        model.validation_ds.manifest_filepath="<Path to manifest file>" \
        model.test_ds.manifest_filepath="<Path to manifest file>" \
        trainer.devices=1 \
        do_training=True \
        do_testing=True

Example of the config file: ``NeMo/examples/text_processing/g2p/conf/t5_g2p.yaml``.


To train G2P-Conformer model and evaluate it after at the end of the training, run:

.. code::

    python examples/text_processing/g2p/g2p_train_and_evaluate.py \
        # (Optional: --config-path=<Path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<Path to manifest file>" \
        model.validation_ds.manifest_filepath="<Path to manifest file>" \
        model.test_ds.manifest_filepath="<Path to manifest file>" \
        model.tokenizer.dir=<Path to pretrained tokenizer> \
        model.tokenizer_grapheme.do_lower=False \
        model.tokenizer_grapheme.add_punctuation=True \
        trainer.devices=1 \

        do_training=True \
        do_testing=True

Example of the config file: ``NeMo/examples/text_processing/g2p/conf/g2p_conformer_ctc.yaml``.


To evaluate a pretrained G2P model, run:

.. code::

    python examples/text_processing/g2p/g2p_train_and_evaluate.py \
        # (Optional: --config-path=<Path to dir of configs> --config-name=<name of config without .yaml>) \
        pretrained_model="<Path to .nemo file or pretrained model name from list_available_models()>" \
        model.test_ds.manifest_filepath="<Path to manifest file>" \
        trainer.devices=1 \
        do_training=False \
        do_testing=True

To run inference with a pretrained G2P model, run:

.. code-block::

    python g2p_inference.py \
        pretrained_model=<Path to .nemo file or pretrained model name for G2PModel from list_available_models()>" \
        manifest_filepath="<Path to .json manifest>" \
        output_file="<Path to .json manifest to save prediction>" \
        batch_size=32 \
        num_workers=4 \
        pred_field="pred_text"

Model's predictions will be saved in `pred_field` of the `output_file`.

.. _sentence_level_dataset_pipeline:

Sentence-level Dataset Preparation Pipeline
-------------------------------------------

Here is the overall overview of the data labeling pipeline for sentence-level G2P model training:

    .. image:: images/data_labeling_pipeline.png
        :align: center
        :alt: Data labeling pipeline for sentence-level G2P model training
        :scale: 70%

Here we describe the automatic phoneme-labeling process for generating augmented data. The figure below shows the phoneme-labeling steps to prepare data for sentence-level G2P model training. We first convert known unambiguous words to their phonetic pronunciations with dictionary lookups, e.g. CMU dictionary.
Next, we automatically label heteronyms using a RAD-TTS Aligner :cite:`g2p--badlani2022one`. More details on how to disambiguate heteronyms with a pretrained Aligner model could be found in `NeMo/tutorials/tts/Aligner_Inference_Examples.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/tts/Aligner_Inference_Examples.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/Aligner_Inference_Examples.ipynb>`_.
Finally, we mask-out OOV words with a special masking token, “<unk>” in the figure below (note, we use `model.tokenizer_grapheme.unk_token="҂"` symbol during G2P model training.)
Using this unknown token forces a G2P model to produce the same masking token as a phonetic representation during training. During inference, the model generates phoneme predictions for OOV words without emitting the masking token as long as this token is not included in the grapheme input.



.. _bert_heteronym_cl:

Purpose-built BERT-based classification model for heteronym disambiguation
--------------------------------------------------------------------------

HeteronymClassificationModel is a BERT-based :cite:`g2p--devlin2018bert` model represents a token classification model and can handle multiple heteronyms at once. The model takes a sentence as an input, and then for every word, it selects a heteronym option out of the available forms.
We mask irrelevant forms to disregard the model’s predictions for non-ambiguous words. E.g., given  the input “The Poems are simple to read and easy to comprehend.” the model scores possible {READ_PRESENT and READ_PAST} options for the word “read”.
Possible heteronym forms are extracted from the WikipediaHomographData :cite:`g2p--gorman2018improving`.

The model expects input to be in `.json` manifest format, where is line contains at least the following fields:

.. code::

  {"text_graphemes": "Oxygen is less able to diffuse into the blood, leading to hypoxia.", "start_end": [23, 30], "homograph_span": "diffuse", "word_id": "diffuse_vrb"}

Manifest fields:

* `text_graphemes` - input sentence

* `start_end` - beginning and end of the heteronym span in the input sentence

* `homograph_span` - heteronym word in the sentence

* `word_id` - heteronym label, e.g., word `diffuse` has the following possible labels: `diffuse_vrb` and `diffuse_adj`. See `https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/wordids.tsv <https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/wordids.tsv>`__ for more details.

To convert the WikipediaHomographData to `.json` format suitable for the HeteronymClassificationModel training, run:

.. code-block::

    # WikipediaHomographData could be downloaded from `https://github.com/google-research-datasets/WikipediaHomographData <https://github.com/google-research-datasets/WikipediaHomographData>`__.

    python NeMo/scripts/dataset_processing/g2p/export_wikihomograph_data_to_manifest.py \
            --data_folder=<Path to WikipediaHomographData>/WikipediaHomographData-master/data/eval/
            --output=eval.json
    python NeMo/scripts/dataset_processing/g2p/export_wikihomograph_data_to_manifest.py \
            --data_folder=<Path to WikipediaHomographData>/WikipediaHomographData-master/data/train/
            --output=train.json

To train and evaluate the model, run:

.. code-block::

    python heteronym_classification_train_and_evaluate.py \
        train_manifest=<Path to manifest file>" \
        validation_manifest=<Path to manifest file>" \
        model.encoder.pretrained="<Path to .nemo file or pretrained model name from list_available_models()>" \
        model.wordids=<Path to wordids.tsv file, similar to https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/wordids.tsv> \
        do_training=True \
        do_testing=True


To run inference with a pretrained HeteronymClassificationModel, run:

.. code-block::

    python heteronym_classification_inference.py \
        manifest="<Path to .json manifest>" \
        pretrained_model="<Path to .nemo file or pretrained model name from list_available_models()>" \
        output_file="<Path to .json manifest to save prediction>"

Note, if the input manifest contains target "word_id", evaluation will be also performed. During inference, the model predicts heteronym `word_id` and saves predictions in `"pred_text"` field of the `output_file`:

.. code::

  {"text_graphemes": "Oxygen is less able to diffuse into the blood, leading to hypoxia.", "pred_text": "diffuse_vrb", "start_end": [23, 30], "homograph_span": "diffuse", "word_id": "diffuse_vrb"}


Requirements
------------

G2P requires NeMo NLP and ASR collections installed. See `Installation instructions <https://github.com/NVIDIA/NeMo/tree/stable/docs/source/starthere/intro.rst#installation>`__ for more details.


References
----------

.. bibliography:: ../text_processing_all.bib ../../tts/tts_all.bib ../../nlp/nlp_all.bib
    :style: plain
    :labelprefix: g2p-
    :keyprefix: g2p--
