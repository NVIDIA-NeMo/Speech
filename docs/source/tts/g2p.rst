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

Example of the config file: ``NeMo/examples/tts/g2p/conf/g2p_t5.yaml``.


To train G2P-Conformer model and evaluate it after at the end of the training, run:

.. code-block::

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

.. code-block::

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


Requirements
------------

G2P requires the NeMo ASR collection to be installed (``pip install nemo_toolkit[asr]``).


References
----------

.. bibliography:: tts_all.bib
    :style: plain
    :labelprefix: g2p-
    :keyprefix: g2p--
