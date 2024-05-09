.. _wfst_tn:

Text (Inverse) Normalization
============================

.. warning::

    TN/ITN transitioned from `NVIDIA/NeMo <https://github.com/NVIDIA/NeMo>`_ repository to a standalone `NVIDIA/NeMo-text-processing <https://github.com/NVIDIA/NeMo-text-processing>`_ repository. All updates and discussions/issues should go to the new repository.

The `nemo_text_processing` Python package is based on WFST grammars :cite:`textprocessing-norm-mohri2005weighted` and supports:

1. Text Normalization (TN) converts text from written form into its verbalized form. It is used as a preprocessing step before Text to Speech (TTS). For example,

.. code-block:: bash

     "123" -> "one hundred twenty three"

`nemo_text_processing` has both a fast version which is deterministic :cite:`textprocessing-norm-zhang2021nemo` which has more language support and a context-aware version :cite:`textprocessing-norm-bakhturina2022shallow`.
In case of ambiguous input, e.g. 

.. code-block:: bash

     "St. Patrick's Day" -> "Saint Patrick's Day"
     "St. Patrick's Day" -> "Street Patrick's Day"


the context-aware TN will convert "St. Patrick's Day" to  "Saint Patrick's Day".


2. Inverse text normalization (ITN) is a part of the Automatic Speech Recognition (ASR) post-processing pipeline and can be used to convert normalized ASR model outputs into written form to improve text readability. For example,
   
.. code-block:: bash
    
     "one hundred twenty three" -> "123"

3. Audio-based provides multiple normalization options. For example,

.. code-block:: bash
    
     "123" -> "one hundred twenty three", "one hundred and twenty three", "one two three", "one twenty three" ...  

The normalization which best reflects what is actually said in an audio is then picked. 
Audio-based TN can be used to normalize ASR training data.

    .. image:: images/task_overview.png
        :align: center
        :alt: Text TN and ITN
        :scale: 50%


Installation
------------

If you have already installed `nemo_text_processing <https://github.com/NVIDIA/NeMo-text-processing>`_, it should have `pynini` python library. Otherwise install explicitly:

.. code-block:: shell-session

    pip install pynini==2.1.5

or if this fails on missing OpenFst headers:

.. code-block:: shell-session

    conda install -c conda-forge pynini=2.1.5


Quick Start Guide
-----------------

Text Normalization
^^^^^^^^^^^^^^^^^^

The standard text normalization based on WFST  :cite:`textprocessing-norm-zhang2021nemo` is not context-aware. It is fast and can be run like this:

.. code-block:: bash

    cd NeMo-text-processing/nemo_text_processing/text_normalization/
    python normalize.py --text="123" --language=en

if you want to normalize a string. To normalize a text file split into sentences, run the following:

.. code-block:: bash

    cd NeMo-text-processing/nemo_text_processing/text_normalization/
    python normalize.py --input_file=INPUT_FILE_PATH --output_file=OUTPUT_FILE_PATH --language=en

The context-aware version :cite:`textprocessing-norm-bakhturina2022shallow` is a shallow fusion of non-deterministic WFST and pretrained masked language model.

    .. image:: images/shallow_fusion.png
        :align: center
        :alt: Text Shallow Fusion of WFST and LM
        :scale: 80%


.. code-block:: bash

    cd NeMo-text-processing/nemo_text_processing/
    python wfst_lm_rescoring.py




Inverse Text Normalization 
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd NeMo-text-processing/nemo_text_processing/inverse_text_normalization/
    python inverse_normalize.py --text="one hundred twenty three" --language=en


Arguments:

* ``text`` - Input text. Should not exceed 500 words.
* ``input_file`` - Input file with lines of input text. Only one of ``text`` or ``input_file`` is accepted.
* ``output_file`` - Output file to save normalizations. Needed if ``input_file`` is specified.
* ``language`` - language id.
* ``input_case`` - Only for text normalization. ``lower_cased`` or ``cased``.
* ``verbose`` - Outputs intermediate information.
* ``cache_dir`` - Specifies a cache directory for compiled grammars. If grammars exist, this significantly improves speed. 
* ``overwrite_cache`` - Updates grammars in cache.
* ``whitelist`` - TSV file with custom mappings of written text to spoken form.



.. warning::
   The maximum length of a single string to be (de-)normalized should not exceed 500 words. To avoid this, please split your string into sentences shorter than this limit and pass it as ``--input_file`` instead.


Audio-based TN 
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd NeMo-text-processing/nemo_text_processing/text_normalization/
    python normalize_with_audio.py --text="123" --language="en" --n_tagged=10 --cache_dir="cache_dir" --audio_data="example.wav" --model="stt_en_conformer_ctc_large" 

Additional Arguments:

* ``text`` - Input text or :ref:`JSON manifest file<section-with-manifest-format-explanation>` with multiple audio paths.
* ``audio_data`` - (Optional) Input audio.
* ``model`` - :ref:`Off-shelf NeMo CTC ASR model name <asr-checkpoint-list-by-language>` or path to local NeMo model checkpoint ending on .nemo
* ``n_tagged`` - number of normalization options to output.


.. note::

    More details can be found in `NeMo-text-processing/tutorials/text_processing/Text_(Inverse)_Normalization.ipynb <https://github.com/NVIDIA/NeMo-text-processing/blob/main/tutorials/Text_(Inverse)_Normalization.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo-text-processing/blob/main/tutorials/Text_(Inverse)_Normalization.ipynb>`_.

Language Support Matrix
------------------------

+------------------+----------+----------+----------+--------------------+----------------------+
| **Language**     | **ID**   | **TN**   | **ITN**  | **Audio-based TN** | **context-aware TN** |
+------------------+----------+----------+----------+--------------------+----------------------+
| English          | en       | x        | x        | x                  | x                    |
+------------------+----------+----------+----------+--------------------+----------------------+
| Spanish          | es       | x        | x        | x                  |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Spanish-English  | es_en    |          | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| French           | fr       | x        | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| German           | de       | x        | x        | x                  |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Arabic           | ar       | x        | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Russian          | ru       |          | x        | x                  |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Swedish          | sv       | x        | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Vietnamese       | vi       |          | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Portuguese       | pt       |          | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Chinese          | zh       | x        | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Hungarian        | hu       | x        |          |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Italian          | it       | x        |          |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Armenian         | hy       | x        | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+
| Marathi          | mr       |          | x        |                    |                      |
+------------------+----------+----------+----------+--------------------+----------------------+


See :ref:`Grammar customization <wfst_customization>` for grammar customization details.

See :doc:`Text Processing Deployment <./wfst_text_processing_deployment>` for deployment in C++ details.

WFST TN/ITN resources could be found in :ref:`here <wfst_resources>`.

References
----------

.. bibliography:: ../tn_itn_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-NORM
    :keyprefix: textprocessing-norm-
