.. _wfst_customization:

Grammar customization
=====================

.. note::

    TN/ITN is transitioning from `NVIDIA/NeMo <https://github.com/NVIDIA/NeMo>`_ repository to a standalone `NVIDIA/NeMo-text-processing <https://github.com/NVIDIA/NeMo-text-processing>`_ repository. All updates and discussions/issues should go to the new repository.


All grammar development is done with `Pynini library <https://www.opengrm.org/twiki/bin/view/GRM/Pynini>`_.
These grammars can be exported to .far files and used with Riva/Sparrowhawk, see :doc:`Text Processing Deployment <wfst_text_processing_deployment>` for details.

Steps to customize grammars
---------------------------

1. Install `NeMo-TN from source <https://github.com/NVIDIA/NeMo-text-processing#from-source>`_
2. Run `nemo_text_processing/text_normalization/normalize.py <https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/text_normalization/normalize.py>`_ or `nemo_text_processing/inverse_text_normalization/inverse_normalize.py <https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/inverse_text_normalization/inverse_normalize.py>`_ with `--verbose` flag to evaluate current behavior on the target case, see argument details in the scripts and `this tutorial <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_(Inverse)_Normalization.ipynb>`_
3. Modify existing grammars or add new grammars to cover the target case using `Tutorial on how to write new grammars <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/WFST_Tutorial.ipynb>`_
4. Add new test cases `here <https://github.com/NVIDIA/NeMo-text-processing/tree/main/tests/nemo_text_processing>`_:
    - Run python tests:

    .. code-block:: bash

        (optionally build grammars first and save to CACHE_DIR)
        cd tests/nemo_text_processing &&
        cd pytest <LANGUAGE>/test_*.py --cpu --tn_cache_dir=CACHE_DIR_WITH_FAR_FILES (--run_audio_based flag to also run audio-based TN tests, optional)

    - Run Sparrowhawk tests:

    .. code-block:: bash

        cd tools/text_processing_deployment &&
        bash export_grammars.sh --GRAMMARS=<TN/ITN grammars> --LANGUAGE=<LANGUAGE> --MODE=test


WFST TN/ITN resources could be found in :doc:`here <wfst_resources>`.

Riva resources
--------------
    - `Riva Text Normalization customization for TTS <https://riva-builder-01.nvidia.com/main/tts/tts-custom.html#custom-text-normalization>`_.
    - `Riva ASR/Inverse Text Normalization customization <https://riva-builder-01.nvidia.com/main/asr/asr-customizing.html>`_.