Dataset Creation Tool Based on CTC-Segmentation
===============================================

This tool provides functionality to align long audio files with the corresponding transcripts and split them into shorter fragments
that are suitable for an Automatic Speech Recognition (ASR) model training.

More details could be found in `NeMo/tutorials/tools/CTC_Segmentation_Tutorial.ipynb <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/CTC_Segmentation_Tutorial.ipynb>`__ (can be executed with `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_).

The tool is based on the `CTC-Segmentation <https://github.com/lumaku/ctc-segmentation>`__ package and
`CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition
<https://arxiv.org/abs/2007.09127>`__ :cite:`tools-kurzinger2020ctc`


References
----------

.. bibliography:: tools_all.bib
    :style: plain
    :labelprefix: TOOLS
    :keyprefix: tools-
