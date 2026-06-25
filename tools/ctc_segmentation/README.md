Dataset creation tool based on CTC-Segmentation
-----------------------------------------------

This tool provides functionality to align long audio files and the corresponding transcripts into shorter fragments
that are suitable for an Automatic Speech Recognition (ASR) model training.

More details could be found in [this tutorial](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/tools/CTC_Segmentation_Tutorial.ipynb).

The tool is based on the [CTC Segmentation](https://github.com/lumaku/ctc-segmentation):
**CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition**
https://doi.org/10.1007/978-3-030-60276-5_27 or pre-print https://arxiv.org/abs/2007.09127

```
@InProceedings{ctcsegmentation,
author="K{\"u}rzinger, Ludwig
and Winkelbauer, Dominik
and Li, Lujun
and Watzel, Tobias
and Rigoll, Gerhard",
editor="Karpov, Alexey
and Potapova, Rodmonga",
title="CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition",
booktitle="Speech and Computer",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="267--278",
abstract="Recent end-to-end Automatic Speech Recognition (ASR) systems demonstrated the ability to outperform conventional hybrid DNN/HMM ASR. Aside from architectural improvements in those systems, those models grew in terms of depth, parameters and model capacity. However, these models also require more training data to achieve comparable performance.",
isbn="978-3-030-60276-5"
}
```
Requirements
~~~~~~~~~~~~
The tool requires:
- packages listed in requirements.txt
- NeMo ASR
- see pysox’s documentation (https://pysox.readthedocs.io/en/latest/) if you want support for mp3, flac and ogg files
