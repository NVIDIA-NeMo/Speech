.. _best-practices:

Best Practices
==============

The NVIDIA NeMo Toolkit is available on GitHub as `open source <https://github.com/NVIDIA/NeMo>`_ as well as
a `Docker container on NGC <https://ngc.nvidia.com/catalog/containers/nvidia:nemo>`_. It's assumed the user has
already installed NeMo by following the :ref:`quick_start_guide` instructions.

The conversational AI pipeline consists of three major stages:

- Automatic Speech Recognition (ASR)
- Natural Language Processing (NLP) or Natural Language Understanding (NLU)
- Text-to-Speech (TTS) Synthesis

As you talk to a computer, the ASR phase converts the audio signal into text, the NLP stage interprets the question
and generates a smart response, and finally the TTS phase converts the text into speech signals to generate audio for
the user. The toolkit enables development and training of deep learning models involved in conversational AI and easily
chain them together.

Why NeMo?
---------

Deep learning model development for conversational AI is complex. It involves defining, building, and training several
models in specific domains; experimenting several times to get high accuracy, fine tuning on multiple tasks and domain
specific data, ensuring training performance and making sure the models are ready for deployment to inference applications.
Neural modules are logical blocks of AI applications which take some typed inputs and produce certain typed outputs. By
separating a model into its essential components in a building block manner, NeMo helps researchers develop state-of-the-art
accuracy models for domain specific data faster and easier.

Collections of modules for core tasks as well as specific to speech recognition, natural language, speech synthesis help
develop modular, flexible, and reusable pipelines.

A neural module’s inputs/outputs have a neural type, that describes the semantics, the axis order and meaning, and the dimensions
of the input/output tensors. This typing allows neural modules to be safely chained together to build models for applications.

NeMo can be used to train new models or perform transfer learning on existing pre-trained models. Pre-trained weights per module
(such as encoder, decoder) help accelerate model training for domain specific data.

ASR, NLP and TTS pre-trained models are trained on multiple datasets (including some languages such as Mandarin) and optimized
for high accuracy. They can be used for transfer learning as well.

NeMo supports developing models that work with Mandarin Chinese data. Tutorials help users train or fine tune models for
conversational AI with the Mandarin Chinese language. The export method provided in NeMo makes it easy to transform a trained
model into inference ready format for deployment.

A key area of development in the toolkit is interoperability with other tools used by speech researchers. Data layer for Kaldi
compatibility is one such example.

NeMo, PyTorch Lightning, And Hydra
----------------------------------

Conversational AI architectures are typically very large and require a lot of data and compute for training. NeMo uses
`Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ for easy and performant multi-GPU/multi-node
mixed precision training.

Pytorch Lightning is a high-performance PyTorch wrapper that organizes PyTorch code, scales model training, and reduces
boilerplate. PyTorch Lightning has two main components, the ``LightningModule`` and the Trainer. The ``LightningModule`` is
used to organize PyTorch code so that deep learning experiments can be easily understood and reproduced. The Pytorch Lightning
Trainer is then able to take the ``LightningModule`` and automate everything needed for deep learning training.

NeMo models are LightningModules that come equipped with all supporting infrastructure for training and reproducibility. This
includes the deep learning model architecture, data preprocessing, optimizer, check-pointing and experiment logging. NeMo
models, like LightningModules, are also PyTorch modules and are fully compatible with the broader PyTorch ecosystem. Any NeMo
model can be taken and plugged into any PyTorch workflow.

Configuring conversational AI applications is difficult due to the need to bring together many different Python libraries into
one end-to-end system. NeMo uses Hydra for configuring both NeMo models and the PyTorch Lightning Trainer. `Hydra <https://github.com/facebookresearch/hydra>`_
is a flexible solution that makes it easy to configure all of these libraries from a configuration file or from the command-line.

Every NeMo model has an example configuration file and a corresponding script that contains all configurations needed for training
to state-of-the-art accuracy. NeMo models have the same look and feel so that it is easy to do conversational AI research across
multiple domains.

Using Optimized Pretrained Models With NeMo
-------------------------------------------

`NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com/catalog>`_ is a software repository that has containers and models optimized
for deep learning. NGC hosts many conversational AI models developed with NeMo that have been trained to state-of-the-art accuracy
on large datasets. NeMo models on NGC can be automatically downloaded and used for transfer learning tasks. Pretrained models
are the quickest way to get started with conversational AI on your own data. NeMo has many `example scripts <https://github.com/NVIDIA/NeMo/tree/stable/examples>`_
and `Jupyter Notebook tutorials <https://github.com/NVIDIA/NeMo#tutorials>`_ showing step-by-step how to fine-tune pretrained NeMo
models on your own domain-specific datasets.

For BERT based models, the model weights provided are ready for
downstream NLU tasks. For speech models, it can be helpful to start with a pretrained model and then continue pretraining on your
own domain-specific data. Jasper and QuartzNet base model pretrained weights have been known to be very efficient when used as
base models. For an easy to follow guide on transfer learning and building domain specific ASR models, you can follow this `blog <https://developer.nvidia.com/blog/how-to-build-domain-specific-automatic-speech-recognition-models-on-gpus/>`_.
All pre-trained NeMo models can be found on the `NGC NeMo Collection <https://ngc.nvidia.com/catalog/collections?orderBy=scoreDESC&pageNumber=0&query=NeMo&quickFilter=&filters=>`_. Everything needed to quickly get started
with NeMo ASR, NLP, and TTS models is there.

Pre-trained models are packaged as a ``.nemo`` file and contain the PyTorch checkpoint along with everything needed to use the model.
NeMo models are trained to state-of-the-art accuracy and trained on multiple datasets so that they are robust to small differences
in data. NeMo contains a large variety of models such as speaker identification and Megatron BERT and the best models in speech and
language are constantly being added as they become available. NeMo is the premier toolkit for conversational AI model building and
training.

For a list of supported models, refer to the :ref:`tutorials` section.

ASR Guidance
------------

This section is to help guide your decision making by answering our most asked ASR questions.

**Q: Is there a way to add domain specific vocabulary in NeMo? If so, how do I do that?**
A: QuartzNet and Jasper models are character-based. So pretrained models we provide for these two output lowercase English
letters and ‘. Users can re-retrain them on vocabulary with upper case letters and punctuation symbols.

**Q: When training, there are “Reference” lines and “Decoded” lines that are printed out. It seems like the reference line should
be the “truth” line and the decoded line should be what the ASR is transcribing. Why do I see that even the reference lines do not
appear to be correct?**
A: Because our pre-trained models can only output lowercase letters and apostrophe, everything else is dropped. So the model will
transcribe 10 as ten. The best way forward is to prepare the training data first by transforming everything to lowercase and convert
the numbers from digit representation to word representation using a simple library such as `inflect <https://pypi.org/project/inflect/>`_. Then, add the uppercase letters
and punctuation back using the NLP punctuation model. Here is an example of how this is incorporated: `NeMo voice swap demo <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/VoiceSwapSample.ipynb>`_.

**Q: What languages are supported in NeMo currently?**
A: Along with English, we provide pre-trained models for Zh, Es, Fr, De, Ru, It, Ca and Pl languages.
For more information, see `NeMo Speech Models <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_.

Data Augmentation
-----------------

Data augmentation in ASR is invaluable. It comes at the cost of increased training time if samples are augmented during training
time. To save training time, it is recommended to pre-process the dataset offline for a one time preprocessing cost and then train
the dataset on this augmented training set.

For example, processing a single sample involves:

- Speed perturbation
- Time stretch perturbation (sample level)
- Noise perturbation
- Impulse perturbation
- Time stretch augmentation (batch level, neural module)

A simple tutorial guides users on how to use these utilities provided in `GitHub: NeMo <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Online_Noise_Augmentation.ipynb>`_.

Speech Data Explorer
--------------------

Speech data explorer is a `Dash-based tool <https://plotly.com/dash/>`_ for interactive exploration of ASR/TTS datasets.

Speech data explorer collects:

- dataset statistics (alphabet, vocabulary, and duration-based histograms)
- navigation across datasets (sorting and filtering)
- inspections of individual utterances (waveform, spectrogram, and audio player)
- errors analysis (word error rate, character error rate, word match rate, mean word accuracy, and diff)

In order to use the tool, it needs to be installed separately. Perform the steps `here <https://github.com/NVIDIA/NeMo/tree/stable/tools/speech_data_explorer>`_ to install speech data explorer.

Using Kaldi Formatted Data
--------------------------

The `Kaldi Speech Recognition Toolkit <https://kaldi-asr.org/>`_ project began in 2009 at `Johns Hopkins University <https://www.jhu.edu/>`. It is a toolkit written in C++. If
researchers have used Kaldi and have datasets that are formatted to be used with the toolkit; they can use NeMo to develop models
based on that data.

To load Kaldi-formatted data, you can simply use ``KaldiFeatureDataLayer`` instead of ``AudioToTextDataLayer``. The ``KaldiFeatureDataLayer``
takes in the argument ``kaldi_dir`` instead of a ``manifest_filepath``. The ``manifest_filepath`` argument should be set to the directory
that contains the files ``feats.scp`` and ``text``.

Using Speech Command Recognition Task For ASR Models
----------------------------------------------------

Speech Command Recognition is the task of classifying an input audio pattern into a set of discrete classes. It is a subset of ASR,
sometimes referred to as Key Word Spotting, in which a model is constantly analyzing speech patterns to detect certain ``action`` classes.

Upon detection of these commands, a specific action can be taken. An example Jupyter notebook provided in NeMo shows how to train a
QuartzNet model with a modified decoder head trained on a speech commands dataset.

.. note:: It is preferred that you use absolute paths to ``data_dir`` when preprocessing the dataset.

NLP Fine-Tuning BERT
--------------------

BERT, or Bidirectional Encoder Representations from Transformers, is a neural approach to pre-train language representations which
obtains near state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks, including the GLUE benchmark and
SQuAD Question & Answering dataset.

BERT model checkpoints (`BERT-large-uncased <https://ngc.nvidia.com/catalog/models/nvidia:bertlargeuncasedfornemo>`_ and `BERT-base-uncased <https://ngc.nvidia.com/catalog/models/nvidia:bertbaseuncasedfornemo>`_) are provided can be used for either fine tuning BERT on your custom
dataset, or fine tuning downstream tasks, including GLUE benchmark tasks, Question & Answering tasks, Joint Intent & Slot detection,
Punctuation and Capitalization, Named Entity Recognition, and Speech Recognition post processing model to correct mistakes.

.. note:: Almost all NLP examples also support RoBERTa and ALBERT models for downstream fine-tuning tasks (see the list of all supported models by calling ``nemo.collections.nlp.modules.common.lm_utils.get_pretrained_lm_models_list()``). The user needs to specify the name of the model desired while running the example scripts.

BioMegatron Medical BERT
------------------------

BioMegatron is a large language model (Megatron-LM) trained on larger domain text corpus (PubMed abstract + full-text-commercial).
It achieves state-of-the-art results for certain tasks such as Relationship Extraction, Named Entity Recognition and Question &
Answering. Follow these tutorials to learn how to train and fine tune BioMegatron; pretrained models are provided on NGC:

- `Relation Extraction BioMegatron <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb>`_
- `Token Classification BioMegatron <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Token_Classification-BioMegatron.ipynb>`_

Efficient Training With NeMo
----------------------------

Using Mixed Precision
^^^^^^^^^^^^^^^^^^^^^

Mixed precision accelerates training speed while protecting against noticeable loss. Tensor Cores is a specific hardware unit that
comes starting with the Volta and Turing architectures to accelerate large matrix to matrix multiply-add operations by operating them
on half precision inputs and returning the result in full precision.

Neural networks which usually use massive matrix multiplications can be significantly sped up with mixed precision and Tensor Cores.
However, some neural network layers are numerically more sensitive than others. Apex AMP is an NVIDIA library that maximizes the
benefit of mixed precision and Tensor Cores usage for a given network.

Multi-GPU Training
^^^^^^^^^^^^^^^^^^

This section is to help guide your decision making by answering our most asked multi-GPU training questions.

**Q: Why is multi-GPU training preferred over other types of training?**
A: Multi-GPU training can reduce the total training time by distributing the workload onto multiple compute instances. This is
particularly important for large neural networks which would otherwise take weeks to train until convergence. Since NeMo supports
multi-GPU training, no code change is needed to move from single to multi-GPU training, only a slight change in your launch command
is required.

**Q: What are the advantages of mixed precision training?**
A: Mixed precision accelerates training speed while protecting against noticeable loss in precision. Tensor Cores is a specific
hardware unit that comes starting with the Volta and Turing architectures to accelerate large matrix multiply-add operations by
operating on half precision inputs and returning the result in full precision in order to prevent loss in precision. Neural
networks which usually use massive matrix multiplications can be significantly sped up with mixed precision and Tensor Cores.
However, some neural network layers are numerically more sensitive than others. Apex AMP is a NVIDIA library that maximizes the
benefit of mixed precision and Tensor Core usage for a given network.

**Q: What is the difference between multi-GPU and multi-node training?**
A: Multi-node is an abstraction of multi-GPU training, which requires a distributed compute cluster, where each node can have multiple
GPUs. Multi-node training is needed to scale training beyond a single node to large amounts of GPUs.

From the framework perspective, nothing changes from moving to multi-node training. However, a master address and port needs to be set
up for inter-node communication. Multi-GPU training will then be launched on each node with passed information. You might also consider
the underlying inter-node network topology and type to achieve full performance, such as HPC-style hardware such as NVLink, InfiniBand
networking, or Ethernet.


Recommendations For Optimization And FAQs
-----------------------------------------

This section is to help guide your decision making by answering our most asked NeMo questions.

**Q: Are there areas where performance can be increased?**
A: You should try using mixed precision for improved performance. Note that typically when using mixed precision, memory consumption
is decreased and larger batch sizes could be used to further improve the performance.

When fine-tuning ASR models on your data, it is almost always possible to take advantage of NeMo's pre-trained modules. Even if you
have a different target vocabulary, or even a different language; you can still try starting with pre-trained weights from Jasper or
QuartzNet ``encoder`` and only adjust the ``decoder`` for your needs.

**Q: What is the recommended sampling rate for ASR?**
A: The released models are based on 16 KHz audio, therefore, ensure you use models with 16 KHz audio. Reduced performance should be
expected for any audio that is up-sampled from a sampling frequency less than 16 KHz data.

**Q: How do we use this toolkit for audio with different types of compression and frequency than the training domain for ASR?**
A: You have to match the compression and frequency.

**Q: How do you replace the 6-gram out of the ASR model with a custom language model? What is the language format supported in NeMo?**
A: NeMo’s Beam Search decoder with Levenberg-Marquardt (LM) neural module supports the KenLM language model.

- You should retrain the KenLM language model on your own dataset. Refer to `KenLM’s documentation <https://github.com/kpu/kenlm#kenlm>`_.
- If you want to use a different language model, other than KenLM, you will need to implement a corresponding decoder module.
- Transformer-XL example is present in OS2S. It would need to be updated to work with NeMo. `Here is the code <https://github.com/NVIDIA/OpenSeq2Seq/tree/master/external_lm_rescore>`_.

**Q: How do I use text-to-speech (TTS) synthesis?**
A:

- Obtain speech data ideally at 22050 Hz or alternatively at a higher sample rate and then down sample to 22050 Hz.
    - If less than 22050 Hz and at least 16000 Hz:
        - Retrain WaveGlow on your own dataset.
        - Tweak the spectrogram generation parameters, namely the ``window_size`` and the ``window_stride`` for their fourier transforms.
    - For below 16000 Hz, look into obtaining new data.
- In terms of bitrate/quantization, the general advice is the higher the better. We have not experimented enough to state how much
  this impacts quality.
- For the amount of data, again the more the better, and the more diverse in terms of phonemes the better. Aim for around 20 hours
  of speech after filtering for silences and non-speech audio.
- Most open speech datasets are in ~10 second format so training spectrogram generators on audio on the order of 10s - 20s per sample is known
  to work. Additionally, the longer the speech samples, the more difficult it will be to train them.
- Audio files should be clean. There should be little background noise or music. Data recorded from a studio mic is likely to be easier
  to train compared to data captured using a phone.
- To ensure pronunciation of words are accurate; the technical challenge is related to the dataset, text to phonetic spelling is
  required, use phonetic alphabet (notation) that has the name correctly pronounced.
- Here are some example parameters you can use to train spectrogram generators:
    - use single speaker dataset
    - Use AMP level O0
    - Trim long silences in the beginning and end
    - ``optimizer="adam"``
    - ``beta1 = 0.9``
    - ``beta2 = 0.999``
    - ``lr=0.001 (constant)``
    - ``amp_opt_level="O0"``
    - ``weight_decay=1e-6``
    - ``batch_size=48 (per GPU)``
    - ``trim_silence=True``

Resources
---------

Ensure you are familiar with the following resources for NeMo.

- Developer blogs
    - `How to Build Domain Specific Automatic Speech Recognition Models on GPUs <https://developer.nvidia.com/blog/how-to-build-domain-specific-automatic-speech-recognition-models-on-gpus/>`_
    - `Develop Smaller Speech Recognition Models with NVIDIA’s NeMo Framework <https://developer.nvidia.com/blog/develop-smaller-speech-recognition-models-with-nvidias-nemo-framework/>`_
    - `Neural Modules for Fast Development of Speech and Language Models <https://developer.nvidia.com/blog/neural-modules-for-speech-language-models/>`_

- Domain specific, transfer learning, Docker container with Jupyter Notebooks
    - `Domain Specific NeMo ASR Application <https://ngc.nvidia.com/catalog/containers/nvidia:nemo_asr_app_img>`_

