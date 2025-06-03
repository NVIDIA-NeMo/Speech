NeMo AutoModel
==============


NeMo AutoModel enables the training and fine-tuning of models accessible through the Hugging Face Transformer AutoModel classes.
Specifically, it supports models such as:

- AutoModelForCausalLM
- AutoModelForImageTextToText
- AutoModelForSpeechSeq2Seq

It covers Large Language Models (LLM), Vision Language Models (VLM), and Automatic Speech Recognition (ASR).


For distributed processing, the NeMo AutoModel provides integration with Distributed Data Parallel (DDP)
and Fully Sharded Data Parallel (FSDP2), ensuring efficient and scalable training across multiple GPUs and nodes.



For more information, browse the developer documentation for your area of interest in the contents section below or on the left sidebar.

.. toctree::
   :maxdepth: 1
   :caption: AutoModel Code Documentation
   :name: AutoModel Code Documentation

   codedocs/hf_automodel_for_causal_lm
   codedocs/hf_automodel_for_image_text_to_text
   codedocs/hf_automodel_for_speech_seq_to_seq

.. toctree::
   :maxdepth: 1
   :caption: AutoModel Data Documentation
   :name: AutoModel Data Documentation

   codedocs/hf_dataset_data_module

.. toctree::
   :maxdepth: 1
   :caption: AutoModel Callbacks Documentation
   :name: AutoModel Callbacks Documentation

   codedocs/jit_callback
