NeMo Megatron
=============

Megatron :cite:`nlp-megatron-shoeybi2019megatron` is a large, powerful transformer developed by the Applied Deep Learning Research 
team at NVIDIA. NeMo Megatron supports several types of models:

* GPT-style models (decoder only)
* T5/BART/UL2-style models (encoder-decoder)
* BERT-style models (encoder only)



.. note::
    NeMo Megatron has an Enterprise edition which contains tools for data preprocessing, hyperparameter tuning, container, scripts for various clouds and more. With Enterprise edition you also get deployment tools. Apply for `early access here <https://developer.nvidia.com/nemo-megatron-early-access>`_ .


.. toctree::
   :maxdepth: 1

   mlm_migration   
   gpt/gpt_training
   batching
   parallelisms  
   prompt_learning


References
----------

.. bibliography:: ../nlp_all.bib
    :style: plain
    :labelprefix: nlp-megatron
    :keyprefix: nlp-megatron-