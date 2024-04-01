Large Language Models
=====================

To learn more about using NeMo to train Large Language Models at scale, please refer to the `NeMo Framework User Guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html>`_.

* GPT-style models (decoder only)
* T5/BART/UL2-style models (encoder-decoder)
* BERT-style models (encoder only)
* RETRO model (decoder only)


.. toctree::
   :maxdepth: 1

   mlm_migration   
   gpt/gpt_training
   batching
   parallelisms  
   prompt_learning
   retro/retro_model
   hiddens/hiddens_module
   peft/landing_page
   flash_attention
   positional_embeddings
   mcore_customization
   packed_sequence


References
----------

.. bibliography:: ../nlp_all.bib
    :style: plain
    :labelprefix: nlp-megatron
    :keyprefix: nlp-megatron-