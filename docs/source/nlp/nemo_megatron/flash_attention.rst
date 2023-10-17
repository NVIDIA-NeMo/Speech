Flash attention
---------------
Flash Attention :cite:`nlp-megatron-dao2022flashattention` is a method designed to enhance the efficiency of Transformer models, which are widely utilized in applications such as natural language processing. Traditional Transformers are slow and consume a lot of memory, especially with long sequences, due to the quadratic time and memory complexity of self-attention. FlashAttention, an IO-aware exact attention algorithm that leverages tiling to minimize the number of memory reads/writes between the GPU's high bandwidth memory (HBM) and on-chip SRAM. This approach is designed to be more efficient in terms of IO complexity compared to standard attention mechanisms.

GPT
^^^
To enable Flash Attention while Megatron GPT model training or fine-tuning, modify the following configuration: 

.. code::

   model.use_flash_attention=True

T5
^^
To enable Flash Attention while Megatron T5 model training, modify the following configuration: 

.. code::

   model.encoder.use_flash_attention=True
   model.decoder.use_flash_attention=True

References
----------

.. bibliography:: ../nlp_all.bib
    :style: plain
    :labelprefix: nlp-megatron
    :keyprefix: nlp-megatron-
