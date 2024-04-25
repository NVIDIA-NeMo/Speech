Memory Optimizations
====================

Parallelism
-----------
Refer to :doc:`Parallelism <./parallelism>`.

Flash Attention
---------------

Overview
^^^^^^^^

Flash Attention is a method designed to enhance the efficiency of Transformer models, which are widely utilized in applications such as Natural Language Processing (NLP). Traditional Transformers are slow and consume a lot of memory, especially with long sequences, due to the quadratic time and memory complexity of self-attention. FlashAttention, an IO-aware exact attention algorithm that leverages tiling to minimize the number of memory reads/writes between the GPU's high bandwidth memory (HBM) and on-chip SRAM. This approach is designed to be more efficient in terms of IO complexity compared to standard attention mechanisms.

Turn Flash Attention On and Off
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the NeMo Framework, Flash Attention is supported through the Transformer Engine with the inclusion of Flash Attention 2. By default, Flash Attention is enabled, but the Transformer Engine may switch to a different kernel if the tensor dimensions are not optimal for Flash Attention. Users can completely disable Flash Attention by setting the environment variable ``NVTE_FLASH_ATTN=0``.

For more details on the supported Dot Attention backend, please refer to the Transformer Engine source code available at `Transformer Engine's Attention Mechanism <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py>`_.

.. bibliography:: ./nlp_all.bib
    :style: plain
    :labelprefix: nlp-megatron
    :keyprefix: nlp-megatron-

Overview
^^^^^^^^

Full Activation Recomputation
"""""""""""""""""""""""""""""
This method recalculates all the intermediate activations during the backward pass of a model's training, instead of storing them during the forward pass. This technique maximizes memory efficiency at the cost of computational overhead, as each activation is recomputed when needed.

Partial Activation Recomputation
""""""""""""""""""""""""""""""""
This method recomputes only a subset of layers during the backward phase. It is a trade-off between the full recomputation and no recomputation, balancing memory savings with computational efficiency.

Selective Activation Recomputation
""""""""""""""""""""""""""""""""""
This method reduces memory footprint of activations significantly via smart activation checkpointing. This approach involves selectively storing only crucial activations and recomputing the others as needed. It is particularly useful in large models to minimize memory usage while controlling the computational cost.

Refer to "Reducing Activation Recomputation in Large Transformer Models" for more details: https://arxiv.org/abs/2205.05198

.. bibliography:: ./nlp_all.bib
    :style: plain
    :labelprefix: nlp-megatron
    :keyprefix: nlp-megatron-