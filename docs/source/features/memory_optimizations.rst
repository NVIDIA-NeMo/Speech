Memory Optimizations
====================

Parallelism
-----------
Refer to :doc:`Parallelism <./parallelisms>`.

Flash Attention
---------------

Overview
^^^^^^^^

Flash Attention is a method designed to enhance the efficiency of Transformer models, which are widely utilized in applications such as Natural Language Processing (NLP). Traditional Transformers are slow and consume a lot of memory, especially with long sequences, due to the quadratic time and memory complexity of self-attention. Flash Attention is an IO-aware exact attention algorithm that leverages tiling to minimize the number of memory reads/writes between the GPU's high-bandwidth memory (HBM) and on-chip SRAM. This approach is designed to be more efficient in terms of IO complexity compared to standard attention mechanisms.

Turn Flash Attention On and Off
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the NeMo Framework, Flash Attention is supported through the Transformer Engine with the inclusion of Flash Attention 2. By default, Flash Attention is enabled, but the Transformer Engine may switch to a different kernel if the tensor dimensions are not optimal for Flash Attention. Users can completely disable Flash Attention by setting the environment variable ``NVTE_FLASH_ATTN=0``.

For more details on the supported Dot Attention backend, please refer to the Transformer Engine source code available at `Transformer Engine's Attention Mechanism <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py>`_.

Activation Recomputation
------------------------

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

Refer to "Reducing Activation Recomputation in Large Transformer Models" for more details: https://arxiv.org/abs/2205.05198.

Multi-query Attention (MQA) and Grouped-query Attention (GQA)
-------------------------------------------------------------

**Multi-query Attention (MQA)** and **Grouped-query Attention (GQA)** are modifications of the traditional multihead attention mechanism in Transformer models. These methods improve the efficiency and effectiveness of attention mechanisms.

Overview
^^^^^^^^

**Multi-query Attention (MQA)**
    MQA treats all attention heads as a single group, reducing computational complexity and accelerating training times. It is beneficial when model scalability or limited computational resources are concerns.

**Grouped-query Attention (GQA)**
    GQA groups the heads into clusters, each processing a subset of queries independently. This method balances the detailed focus of traditional multihead attention with the broad approach of MQA, enhancing nuanced input data processing.

These attention variants offer:

- **Reduced computational load**: Both methods decrease computation, beneficial for large models.
- **Increased processing speed**: Simplifying attention leads to faster training and inference.
- **Flexibility and adaptability**: Adjustments can be made based on task needs or hardware constraints.

Enable MQA and GQA
^^^^^^^^^^^^^^^^^^

To use MQA or GQA in the NeMo Framework, adjust the ``num_query_groups`` parameter in the model configuration:

1. **For Multi-query Attention (MQA)**:
   - Set ``num_query_groups`` to `1` to treat all attention heads as a single group.

   .. code-block:: yaml

       num_query_groups: 1  # Enables Multi-query Attention

2. **For Grouped-query Attention (GQA)**:
   - Set ``num_query_groups`` to a number that is a divisor of the total number of attention heads (more than one but less than the total heads).

   .. code-block:: yaml

       num_query_groups: <number_of_groups>  # Enables Grouped-query Attention

   - For regular attention, set this parameter to `None` or match it with the number of heads.

   .. code-block:: yaml

       num_query_groups: null  # Default setting for regular multihead attention

Adjust the ``num_query_groups`` to explore different attention mechanisms and optimize your model's performance based on specific needs.

Implement MQA or GQA
^^^^^^^^^^^^^^^^^^^^

NeMo's support for GQA and MQA is enabled through the integration of Megatron Core's Attention mechanism. The underlying implementation details can be explored within the Attention class of Megatron Core, which provides the functional backbone for these advanced attention methods. To understand the specific modifications and implementations of MQA and GQA, refer to the source code in the Attention class:

Check implementation details from Attention Class in Megatron Core Repo: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py#L49
