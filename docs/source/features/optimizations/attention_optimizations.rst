Attention Optimizations
=======================

Flash Attention
---------------

Overview
^^^^^^^^

Flash attention is an algorithm designed to improve the efficiency of the attention mechanism in transformer models such as GPT and BERT. The attention mechanism has quadratic time and memory complexity in sequence length and can present significant runtime and memory challenges for longer sequences.

Compared to the standard, non-flash algorithm, flash attention applies two techniques to lower the memory requirement and improve compute efficiency.

The tiling technique decomposes the inputs based on the shared memory size and calculates the softmax one tile at a time. Instead of working on the entire query, key, value tensors at once, it makes several passes at these tensors and then combines the results in a subsequent step.

The recomputation technique stores the softmax normalization factors (linear to sequence length), instead of the softmax results (qudratic to sequence length), and uses these normalization factors to recompute the attention scores. This saves the amount of data to write to global memory and reduces both the memory requirement and I/O traffic between global memory and shared memory.

Flash attention lowers the memory footprint and computational complexity from quadratic to linear, and greatly extending the range of sequence length allowed in large language models.

The flash attention algorithm was first propsed `here <https://arxiv.org/pdf/2205.14135>`_. Two of its implementations are `flash-attention <https://github.com/Dao-AILab/flash-attention>`_ by Tri Dao *et al*, and `fused flash attention <https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/developer-guide/index.html#flash-fused-multi-head-att-fprop>`_ by NVIDIA cuDNN.

Turn Flash Attention On and Off
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the NeMo framework, flash attention is supported through `Transformer Engine <https://github.com/NVIDIA/TransformerEngine/tree/main>`_, including both of the implementations mentioned above. Transformer Engine selects the appropriate implementation based on input information such as sequence length, number of heads and head dimension. When both implementations are applicable, Transformer Engine prefers cuDNN flash attention on Hopper+ architectures and Tri Dao flash attention on Ampere architectures.

To disable Tri Dao flash attention, set the environment variable ``NVTE_FLASH_ATTN=0``. To disable cuDNN flash attention, set ``NVTE_FUSED_ATTN=0``.

For more details on the Dot Product Attention backends supported in Transformer Engine, please refer to the source code at `Transformer Engine's Attention Mechanism <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py>`_.

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

Check implementation details from Attention Class in Megatron Core Repo: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py#L49.
