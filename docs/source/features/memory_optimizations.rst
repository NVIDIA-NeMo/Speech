Memory Optimizations
====================

Parallelism
-----------
Refer to :doc:`Parallelism <./parallelisms>`.


Mixture of Experts
------------------

Overview
^^^^^^^^

NeMo supports Mixture of Experts (MoE) in the transformer layer for NLP models.

MoE is a machine learning technique where multiple specialized models (experts,
usually multi-layer perceptrons) are combined to solve a complex task. Each expert
focuses on a specific subtask or domain, while a gating network dynamically activates
the most appropriate expert based on the current input.


To use MoE  in the NeMo Framework, adjust the ``num_moe_experts`` parameter in the model configuration:

1. Set ``num_moe_experts`` to `8` to leverage 8 experts in the MoE module.

   .. code-block:: yaml

       num_moe_experts: 8  # Set MoE to use 8 experts

2. Set ``moe_router_topk`` to the number of experts you want activated. For example, if you want to process each input with two experts:

   .. code-block:: yaml

       moe_router_topk: 2  # Processes each token using 2 experts.

In addition, NeMo provides options to configure MoE-specific loss function.
To balance token distribution across experts:

1. Set ``moe_router_load_balancing_type`` to specify the load balancing method:

   .. code-block:: yaml

      moe_router_load_balancing_type: aux_loss  # to use the auxilary loss, other options include "sinkhorn".

2. Set ``moe_aux_loss_coeff`` to specify the weight of the auxilary loss. Values in the 1e-2 range are a good start, as follows:

   .. code-block:: yaml

      moe_aux_loss_coeff: 1e-2  # set the aux-loss weight to 1e-2

3. Set ``moe_z_loss_coeff`` to specify the weight of the z-loss. A starting value of 1e-3 is recommended, as follows:

   .. code-block:: yaml

      moe_z_loss_coeff: 1e-3

Other options include:

1. ``moe_input_jitter_eps`` adds noise to the input tensor by applying jitter with a specified epsilon value.

2. ``moe_token_dropping`` enables selectively dropping and padding tokens for each expert to achieve
   a specified capacity.

3. ``moe_token_dropping`` specifies the token dispatcher type, options include 'allgather' and 'alltoall'.

4. ``moe_per_layer_logging`` enables per-layer logging for MoE, currently support aux-loss and z-loss.

5. ``moe_expert_capacity_factor`` the capacity factor for each expert, None means no token will be dropped. The default is None.

6. ``moe_pad_expert_input_to_capacity`` if True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False.

7. ``moe_token_drop_policy`` the policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped. Default value is "probs".

8. ``moe_layer_recompute`` if True, checkpointing moe_layer to save activation memory, default is False.






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

Activation Recomputation
------------------------

Overview
^^^^^^^^

Full Activation Recomputation
"""""""""""""""""""""""""""""
The full activation recomputation method recalculates all the intermediate activations during the backward pass of a model's training, instead of storing them during the forward pass. This technique maximizes memory efficiency at the cost of computational overhead, as each activation is recomputed when needed.

Partial Activation Recomputation
""""""""""""""""""""""""""""""""
The partial activation recomputation method recomputes only a subset of layers during the backward phase. It is a trade-off between the full recomputation and no recomputation, balancing memory savings with computational efficiency.

Selective Activation Recomputation
""""""""""""""""""""""""""""""""""
The selective activation recomputation method reduces memory footprint of activations significantly via smart activation checkpointing. This approach involves selectively storing only crucial activations and recomputing the others as needed. It is particularly useful in large models to minimize memory usage while controlling the computational cost.

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

Check implementation details from Attention Class in Megatron Core Repo: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py#L49.


CPU Offloading
--------------

Overview
^^^^^^^^

CPU Offloading in NeMo is a feature that reduces the peak memory usage of the GPU by offloading activations and inactive weights to CPU storage. NeMo supports offloading at the transformer layer level, allowing users to specify the number of transformer layers in their language model that require CPU offloading. During the forward pass, NeMo offloads activations at the optimal time and reloads them as needed during the backward pass.

Features
^^^^^^^^
. Supports training models with long sequence lengths by managing activation memory efficiently.
. Enables high batch sizes per GPU by offloading activation memory.
. Overlaps computation with data transfers (Host2Device and Device2Host) during offloading and reloading.

Usage
^^^^^
. Set cpu_offloading to True to enable CPU offloading.
. Set cpu_offloading_num_layers to a value between 0 and the total number of layers in the model minus one.
. Set cpu_offloading_activations and cpu_offloading_weights based on your needs to offload activations only, weights only, or both.
