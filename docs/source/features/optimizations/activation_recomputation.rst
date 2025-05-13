Activation Recomputation
========================

The input activations of network layers are stored in device memory and are used to compute gradients during back-propagation. When training a LLM with a long sequence length or a large micro-batch size, these input activations can quickly saturate device memory. Checkpointing a few activations and recomputing the rest is a common technique to reduce device memory usage.

Transformer Layer Recomputation
-------------------------------

NeMo supports transformer layer recomputation, which checkpoints the input of each transformer layer and recomputes the activations for the remaining layers. This technique significantly reduces activation memory usage. However, it increases the per-transformer layer computation cost by 30% due to re-executing the entire layer’s forward computation.
NeMo also supports partial transformer layer recomputation, which is beneficial when recomputing a few transformer layers help to reduce enough GPU memory for model to fit. This approach avoids the need to recompute the rest of the layers.

The recomputation config can be enabled via the transformer config `TransformerConfig <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py#L25>`_.

Transformer layer recomputation is enabled by setting ``recompute_method=full``.
The number of transformer layers to recompute can be set using ``recompute_num_layers`` along with ``recompute_method=block``.
If you set ``recompute_num_layers`` as the total number of layers, the inputs of all transformer layers are checkpointed and recomputed.
When training with the pipeline parallelism, ``recompute_num_layers`` indicates the layers per pipeline stage.
When using virtual pipelining, ``recompute_num_layers`` specifies the number of layers per virtual pipeline stage.

NeMo also supports checkpointing the input to a block of multiple consecutive transformer layers, meaning that a block of transformer layers becomes the recomputation granularity. This approach can save activation memory but increases the recomputation buffer memory. Thus, it is only beneficial for memory savings when the model has many transformer layers or when the intermediate layers of a transformer layer hold relatively small activation stores.
This recomputation mode can be enabled by setting ``recompute_method=uniform``, with the number of transformer layers per recomputation block set using ``recompute_num_layers``.

   .. code-block:: python

       from nemo.collections import llm
       from functools import partial

       # Load train recipe
       recipe = partial(llm.llama3_8b.pretrain_recipe)()

       recipe.model.config.recompute_method = "block"  # Enable 'block'-wise recomputation
       recipe.model.config.recompute_num_layers = 4

Self-attention Recomputation
----------------------------

NeMo supports the self-attention recomputation that checkpoints the inputs of each self-attention block and recomputes the intermediate input activations.
This cost-efficient method achieves high memory savings with minimal recomputation cost.
The intermediate layers of the self-attention block accounts for the majority of the activation memory.
This is because the input sizes of softmax, dropout, and qkv dot-product attention layers have the memory complexity of the sequence length square.
However, their recomputation cost is relatively smaller than the other linear projection layers that are linear with the hidden size square.

Self-attention recomputation is hard-enabled when using FlashAttention, which is supported in Transformer Engine.
Also, you can use the self-attention recomputation without FlashAttention by setting ``recompute_method=selective``.

   .. code-block:: python

       from nemo.collections import llm
       from functools import partial

       # Load train recipe
       recipe = partial(llm.llama3_8b.pretrain_recipe)()

       recipe.model.config.recompute_method = "selective"  # Enable selective recomputation

Scheme of full and selective checkpointing granularity:

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v2.0.0rc0/asset-post-activation-recomputation-exampe-2.jpg
    :align: center
    :alt: activation-recomputation-example-2

Scheme of uniform and block checkpointing method (full checkpointing granularity):

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v2.0.0rc0/asset-post-activation-recomputation-exampe-1.jpg
    :align: center
    :alt: activation-recomputation-example-1
