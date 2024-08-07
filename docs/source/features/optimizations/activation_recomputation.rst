Activation Recomputation
========================

The input activations of network layers are stored in the device memory to compute the gradients in back-propagation.
The input activation stores easily saturate the device memory when training a LLM with a large sequence length or a large micro-batch size.
Check-pointing a few activations and recomputing the rest of activations is a common technique to reduce the need of device memory.

Transformer Layer Recomputation
-------------------------------

NeMo supports Transformer layer recomputation that checkpoints the input of each Transformer layer and recomputes the activations on the rest of the layers.
Transformer layer recomputation significantly reduces the activation memory usage.
However, this approach increases per-Transformer layer computation cost by 30%, which comes from re-executing the entire layer forwarding computation.
NeMo also supports partial Transformer layer recomputation, which is beneficial when recomputing a few Transformer layers would fit the training workload on GPU memory.
This would avoid recomputing the rest of layers.

Transformer layer recomputation is enabled by setting ``activations_checkpoint_granularity=full``.
The number of Transformer layers to recompute can be set using ``activations_checkpoint_num_layers`` along with ``activations_checkpoint_method=block``.
If one sets ``activations_checkpoint_num_layers`` as the total number of layers, the inputs of all Transformer layers are check-pointed and recomputed.
When training with the pipeline parallelism, ``activations_checkpoint_num_layers`` indicates the layers per pipeline stage.
If the virtual pipelining is used, ``activations_checkpoint_num_layers`` means the layers per virtual pipeline stage.

NeMo also supports checkpointing the input to a block of multiple consecutive Transformer layers meaning that a block of Transformer layers becomes the recomputation granularity.
This can further save activation memory at the cost of increasing the recomputation buffer memory.
Thus, it is only beneficial for memory savings when the model has many Transformer layers or the intermediate layers of a Transformer layer hold relatively small activation stores.
This recomputation mode can be enabled by setting ``activations_checkpoint_method=uniform``, and the number of Transformer layers per recomputation block is set using ``activations_checkpoint_num_layers``.

Self-attention Recomputation
----------------------------

NeMo supports the self-attention recomputation that checkpoints the inputs of each self-attention block and recomputes the intermediate input activations.
This is a cost-efficient recomputation method; achieves high memory saving with lost recomputation cost.
The intermediate layers of the self-attention block accounts for the majority portion the activation memory.
This is because the input sizes of softmax, dropout, and qkv dot-product attention layers have the memory complexity of the sequence length square.
However, their recomputation cost is relatively smaller than the other linear projection layers that are linear with the hidden size square.

Self-attention recomputation is hard-enabled when using FlashAttention, which is supported in Transformer Engine.
Also, a user can use the self-attention recomputation without FlashAttention by setting ``activations_checkpoint_granularity=selective``.

Scheme of full and selective checkpointing granularity:

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v2.0.0rc0/asset-post-activation-recomputation-exampe-2.jpg
    :align: center
    :alt: activation-recomputation-example-2
    :scale: 50%

Scheme of uniform and block checkpointing method (full checkpointing granularity):

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v2.0.0rc0/asset-post-activation-recomputation-exampe-1.jpg
    :align: center
    :alt: activation-recomputation-example-1
    :scale: 50%