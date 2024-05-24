.. _parallelisms:

Parallelisms
------------

NeMo Megatron supports five types of parallelism (which can be mixed together arbitrarily).

Data Parallelism
^^^^^^^^^^^^^^^^

Data Parallelism (DP) creates identical copies of the model across
multiple GPUs. Data batches are distributed between GPUs so that the
GPUs can process them independently. While compute is efficiently
distributed between GPUs, communication is required in order to keep
the model copies consistent with each other.

Distributed Data Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distributed Data Parallelism (DDP) keeps model copies consistent by
synchronizing parameter gradients before each optimization step. More
specifically, it sums gradients over all model copies using an
all-reduce communication collective.

.. image:: ../nlp/nemo_megatron/images/ddp.gif
    :align: center
    :width: 800px
    :alt: Distributed Data Parallel

Distributed Optimizer (ZeRO-1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZeRO-1 algorithm keeps model copies consistent by sharding the
optimizer state between GPUs. During each optimization step, the
parameter gradients are first summed and sharded (with a
reduce-scatter collective), each GPU applies an optimization to its
local shard of the parameters, and the updated parameter shards are
broadcast to update all of the model copies (with an all-gather
collective). This approach is attractive for large models since
sharding the optimizer state can significantly reduce its memory
footprint on individual GPUs. It also has, in theory, the same
communication volume as DDP and its communication pattern has more
opportunities for overlapping with compute.

Enable Data Parallelism
~~~~~~~~~~~~~~~~~~~~~~~

DDP is the default parallelism scheme when NeMo is run on multiple
GPUs. Enabling other parallelism schemes in the model configuration
will decrease the size of the DP group, that is the number of
identical model copies.

To enable the distributed optimizer, set
``model.optim.name=distributed_fused_adam`` in the model
configuration. It can be configured with the following options:

===========================  =========  ==================================================================================================================================
Option                       Default    Description
===========================  =========  ==================================================================================================================================
``dtype``                    fp32       Optimizer state datatype
``grad_sync_dtype``          ``dtype``  Gradient reduce-scatter datatype
``overlap_grad_sync``        True       Overlap gradient reduce-scatter with compute
``overlap_param_sync``       False      Overlap parameter all-gather with compute
``bucket_cap_mb``            100        Buffer size (in MiB) for internal state and workspaces. Larger buckets have lower runtime overheads but may increase memory usage.
``contiguous_param_buffer``  False      Allocate parameters as views into a large buffer. Helps avoid some data copies.
``contiguous_grad_buffer``   True       Allocate parameter gradients as views into a large buffer. Helps avoid some data copies.
===========================  =========  ==================================================================================================================================

See the keyword arguments in `Apex DistributedFusedAdam <https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/distributed_fused_adam.py>`_ and `NeMo MegatronDistributedFusedAdam <https://github.com/NVIDIA/NeMo/blob/main/nemo/core/optim/distributed_adam.py>`_ for a full list of distributed optimizer options.

Implement Data Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~

DDP in NeMo either uses PyTorch
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
(default) or a custom implementation (if custom multi-precision
training is enabled with ``megatron_amp_O2``).

The distributed optimizer in NeMo is built on top of
`DistributedFusedAdam <https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/distributed_fused_adam.py>`_
from Apex.

Tensor Parallelism
^^^^^^^^^^^^^^^^^^

Tensor Parallelism (TP) is a method for distributing a model's computation across multiple GPUs by splitting tensors into non-overlapping pieces. This allows different parts of the tensor to be processed simultaneously on separate GPUs, enhancing performance and enabling the training of larger models.

.. image:: ../nlp/nemo_megatron/images/tp.gif
    :align: center
    :width: 800px
    :alt: Tensor Parallel

Enable Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable TP in the NeMo framework, configure the ``tensor_model_parallel_size`` parameter in the model configuration. This parameter determines the number of GPUs among which the model's tensors are partitioned.

**For Tensor Parallelism**:

Set ``tensor_model_parallel_size`` to greater than ``1`` to enable intra-layer model parallelism.

   .. code-block:: yaml

       tensor_model_parallel_size: 1  # Example to enable Tensor Parallelism

The configuration file can be adjusted here: `NeMo Megatron GPT Config <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml#L65>`_.

Implement Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo integrates Tensor Parallelism through the implementation from Megatron Core. To understand how TP is activated within transformer blocks, refer to the code in the following repository: `Megatron-LM Transformer Block <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py>`_.

For detailed API usage and additional configurations, consult the `Megatron Core Developer Guide <https://docs.nvidia.com/Megatron-Core/developer-guide/latest/api-guide/tensor_parallel.html>`_.

Pipeline Parallelism
^^^^^^^^^^^^^^^^^^^^

Pipeline Parallelism (PP) is a technique that assigns consecutive layers or segments of a neural network to different GPUs. This division allows each GPU to process different stages of the network sequentially.

.. image:: ../nlp/nemo_megatron/images/pp.gif
    :align: center
    :width: 800px
    :alt: Pipeline Parallel


Enable Pipeline Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To utilize PP in the NeMo framework, you need to set the ``pipeline_model_parallel_size`` parameter in the model's configuration. This parameter specifies the number of GPUs among which the model's layers are distributed.

**For Pipeline Parallelism**:

Set ``pipeline_model_parallel_size`` to a value greater than ``1`` to enable inter-layer model parallelism.

   .. code-block:: yaml

       pipeline_model_parallel_size: 1  # Example to enable Pipeline Parallelism

Adjust the configuration accordingly here: `NeMo Megatron GPT Config <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml#L66>`_.

Interleaved Pipeline Parallel Schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To minimize the pipeline bubble, the computation on each GPU can be divided into multiple subsets of layers (referred to as model chunks), rather than a single contiguous block. For instance, instead of each GPU processing a continuous set of four layers, it might handle two model chunks with two layers each.

   .. code-block:: yaml

       virtual_pipeline_model_parallel_size: 2 # Set for interleaved pipeline

For more insights into this approach, see our detailed blog: `Scaling Language Model Training <https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism>`_.

Implement Pipeline Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NeMo implementation of PP leverages functionalities from Megatron Core. For a practical example of how PP is implemented within transformer blocks in NeMo, you can inspect the following codebase: `Megatron-LM Transformer Block <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py>`_.

For more detailed API usage and configurations related to PP, visit the `Megatron Core Developer Guide <https://docs.nvidia.com/Megatron-Core/developer-guide/latest/api-guide/tensor_parallel.html>`_.

Sequence Parallelism
^^^^^^^^^^^^^^^^^^^^

Sequence Parallelism extends tensor-level model parallelism by distributing computing load and activation memory across multiple GPUs along the sequence dimension of transformer layers. This method is particularly useful for portions of the layer that have previously not been parallelized, enhancing overall model performance and efficiency.

.. image:: ../nlp/nemo_megatron/images/sp.gif
    :align: center
    :width: 800px
    :alt: Sequence Parallel

Enable Sequence Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To utilize Sequence Parallelism in NeMo, set the ``sequence_parallel`` parameter to ``True`` in the model's configuration. Note that this feature is effective only when the tensor parallel size (``tensor_model_parallel_size``) is greater than ``1``.

   .. code-block:: yaml

       sequence_parallel: True  # Enable Sequence Parallelism

For further information on configuration, refer to the following documentation: `NeMo Megatron GPT Config <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml#L66>`_.

Implement Sequence Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NeMo implementation of Sequence Parallelism utilizes functionality from Megatron Core. For an in-depth look at how Sequence Parallelism is integrated into the Megatron Core architecture, you can examine the source code here: `Megatron-LM Sequence Parallel Source Code <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py>`_.

Context Parallelism
^^^^^^^^^^^^^^^^^^^

Context Parallelism (CP) is a method for parallelizing the processing of neural network activations across multiple GPUs, focusing on the sequence dimension of the input data. Unlike Sequence Parallelism (SP) that only partitions specific types of activations, CP divides all network activations along the sequence dimension.

Enable Context Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~

To activate CP in the NeMo framework, set the ``context_parallel_size`` parameter in the model configuration. This parameter specifies the number of GPUs among which the model's sequence activations are distributed.

**For Context Parallelism**:

Set ``context_parallel_size`` to a value greater than ``1`` to enable sequence-wide model parallelism.

   .. code-block:: yaml

       context_parallel_size: 1  # Example to enable Context Parallelism

The configuration can be found and modified here: `NeMo Megatron Core Context Config <https://docs.nvidia.com/Megatron-Core/developer-guide/latest/api-guide/context_parallel.html>`_.

Implement Context Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo leverages functionalities from both Megatron Core and Transformer Engine to implement CP efficiently. During forward propagation, each GPU handles a segment of the sequence, storing only the necessary Key and Value (KV) pairs. In the backward pass, these KV pairs are reassembled across GPUs using advanced communication schemes like all-gather and reduce-scatter transformed into point-to-point communications in a ring topology. This method reduces the memory footprint significantly while maintaining computational efficiency.

Visit our source code for more insights into the implementation:
- `Megatron Core wrappers for Transformer Engine <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/custom_layers/transformer_engine.py>`_
- `Transformer Engine attention modules <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py>`_


Expert Parallelism
^^^^^^^^^^^^^^^^^^
Expert Parallelism (EP) is a type of model parallelism that distributes experts of an MoE across GPUs.

.. image:: ../nlp/nemo_megatron/images/ep.png
    :align: center
    :width: 800px
    :alt: Expert Parallelism

Enable Expert Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable EP, set ``model.expert_model_parallel_size`` to the desired expert parallel size. For example, if the model has six experts (``model.num_moe_experts=6``), then setting ``model.expert_model_parallel_size=3`` results in each GPU processing two experts. The number of experts should be divisible by the expert parallel size.

   .. code-block:: yaml

       expert_model_parallel_size: 3  # Set EP to 3

For further information on configuration, refer to the following documentation: `NeMo Megatron GPT Config <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml#L68>`_.


Implement Expert Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NeMo implementation of Expert Parallelism uses functionality from Megatron Core. Please consult the `Megatron Core MoE layer <https://github.com/NVIDIA/Megatron-LM/blob/e2ec14ab5690fead7e33760b0f8fb20c83b4fd1f/megatron/core/transformer/moe/moe_layer.py#L29>`_ for more MoE implementation details.


Parallelism nomenclature
^^^^^^^^^^^^^^^^^^^^^^^^

The following figure illustrates some terms that you may encounter in the NeMo Megatron codebase.

.. image:: ../nlp/nemo_megatron/images/pnom.gif
    :align: center
    :width: 800px
    :alt: Parallelism nomenclature
