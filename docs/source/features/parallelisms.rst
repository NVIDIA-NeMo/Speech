.. _parallelisms:

Parallelisms
------------

NeMo Megatron supports 5 types of parallelisms (which can be mixed together arbitrarily):

Distributed Data Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Distributed Data Parallelism (DDP) creates idential copies of the model across multiple GPUs.

.. image:: ../nlp/nemo_megatron/images/ddp.gif
    :align: center
    :width: 800px
    :alt: Distributed Data Parallel


Tensor Parallelism
^^^^^^^^^^^^^^^^^^

**Tensor Parallelism (TP)** is a method for distributing a model's computation across multiple GPUs by splitting tensors into non-overlapping pieces. This allows different parts of the tensor to be processed simultaneously on separate GPUs, enhancing performance and enabling the training of larger models.

.. image:: ../nlp/nemo_megatron/images/tp.gif
    :align: center
    :width: 800px
    :alt: Tensor Parallel

Enable Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable TP in the NeMo framework, configure the ``tensor_model_parallel_size`` parameter in the model configuration. This parameter determines the number of GPUs among which the model's tensors are partitioned.

**For Tensor Parallelism**:
   - Set ``tensor_model_parallel_size`` to greater than ``1`` to enable intra-layer model parallelism.

   .. code-block:: yaml

       tensor_model_parallel_size: 1  # Example to enable Tensor Parallelism

The configuration file can be adjusted here: `NeMo Megatron GPT Config <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml#L65>`_.

Implement Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo integrates Tensor Parallelism through the implementation from Megatron Core. To understand how TP is activated within transformer blocks, refer to the code in the following repository: `Megatron-LM Transformer Block <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py>`_.

For detailed API usage and additional configurations, consult the `Megatron Core Developer Guide <https://docs.nvidia.com/Megatron Core/developer-guide/latest/api-guide/tensor_parallel.html>`_.

Pipeline Parallelism
^^^^^^^^^^^^^^^^^^^^

**Pipeline Parallelism (PP)** is a technique that assigns consecutive layers or segments of a neural network to different GPUs. This division allows each GPU to process different stages of the network sequentially.

.. image:: ../nlp/nemo_megatron/images/pp.gif
    :align: center
    :width: 800px
    :alt: Pipeline Parallel


Enable Pipeline Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To utilize PP in the NeMo framework, you need to set the ``pipeline_model_parallel_size`` parameter in the model's configuration. This parameter specifies the number of GPUs among which the model's layers are distributed.

**For Pipeline Parallelism**:
   - Set ``pipeline_model_parallel_size`` to a value greater than ``1`` to enable inter-layer model parallelism.

   .. code-block:: yaml

       pipeline_model_parallel_size: 1  # Example to enable Pipeline Parallelism

Adjust the configuration accordingly here: `NeMo Megatron GPT Config <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml#L66>`_.

Interleaved Pipeline Parallel Schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To minimize the pipeline bubble, the computation on each GPU can be divided into multiple subsets of layers (referred to as model chunks), rather than a single contiguous block. For instance, instead of each GPU processing a continuous set of four layers, it might handle two model chunks with two layers each. This method ensures that each GPU in the pipeline manages multiple stages rather than on a single contiguous block.

   .. code-block:: yaml

       virtual_pipeline_model_parallel_size: 2 # Set for interleaved pipeline

For more insights into this approach, see our detailed blog: `Scaling Language Model Training <https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism>`_.

Implement Pipeline Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo's implementation of PP leverages functionalities from Megatron Core. For a practical example of how PP is implemented within transformer blocks in NeMo, you can inspect the following codebase: `Megatron-LM Transformer Block <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py>`_.

For more detailed API usage and configurations related to PP, visit the `Megatron Core Developer Guide <https://docs.nvidia.com/Megatron Core/developer-guide/latest/api-guide/tensor_parallel.html>`_.

Sequence Parallelism
^^^^^^^^^^^^^^^^^^^^

**Sequence Parallelism** extends tensor-level model parallelism by distributing computing load and activation memory across multiple GPUs along the sequence dimension of transformer layers. This method is particularly useful for portions of the layer that have previously not been parallelized, enhancing overall model performance and efficiency.

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

NeMo's implementation of Sequence Parallelism utilizes functionality from Megatron Core. For an in-depth look at how Sequence Parallelism is integrated into the Megatron Core architecture, you can examine the source code here: `Megatron-LM Sequence Parallel Source Code <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py>`_.

Context Parallelism
^^^^^^^^^^^^^^^^^^^

**Context Parallelism (CP)** is a method for parallelizing the processing of neural network activations across multiple GPUs, focusing on the sequence dimension of the input data. Unlike Sequence Parallelism (SP) that only partitions specific types of activations, CP divides all network activations along the sequence dimension.

Enable Context Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~

To activate CP in the NeMo framework, set the ``context_parallel_size`` parameter in the model configuration. This parameter specifies the number of GPUs among which the model's sequence activations are distributed.

**For Context Parallelism**:
   - Set ``context_parallel_size`` to a value greater than ``1`` to enable sequence-wide model parallelism.

   .. code-block:: yaml

       context_parallel_size: 1  # Example to enable Context Parallelism

The configuration can be found and modified here: `NeMo Megatron Core Context Config <https://docs.nvidia.com/Megatron Core/developer-guide/latest/api-guide/context_parallel.html>`_.

Implement Context Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo leverages functionalities from both Megatron Core and transformer-engine to implement CP efficiently. During forward propagation, each GPU handles a segment of the sequence, storing only the necessary Key and Value (KV) pairs. In the backward pass, these KV pairs are reassembled across GPUs using advanced communication schemes like all-gather and reduce-scatter transformed into point-to-point communications in a ring topology. This method reduces the memory footprint significantly while maintaining computational efficiency.

Additionally, NeMo's CP supports integration with various forms of model parallelism such as TP (Tensor Parallelism), PP (Pipeline Parallelism), and DP (Data Parallelism), ensuring broad usability and flexibility in large-scale model training environments.

Visit our source code for more insights into the implementation:
- Megatron Core transformer engine: `Megatron Core <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/custom_layers/transformer_engine.py>`_
- Transformer Engine repository: `Transformer Engine Code <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py>`_


Expert Parallelism
^^^^^^^^^^^^^^^^^^
**Expert Parallelism (EP)** is a type of model parallelism that distributes experts of an MoE across GPUs.

.. image:: ../nlp/nemo_megatron/images/ep.png
    :align: center
    :width: 800px
    :alt: Expert Parallelism

Enable Expert Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable it users can pass ``model.expert_model_parallel_size=k``, where k is an integer with the desired
expert parallelism level, for example if the model has three experts (i.e. ``model.num_moe_experts=3``), we can specify
k=3 (i.e. via CLI using ``model.expert_model_parallel_size=3``). The number of experts should be exactly divisible by the ``expert_model_parallel_size``.

   .. code-block:: yaml

       expert_model_parallel_size: 3  # Set EP to 3

For further information on configuration, refer to the following documentation: `NeMo Megatron GPT Config <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml#L68>`_.


Implement Expert Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo's expert parallelism functionality is provided by Megatron-LM repository, please consult the corresponding `Moe-layer <https://github.com/NVIDIA/Megatron-LM/blob/e2ec14ab5690fead7e33760b0f8fb20c83b4fd1f/megatron/core/transformer/moe/moe_layer.py#L29>`_ for more moe implementation details.


Parallelism nomenclature
^^^^^^^^^^^^^^^^^^^^^^^^

When reading and modifying NeMo Megatron code you will encounter the following terms.

.. image:: ../nlp/nemo_megatron/images/pnom.gif
    :align: center
    :width: 800px
    :alt: Parallelism nomenclature
