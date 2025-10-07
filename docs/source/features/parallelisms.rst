.. _parallelisms:

Parallelisms
============

NeMo Megatron supports various data-parallel and model-parallel deep learning workload deployment methods, which can be mixed together arbitrarily.

Data Parallelism
----------------

Data Parallelism (DP) replicates the model across multiple GPUs.
Data batches are evenly distributed between GPUs and the data-parallel GPUs process them independently.
While the computation workload is efficiently distributed across GPUs, inter-GPU communication is required in order to keep the model replicas consistent between training steps.

Distributed Data Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distributed Data Parallelism (DDP) keeps the model copies consistent by synchronizing parameter gradients across data-parallel GPUs before each parameter update.
More specifically, it sums the gradients of all model copies using all-reduce communication collectives.

.. image:: ../nlp/nemo_megatron/images/ddp.gif
    :align: center
    :width: 800px
    :alt: Distributed Data Parallel

Distributed Optimizer
^^^^^^^^^^^^^^^^^^^^^

Distributed optimizer is a memory-optimized data-parallel deployment method.
It shards the optimizer states and the high-precision master parameters across data-parallel GPUs instead replicating them.
At the parameter optimizer step, each data-parallel GPU updates its shard of parameters.
Since each GPU needs its own gradient shard, the distributed optimizer conducts reduce-scatter of the parameter gradients instead of all-reduce of them.
Then, the updated parameter shards are all-gathered across data-parallel GPUs.
This approach significantly reduces the memory need of large-scale LLM training.
Also, when the precision of the gradient is higher than the parameter precision, the split execution of gradient reduce-scatter and parameter all-gather can reduce the total communication volume.
This split collective execution increases the total computation to overlap with the communication, which improves the overlap opportunity.

Enable Data Parallelism
~~~~~~~~~~~~~~~~~~~~~~~

In NeMo Framework, DDP is the default parallel deployment method.
This means that the total number of GPUs corresponds to the size of the DP group, and training an LLM with model parallelism decreases the size of the DP group.

Currently, the NeMo Framework supports optimizer distribution only for the Megatron Core Adam distributed optimizer.
To enable the distributed adam optimizer, set up ``distributed_fused_adam_with_cosine_annealing`` optimizer recipe from ``nemo.collections.llm.recipes.optim.adam`` or you can create your own optimizer recipe.

.. code-block:: python
    
    # Use optimizer recipe created by NeMo team
    from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing

    optim = distributed_fused_adam_with_cosine_annealing(max_lr=3e-4)
    optim.config.bf16 = True

    # Create your own optimizer recipe with cosine annealing scheduler
    import nemo_run as run
    from megatron.core.optimizer import OptimizerConfig

    from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule, PytorchOptimizerModule

    @run.cli.factory
    def distributed_optimizer_recipe(
        precision: str = "bf16-mixed",  # or "16-mixed"
        warmup_steps: int = 1000,
        constant_steps: int = 1000,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        max_lr: float = 1e-4,
        min_lr: float = 1e-5,
        clip_grad: float = 1.0,
    ) -> run.Config[PytorchOptimizerModule]:

        opt_cfg = run.Config(
            OptimizerConfig,
            optimizer="adam",
            lr=max_lr,
            weight_decay=0.1,
            bf16=precision == "bf16-mixed",
            fp16=precision == "16-mixed",
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=1e-5,
            use_distributed_optimizer=True,
            clip_grad=clip_grad,
        )

        sched = run.Config(
            CosineAnnealingScheduler,
            warmup_steps=warmup_steps,
            constant_steps=constant_steps,
            min_lr=min_lr,
        )

        return run.Config(
            MegatronOptimizerModule,
            config=opt_cfg,
            lr_scheduler=sched,
        )

For more optimzier options, please visit `this page <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py>`_.

..
    FSDP is not supported in NeMo 2.0 yet.
    Fully-Shared Data Parallelism
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    NeMo Framework supports Fully-Sharded Data Parallelism (FSDP), which shards parameter gradients and low-precision parameters for computation. This is in addition to the model states that the distributed optimizer shards, including optimizer states and high-precision parameters.
    Since FSDP shards the entire model states, it ensures linear model state memory savings with increasing DP size.
    FSDP is preferred for LLM training with unbalanced workloads between pipeline stages (or Transformer layers) or with a large vocabulary size, where pipelining would cause significant computation bubbles due to workload imbalance.
    Additionally, FSDP eliminates the need to search for performance-optimal mappings with 3D parallelism (TP/PP/DP) because it operates within a single parallelization domain.


    NeMo Framework uses `PyTorch's FSDP interface <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_ to shard LLM model states, flattening the parameters of each transformer layer and partitioning them across data-parallel GPUs.
    FSDP introduces collective operations across data-parallel GPUs, including all-gather for parameter computation and reduce-scatter for parameter gradients.
    The all-gather operation occurs during both the network forward and back-propagation phases, while the gradient reduce-scatter operation happens only during back-propagation.
    These FSDP communications are overlapped with transformer layer computations.

    Setting ``fsdp=true`` enables FSDP.
    The mixed precision recipe can be set by ``precision`` knob, which determines both the computation and communication precisions.
    Also, one can use ``grad_reduce_dtype`` to override the gradient reduction precision specifically.


Model Parallelism
-----------------

Model Parallelism (MP) is a distributed model deployment method that partitions the model parameters across GPUs to reduce the need of per-GPU memory.
NeMo Framework supports various model-parallel methods, which can be mixed to maximize LLM training performance.

Tensor Parallelism
^^^^^^^^^^^^^^^^^^

Tensor Parallelism (TP) is a model-parallel partitioning method that distributes the parameter tensor of an individual layer across GPUs.
In addition to reducing model state memory usage, it also saves activation memory as the per-GPU tensor sizes shrink.
However, the reduced per-GPU tensor size increases CPU overhead due to smaller per-GPU kernel workloads.

.. image:: ../nlp/nemo_megatron/images/tp1.png
    :align: center
    :width: 800px
    :alt: Tensor Parallel

.. image:: ../nlp/nemo_megatron/images/tp2.png
    :align: center
    :width: 800px
    :alt: Tensor Parallel

Enable Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable TP in the NeMo Framework, configure the ``tensor_model_parallel_size`` parameter in the model configuration. This parameter determines the number of GPUs among which the model's tensors are partitioned.

Set ``tensor_model_parallel_size`` to greater than ``1`` to enable intra-layer model parallelism.

   .. code-block:: python

       from nemo.collections import llm
       from functools import partial

       # Load train recipe
       recipe = partial(llm.llama3_8b.pretrain_recipe)()

       # Set tensor model parallel size
       recipe.trainer.strategy.tensor_model_parallel_size = 2

Set tensor parallelism directly from CLI:

    .. code-block:: bash
      
      nemo llm pretrain --factory llama3_8b trainer.strategy.tensor_model_parallel_size=2

Implement Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo Framework integrates TP through the implementation from Megatron Core. To understand how TP is activated within transformer blocks, refer to the code in the following repository: `Megatron-LM Transformer Block <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py>`__.

For detailed API usage and additional configurations, consult the `Megatron Core Developer Guide <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html>`_.

..
    FSDP is not supported in NeMo 2.0 yet.

    FSDP with Tensor Parallelism
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    NeMo Framework supports FSDP along with TP. This is done by restricting the model state sharding to the data-parallel domain.
    Using FSDP with TP can be helpful when the model doesn't have sufficient parallelism to deploy on a large-scale training system with the data-parallel mapping. For example, running a model with the global batch size of 1024 on 2048 GPUs.
    Also, TP enables FSDP feasibility by reducing the model state size and the activation size per GPU, thus lower the FSDP communication overhead and the activation memory overhead.

    Using both FSDP and TP works by enabling FSDP (``fsdp=true``) and setting ``tensor_model_parllel_size > 1``.
    Unset the ``CUDA_DEVICE_MAX_CONNECTIONS`` environment variable to set the number of GPU kernel queues, allowing the overlap of FSDP communication with computation kernels.

Pipeline Parallelism
^^^^^^^^^^^^^^^^^^^^

Pipeline Parallelism (PP) is a technique that assigns consecutive layers or segments of a neural network to different GPUs. This division allows each GPU to process different stages of the network sequentially.

.. image:: ../nlp/nemo_megatron/images/pp.gif
    :align: center
    :width: 800px
    :alt: Pipeline Parallel


Enable Pipeline Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To utilize Pipeline Parallelism (PP) in NeMo Framework, set the ``pipeline_model_parallel_size`` parameter in the model's configuration. This parameter specifies the number of GPUs among which the model's layers are distributed.

Set ``pipeline_model_parallel_size`` to a value greater than ``1`` to enable inter-layer model parallelism.

.. code-block:: python

       from nemo.collections import llm
       from functools import partial
       
       # Load train recipe
       recipe = partial(llm.llama3_8b.pretrain_recipe)()

       # Set pipeline model parallel size
       recipe.trainer.strategy.pipeline_model_parallel_size = 2

Set pipeline parallelism directly from CLI:

    .. code-block:: bash
      
      nemo llm pretrain --factory llama3_8b trainer.strategy.pipeline_model_parallel_size=2

Interleaved Pipeline Parallel Schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To minimize the pipeline bubble, the computation on each GPU can be divided into multiple subsets of layers (referred to as model chunks), rather than a single contiguous block. For instance, instead of each GPU processing a continuous set of four layers, it might handle two model chunks with two layers each.
    
    .. code-block:: python

       from nemo.collections import llm
       from functools import partial
    
       # Load train recipe
       recipe = partial(llm.llama3_8b.pretrain_recipe)()

       # Set pipeline model parallel size > 1 and enable interleaved pipeline
       recipe.trainer.strategy.pipeline_model_parallel_size = 2
       recipe.trainer.strategy.virtual_pipeline_model_parallel_size = 2

Enable interleaved pipeline directly from CLI:

    .. code-block:: bash
      
      nemo llm pretrain --factory llama3_8b trainer.strategy.pipeline_model_parallel_size=2 trainer.strategy.virtual_pipeline_model_parallel_size=2

For more insights into this approach, see our detailed blog: `Scaling Language Model Training <https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism>`_.

Implement Pipeline Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NeMo Framework implementation of PP leverages functionalities from Megatron Core. For a practical example of how PP is implemented within transformer blocks in NeMo, you can inspect the following codebase: `Megatron-LM Transformer Block <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py>`_.

For more detailed API usage and configurations related to PP, visit the `Megatron Core Developer Guide <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html>`_.

Expert Parallelism
^^^^^^^^^^^^^^^^^^
Expert Parallelism (EP) is a type of model parallelism that distributes experts of an MoE across GPUs.
Unlike other model-parallel techniques, EP is applied to only the expert layers thus does not impact the parallel mapping of the rest of layers.

.. image:: ../nlp/nemo_megatron/images/ep.png
    :align: center
    :width: 800px
    :alt: Expert Parallelism

Enable Expert Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable EP, set ``expert_model_parallel_size`` to the expert parallel size you want in the ``MegatronStrategy``'s arguments.
For example, if the model has eight experts (``num_moe_experts=8``), then setting ``expert_model_parallel_size=4`` results in each GPU processing two experts.
The number of experts should be divisible by the expert parallel size.

   .. code-block:: python

       from nemo.collections import llm
       from functools import partial
       
       # Load train recipe
       recipe = partial(llm.mixtral_8x7b.pretrain_recipe)()

       # Set expert model parallel size
       recipe.trainer.strategy.expert_model_parallel_size = 4

Set expert parallelism directly from CLI:

    .. code-block:: bash
      
      nemo llm pretrain --factory mixtral_8x7b trainer.strategy.expert_model_parallel_size=4


Enable Expert Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To enable ETP, set ``expert_tensor_parallel_size`` to the size you want in the ``MegatronStrategy``'s arguments. For example:

   .. code-block:: python

       from nemo.collections import llm
       from functools import partial

       # Load train recipe
       recipe = partial(llm.mixtral_8x7b.pretrain_recipe)()

       # Set expert tensor parallel size
       recipe.trainer.strategy.expert_tensor_parallel_size = 4

Set expert tensor parallelism directly from CLI:

    .. code-block:: bash

      nemo llm pretrain --factory mixtral_8x7b trainer.strategy.expert_tensor_parallel_size=4


Expert Parallelism Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NeMo Framework implementation of EP uses functionality from Megatron Core. Please consult the `Megatron Core MoE layer <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py#L42>`_ for more MoE implementation details.



Activation Partitioning
-----------------------

In LLM training, a large memory space is needed to store the input activations of the network layers.
NeMo Framework provides effective activation distribution methods, which is critical in training LLM with a large sequence length or large per-GPU micro-batch size.

Sequence Parallelism
^^^^^^^^^^^^^^^^^^^^

Sequence Parallelism (SP) extends tensor-level model parallelism by distributing computing load and activation memory across multiple GPUs along the sequence dimension of transformer layers. This method is particularly useful for portions of the layer that have previously not been parallelized, enhancing overall model performance and efficiency.

.. image:: ../nlp/nemo_megatron/images/sp.png
    :align: center
    :width: 800px
    :alt: Sequence Parallel

Enable Sequence Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To utilize SP in NeMo Framework, set the ``sequence_parallel`` parameter to ``True`` in the model's configuration. Note that this feature is effective only when the tensor parallel size (``tensor_model_parallel_size``) is greater than ``1``.

   .. code-block:: python

       from nemo.collections import llm
       from functools import partial
       
       # Load train recipe
       recipe = partial(llm.llama3_8b.pretrain_recipe)()
       
       # Set tensor model parallel size and enable sequence parallelism
       recipe.trainer.strategy.tensor_model_parallel_size = 2
       recipe.trainer.strategy.sequence_parallelism = True

Enable sequence parallelism directly from CLI:

    .. code-block:: bash
      
      nemo llm pretrain --factory llama3_8b trainer.strategy.tensor_model_parallel_size=2 trainer.strategy.sequence_parallelism=True

Implement Sequence Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NeMo Framework implementation of SP utilizes functionality from Megatron Core. For an in-depth look at how Sequence Parallelism is integrated into the Megatron Core architecture, you can examine the source code here: `Megatron-LM Sequence Parallel Source Code <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py>`_.

Context Parallelism
^^^^^^^^^^^^^^^^^^^

Context Parallelism (CP) is a method for parallelizing the processing of neural network activations across multiple GPUs, partitioning the input tensors in the sequence dimension.
Unlike SP, which partitions the activations of specific layers, CP divides the activations of all layers.

Enable Context Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~

To activate CP in the NeMo Framework, set the ``context_parallel_size`` parameter in the model configuration. This parameter specifies the number of GPUs among which the model's sequence activations are distributed.

Set ``context_parallel_size`` to a value greater than ``1`` to enable sequence-wide model parallelism.

   .. code-block:: python

       from nemo.collections import llm
       from functools import partial

       # Load train recipe
       recipe = partial(llm.llama3_8b.pretrain_recipe)()

       # Set context parallel size
       recipe.trainer.strategy.context_parallel_size = 2

Set ``context_parallel_size`` directly from CLI:

    .. code-block:: bash
      
      nemo llm pretrain --factory llama3_8b model.config.context_parallel_size=2

The configuration can be found and modified here: `NeMo Megatron Core Context Config <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html>`_.

Implement Context Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo Framework leverages functionalities from both Megatron Core and Transformer Engine to implement CP efficiently. During forward propagation, each GPU handles a segment of the sequence, storing only the necessary Key and Value (KV) pairs. In the backward pass, these KV pairs are reassembled across GPUs using advanced communication schemes like all-gather and reduce-scatter transformed into point-to-point communications in a ring topology. This method reduces the memory footprint significantly while maintaining computational efficiency.

Visit our source code for more insights into the implementation:
- `Megatron Core wrappers for Transformer Engine <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/extensions/transformer_engine.py>`_
- `Transformer Engine attention modules <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention>`_


Parallelism Nomenclature
^^^^^^^^^^^^^^^^^^^^^^^^

The following figure illustrates some terms that you may encounter in the NeMo Megatron codebase.

.. image:: ../nlp/nemo_megatron/images/pnom.gif
    :align: center
    :width: 800px
    :alt: Parallelism nomenclature
