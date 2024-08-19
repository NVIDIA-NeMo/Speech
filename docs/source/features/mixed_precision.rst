.. _mix_precision:

Mixed Precision Training
------------------------

Mixed precision training significantly enhances computational efficiency by conducting operations in low-precision format, while selectively maintaining minimal data in single-precision to preserve critical information throughout key areas of the network. NeMo Framework now supports FP16, BF16, and FP8 via Transformer Engine (TE) across most models.


Half-precision Training
=======================

NeMo Framework supports half-precision FP16 and BF16 computation training via Megatron Core and the distributed optimizer.
This training recipe uses half-precision in all layer computation keeping the model states (optimizer states and master parameters) in single-precision.
To avoid repeated data type casting at each layer computation, Megatron Core keeps a separate copy of half-precision parameters that is updated after each optimizer step.

Half-precision training is enabled when setting ``precision`` to either of ``fp16-mixed`` or ``bf16-mixed`` along with  ``megatron_amp_O2=true``.
The parameter gradients are computed in the same half-precision, and the precision of gradient reduce-scatter across data-parallel GPUs can be set by ``optim.grad_sync_dtype``.

FP8 Training
============

NVIDIA H100 GPU introduced support for a new datatype, FP8 (8-bit floating point), enabling higher throughput of matrix multiplies and convolutions. NeMo Framework uses the NVIDIA `TransformerEngine <https://github.com/NVIDIA/TransformerEngine>`_ (TE) to leverage speedups from FP8. The following table summarizes the FP8 related arguments that can be configured in NeMo (`example config setting <https://github.com/NVIDIA/NeMo/blob/2e1814c9f031ad2aeeebad44597365e97253d2c4/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml/#L192-L200>`_). For a more detailed overview, refer to the TE `documentation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_, specifically the FP8 `format <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.Format>`_ and `recipe <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.DelayedScaling>`_.

.. list-table:: FP8 arguments
   :widths: 10 20
   :header-rows: 1

   * - Argument
     - Description
   * - transformer_engine
     - TE and related functionality can be enabled by setting this boolean argument to True. If this argument is not set to True, all subsequent arguments will be ignored.
   * - fp8
     - Enables FP8 training. For transformer networks, the QKV, projection, FC1, and FC2 matrix multiplications are executed using the fourth-generation NVIDIA H100 Tensor Cores with FP8 support.
   * - fp8_e4m3
     - Training recipe format for FP8. Activations, weights, and gradient tensors use the E4M3 format.
   * - fp8_hybrid
     - Training recipe format for FP8. Activations and weight tensors use the E4M3 format, whereas gradient use the E5M2 format to satisfy the additional dynamic range requirement for backward tensors. This is the default setting.
   * - fp8_margin
     - The scaling factor for FP8 tensors can be shifted by a factor of $2 ^ {margin}$ using this argument.
   * - fp8_amax_history_len
     - Window size for amax history. The window size determines how many instances of the most recent absolute max values (amaxes) are stored per tensor.
   * - fp8_amax_compute_algo
     - The choice between “max” and “most_recent” specifies how to select an amax value from the given history.
   * - reduce_amax
     - Indicates whether or not to perform an allreduce on the amax (absolute max) values for the FP8 tensors. Since the amax is directly used to compute the scaling factor for FP8 tensors, setting this argument ensures that the scaling factors for a tensor remain synchronized across devices in multi-GPU training configurations.
   * - fp8_params
     - Indicates whether to store module-level parameters in FP8. Enabling this option can reduce memory consumption by eliminating the need to store a copy of weights in higher precision for cases where these weights are externally maintained, such as master parameters in the optimizer. For more information, refer to the `fp8_model_init <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.fp8_model_init>`_ API in TE.

Resources
^^^^^^^^^

- `Transformer Engine documentation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_
- `Intro to FP8, floating point formats, and mixed precision training <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Introduction-to-FP8>`_
- `Performance optimizations <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html>`_ that are natively supported in NeMo Framework by enabling FP8 training with TE
- `Transformer Engine installation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_
