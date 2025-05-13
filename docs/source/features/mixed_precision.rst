.. _mix_precision:

Mixed Precision Training
------------------------

Mixed precision training significantly enhances computational efficiency by conducting operations in low-precision format, while selectively maintaining minimal data in single-precision to preserve critical information throughout key areas of the network. NeMo Framework now supports FP16, BF16, and FP8 via Transformer Engine (TE) across most models.


Half-Precision Training
=======================

NeMo Framework supports half-precision FP16 and BF16 computation training via Megatron Core and the distributed optimizer.
This training recipe uses half-precision in all layer computation keeping the model states (optimizer states and master parameters) in single-precision.
To avoid repeated data type casting at each layer computation, Megatron Core keeps a separate copy of half-precision parameters that is updated after each optimizer step.

Half-precision training is enabled when setting trainer's ``plugins`` to either of ``fp16-mixed`` or ``bf16-mixed``.
The parameter gradients are computed in the same half-precision, and the precision of gradient reduce-scatter across data-parallel GPUs is set automatically according to the trainer's precision.

Implement Half-Precision Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import nemo_run as run

  from nemo import lightning as nl
  from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, fp16_mixed
  
  trainer_args = {TRAINER_ARGS}

  # Set up trainer with bf16 precision
  trainer_bf16 = run.Config(
    nl.Trainer,
    plugins=bf16_mixed(),
    **trainer_args,
  )

  # Set up trainer with fp16 precision
  trainer_fp16 = run.Config(
    nl.Trainer,
    plugins=fp16_mixed(),
    **trainer_args,
  )

It's also possible to change precision for a specific recipe:

.. code-block:: python

  from functools import partial

  from nemo.collections import llm
  from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, fp16_mixed

  # Load recipe
  recipe = partial(llm.llama3_8b.pretrain_recipe)()

  # Change precision
  recipe.trainer.plugins = fp16_mixed()

FP8 Training
============

NVIDIA H100 GPU introduced support for a new datatype, FP8 (8-bit floating point), enabling higher throughput of matrix multiplies and convolutions. NeMo Framework uses the NVIDIA `TransformerEngine <https://github.com/NVIDIA/TransformerEngine>`_ (TE) to leverage speedups from FP8. For a more detailed overview, refer to the TE `documentation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_, specifically the FP8 `format <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.Format>`_ and `recipe <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.DelayedScaling>`_.

.. list-table:: FP8 arguments
   :widths: 10 20
   :header-rows: 1

   * - Argument
     - Description
   * - fp8
     - The training recipe format for FP8 can be set to either 'hybrid' or 'e4m3', with 'hybrid' being the default. In the 'hybrid' format, activations and weight tensors use the E4M3 format, while gradients use the E5M2 format to meet the additional dynamic range requirements for backward tensors.
   * - fp8_margin
     - The scaling factor for FP8 tensors can be shifted by a factor of $2 ^ {margin}$ using this argument.
   * - fp8_amax_history_len
     - The window size for amax history. The window size determines how many instances of the most recent absolute max values (amaxes) are stored per tensor.
   * - fp8_amax_compute_algo
     - The choice between “max” and “most_recent” specifies how to select an amax value from the given history.
   * - fp8_params
     - Indicates whether to store module-level parameters in FP8. Enabling this option can reduce memory consumption by eliminating the need to store a copy of weights in higher precision for cases where these weights are externally maintained, such as master parameters in the optimizer. For more information, refer to the `fp8_model_init <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.fp8_model_init>`_ API in TE.

Implement FP8 Training
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import nemo_run as run

  from nemo import lightning as nl
  from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed, fp16_with_fp8_mixed
  
  trainer_args = {TRAINER_ARGS}
  fp8_args = {FP8_ARGS}

  # Set up trainer with bf16 & fp8 precision
  trainer_bf16_fp8 = run.Config(
    nl.Trainer,
    plugins=bf16_with_fp8_mixed(),
    **trainer_args,
    **fp8_args,
  )

  # Set up trainer with fp16 & fp8 precision
  trainer_fp16_fp8 = run.Config(
    nl.Trainer,
    plugins=fp16_with_fp8_mixed(),
    **trainer_args,
    **fp8_args,
  )

Resources
^^^^^^^^^

- `Transformer Engine documentation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_
- `Intro to FP8, floating point formats, and mixed precision training <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Introduction-to-FP8>`_
- `Performance optimizations <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html>`_ that are natively supported in NeMo Framework by enabling FP8 training with TE
- `Transformer Engine installation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_
