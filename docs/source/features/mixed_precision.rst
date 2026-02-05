.. _mix_precision:

Mixed Precision Training
========================

Mixed precision training enhances computational efficiency by conducting operations in low-precision
format while selectively maintaining critical data in single-precision. NeMo supports FP16 and BF16
mixed precision training via PyTorch Lightning.

Enabling Mixed Precision
------------------------

Mixed precision is configured through the PyTorch Lightning trainer's ``precision`` argument:

- ``"bf16-mixed"``: BF16 mixed precision (recommended for GPUs with BF16 support, e.g. A100, H100)
- ``"16-mixed"``: FP16 mixed precision

Example configuration in YAML:

.. code-block:: yaml

    trainer:
        precision: "bf16-mixed"

Or when configuring the trainer in Python:

.. code-block:: python

    import lightning.pytorch as pl

    trainer = pl.Trainer(
        precision="bf16-mixed",
        devices=2,
        accelerator="gpu",
    )

Choosing a Precision Format
----------------------------

- **BF16** has the same dynamic range as FP32, which makes it more numerically stable and generally
  easier to use. It is the recommended choice for most Speech AI training workloads.
- **FP16** offers slightly higher throughput on some hardware but requires loss scaling to handle
  its reduced dynamic range.

PyTorch Lightning handles loss scaling automatically when using ``"16-mixed"`` precision.
