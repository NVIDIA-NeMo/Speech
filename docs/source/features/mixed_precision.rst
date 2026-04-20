.. _mix_precision:

Mixed Precision Training
========================

Mixed precision training enhances computational efficiency by conducting operations in low-precision
format while selectively maintaining critical data in single-precision. NeMo supports FP16 and BF16
precision via PyTorch Lightning, in mixed, true, and flash half-precision modes.

Precision Modes
---------------

PyTorch Lightning provides two categories of half-precision training:

**Mixed Precision** (``"bf16-mixed"`` / ``"16-mixed"``):
    Operations run in half-precision where safe, but model weights are kept in FP32.
    Gradients are computed in half-precision and accumulated in FP32. This is the safest
    option and generally a good default for ASR and TTS training.

**True Half Precision** (``"bf16-true"`` / ``"fp16-true"``):
    The entire model -- weights, activations, and gradients -- runs in half-precision.
    This uses less memory than mixed precision (no FP32 weight copy) and is faster,
    but requires the model to be numerically stable in half-precision.
    SpeechLM2 models use ``"bf16-true"`` by default for training.

**Flash Precision** (``"bf16-flash"`` / ``"fp16-flash"``):
    The model also runs in half-precision, but NeMo avoids Lightning's global
    default-dtype override and autocast context. This mode is intended for use
    with FlashOptim, a library of drop-in optimizers that reduces training
    memory by shrinking optimizer states, master weights, and gradients. In
    practice, this may be a better fit than AMP / mixed precision when
    optimizer-state memory or checkpoint size is the bottleneck, and may lead to
    improved convergence compared to Lightning's true half-precision as it keeps
    track of the residual between half and full precision weights.

Configuration
-------------

Set precision through the PyTorch Lightning trainer's ``precision`` argument.

In YAML (with Hydra):

.. code-block:: yaml

    trainer:
        precision: "bf16-mixed"    # BF16 mixed precision
        # precision: "16-mixed"    # FP16 mixed precision
        # precision: "bf16-true"   # True BF16 half precision
        # precision: "fp16-true"   # True FP16 half precision
        # precision: "bf16-flash"  # BF16 flash precision
        # precision: "fp16-flash"  # FP16 flash precision

In Python:

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
- **FP16** offers slightly higher throughput on some hardware but has a reduced dynamic range.
  In mixed precision mode, PyTorch Lightning handles loss scaling automatically.

HalfPrecisionForAudio
----------------------

Audio waveform tensors are sensitive to precision loss -- downcasting raw audio samples to half-precision
can degrade signal quality and hurt model accuracy. NeMo provides the ``HalfPrecisionForAudio`` plugin
(in ``nemo.utils.trainer_utils``) that extends Lightning's ``HalfPrecision`` plugin to preserve
full-precision for audio tensors while still casting all other inputs to half-precision.

Specifically, when the training mini-batch is a dictionary, any tensor whose key contains
the substring ``"audio"`` is kept in its original precision (typically FP32). All other floating-point
tensors are cast to the target half-precision dtype.

This plugin is used automatically when you launch training with NeMo's ``resolve_trainer_cfg``
utility (used by many NeMo example training scripts). When the trainer config specifies
``precision: "bf16-true"`` or ``precision: "fp16-true"``, ``resolve_trainer_cfg`` replaces
the precision setting with the ``HalfPrecisionForAudio`` plugin:

.. code-block:: python

    from nemo.utils.trainer_utils import resolve_trainer_cfg

    # In YAML: trainer.precision = "bf16-true"
    # resolve_trainer_cfg automatically installs HalfPrecisionForAudio
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))

If you construct the trainer manually, you can install the plugin directly:

.. code-block:: python

    from nemo.utils.trainer_utils import HalfPrecisionForAudio

    trainer = pl.Trainer(
        plugins=[HalfPrecisionForAudio("bf16-true")],
        devices=2,
        accelerator="gpu",
    )

FlashPrecision
---------------

NeMo provides the ``FlashPrecision`` plugin (in
``nemo.utils.trainer_utils``) primarily for FlashOptim-backed training.
According to the official FlashOptim README, FlashOptim provides drop-in
optimizer replacements that reduce training memory by compressing optimizer
states, master weights, and gradients while preserving the standard PyTorch
optimizer API.

FlashOptim generally expects the model parameters to already be in bf16/fp16,
while the optimizer manages reduced-precision state and master-weight
correction internally. ``FlashPrecision`` fits that model: it preserves the
same audio-aware input casting behavior as ``HalfPrecisionForAudio``, but does
not enter autocast and does not change PyTorch's global default dtype. This
avoids layering Lightning's global precision policy on top of FlashOptim's own
reduced-precision optimizer behavior.

When the trainer config specifies ``precision: "bf16-flash"`` or
``precision: "fp16-flash"``, ``resolve_trainer_cfg`` replaces the precision
setting with the ``FlashPrecision`` plugin:

.. code-block:: python

    from nemo.utils.trainer_utils import resolve_trainer_cfg

    # In YAML: trainer.precision = "bf16-flash"
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))

If you construct the trainer manually, you can install the plugin directly:

.. code-block:: python

    from nemo.utils.trainer_utils import FlashPrecision

    trainer = pl.Trainer(
        plugins=[FlashPrecision("bf16-flash")],
        devices=2,
        accelerator="gpu",
    )

If you're going to use ``FlashPrecision``, make sure to set up ``flashoptim`` optimizer, e.g.:

.. code-block:: yaml

    optimizer:
      _target_: flashoptim.FlashAdamW
      lr: 1e-4
      betas: [0.9, 0.999]
      weight_decay: 5e-2

