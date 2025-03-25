.. _reset_learning_rate:

Reset Learning Rate
-------------------

The reset learning rate feature provides the ability to reset the learning rate for an existing checkpoint to its initial value (either 0 or ``optim.min_lr`` depending on the warmup steps) when performing continual pretraining.

Parameters
----------

* ``reset_lr`` (boolean): Enables resetting the learning rate to the initial value. This feature is only supported with the distributed optimizer and megatron_amp_O2.
* ``reset_lr_steps`` (boolean): Enables adjusting the learning rate's max_steps and decay_steps by subtracting the number of steps already completed at the checkpoint.

Use Cases
---------

1. ``reset_lr=True, reset_lr_steps=False``
When pretraining an existing checkpoint "from scratch" on a different dataset. The learning rate will be reset to its initial value. This allows the model to start training on a new dataset with the same learning rate dynamics as if it were starting from scratch.

2. ``reset_lr=True, reset_lr_steps=True``
When continuing training from an existing checkpoint with the same configuration. The learning rate will be reset to its initial value, and the ``max_steps`` and ``decay_steps`` for learning rate schedule will be recalculated by subtracting the number of steps already completed at the checkpoint. Specifically:
* ``max_steps`` will be recalculated as ``max_steps -= completed_steps``.
* ``decay_steps`` will be recalculated as ``decay_steps -= completed_steps``.

This ensures that the learning rate reaches the ``min_lr`` value by the end of training without changing the ``trainer.max_steps``:

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v2.0.0rc0/asset-post-reset-learning-rate-example.png
  :alt: 
  :width: 1080px


