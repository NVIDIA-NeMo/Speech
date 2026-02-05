Checkpoints
===========


In this section, we present key functionalities of NVIDIA NeMo related to checkpoint management.

Checkpoint Formats
------------------

A ``.nemo`` checkpoint is fundamentally a tar file that bundles the model configurations (specified inside a YAML file), model weights (inside a ``.ckpt`` file), and other artifacts like tokenizer models or vocabulary files. This consolidated design streamlines sharing, loading, tuning, evaluating, and inference.

In contrast, the ``.ckpt`` file, created during PyTorch Lightning training, contains both the model weights and the optimizer states, and is usually used to resume training.
