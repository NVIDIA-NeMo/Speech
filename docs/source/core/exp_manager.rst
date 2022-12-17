
.. _exp-manager-label:

Experiment Manager
==================

NeMo's Experiment Manager leverages PyTorch Lightning for model checkpointing, TensorBoard Logging, Weights and Biases, DLLogger and MLFlow logging. The
Experiment Manager is included by default in all NeMo example scripts.

To use the experiment manager simply call :class:`~nemo.utils.exp_manager.exp_manager` and pass in the PyTorch Lightning ``Trainer``.

.. code-block:: python

    exp_manager(trainer, cfg.get("exp_manager", None))

And is configurable via YAML with Hydra.

.. code-block:: bash

    exp_manager:
        exp_dir: /path/to/my/experiments
        name: my_experiment_name
        create_tensorboard_logger: True
        create_checkpoint_callback: True

Optionally, launch TensorBoard to view the training results in ``./nemo_experiments`` (by default).

.. code-block:: bash

    tensorboard --bind_all --logdir nemo_experiments

..

If ``create_checkpoint_callback`` is set to ``True``, then NeMo automatically creates checkpoints during training
using PyTorch Lightning's `ModelCheckpoint <https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint>`_.
We can configure the ``ModelCheckpoint`` via YAML or CLI.

.. code-block:: yaml

    exp_manager:
        ...
        # configure the PyTorch Lightning ModelCheckpoint using checkpoint_call_back_params
        # any ModelCheckpoint argument can be set here

        # save the best checkpoints based on this metric
        checkpoint_callback_params.monitor=val_loss

        # choose how many total checkpoints to save
        checkpoint_callback_params.save_top_k=5

We can auto-resume training as well by configuring the ``exp_manager``. Being able to auto-resume is important when doing long training
runs that are premptible or may be shut down before the training procedure has completed. To auto-resume training, set the following
via YAML or CLI:

.. code-block:: yaml

    exp_manager:
        ...
        # resume training if checkpoints already exist
        resume_if_exists: True

        # to start training with no existing checkpoints
        resume_ignore_no_checkpoint: True

        # by default experiments will be versioned by datetime
        # we can set our own version with
        exp_manager.version: my_experiment_version
