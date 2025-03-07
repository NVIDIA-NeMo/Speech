Implement FSDP2Strategy
=======================

Overview
========
The **FSDP2Strategy** implements Fully Sharded Data Parallel (FSDP) via PyTorch's FSDP2 implementation.
It enables distributed training with automatic model sharding and mixed precision support.

Features
========
- Automatic model parallelism
- Mixed precision training
- Checkpoint management
- Deferred optimizer state restoration
- Device mesh initialization

Initialize
==========
To initialize the **FSDP2Strategy**, use the following arguments:

.. code-block:: python

   strategy = FSDP2Strategy(
       data_parallel_size="auto",
       tensor_parallel_size="auto",
       checkpoint_io=None,
       mp_policy=None,
       parallelize_fn=None,
       **kwargs,
   )

Arguments:
----------
- **data_parallel_size** (*Union["auto", int]*): Number of data-parallel replicas.
- **tensor_parallel_size** (*Union["auto", int]*): Number of tensor-parallel groups.
- **checkpoint_io** (*optional*): Checkpoint I/O handler.
- **mp_policy** (*optional*): Mixed precision policy.
- **parallelize_fn** (*callable, optional*): Model parallelization function.

Parallelize
===========
The `parallelize()` method applies the sharding process to the model:

.. code-block:: python

   strategy.parallelize()

This method ensures that the model is only parallelized once.

Environment Setup
=================
The `setup_environment()` method initializes the distributed environment and device mesh:

.. code-block:: python

   strategy.setup_environment()

Manage Checkpoints
==================

Save Checkpoints
----------------
The `save_checkpoint()` method unshards the checkpoint and saves it to disk:

.. code-block:: python

   strategy.save_checkpoint(checkpoint, filepath)

Load Checkpoints
----------------
The `load_checkpoint()` method loads a checkpoint from disk:

.. code-block:: python

   checkpoint = strategy.load_checkpoint(filepath)

Restore Optimizer State
=======================
Optimizer state is deferred until the first training step. Use the following method to store the optimizer state:

.. code-block:: python

   strategy.load_optimizer_state_dict(checkpoint)

Train and Evaluate the Model
============================
Training Step
-------------
The `training_step()` method defines a single training iteration:

.. code-block:: python

   loss = strategy.training_step(batch, batch_idx)

Validation Step
---------------
The `validation_step()` method defines a validation iteration:

.. code-block:: python

   loss = strategy.validation_step(batch, batch_idx)

Test Step
---------
The `test_step()` method defines a test iteration:

.. code-block:: python

   loss = strategy.test_step(batch, batch_idx)

Prediction Step
---------------
The `predict_step()` method defines a prediction iteration:

.. code-block:: python

   result = strategy.predict_step(batch, batch_idx)

Process DataLoader
==================
Use `process_dataloader()` to apply custom data sampling to a DataLoader:

.. code-block:: python

   dataloader = strategy.process_dataloader(dataloader)

Retrieve State Dictionary
=========================
Retrieve the model's state dictionary using `lightning_module_state_dict()`:

.. code-block:: python

   state_dict = strategy.lightning_module_state_dict()

Remove Checkpoints
==================
Remove a checkpoint from the filesystem:

.. code-block:: python

   strategy.remove_checkpoint(filepath)

Initialize Tensors
==================
Use the `tensor_init_context()` context manager for tensor initialization:

.. code-block:: python

   with strategy.tensor_init_context():
       # Initialization code
       pass
