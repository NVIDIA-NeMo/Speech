JitTransform Class
==================

.. currentmodule:: nemo.lightning.pytorch.callbacks

.. autoclass:: JitTransform
    :members:
    :undoc-members:
    :show-inheritance:

Description
-----------

The `JitTransform` class is a callback for PyTorch Lightning that applies Just-In-Time (JIT) compilation to PyTorch models. It leverages either `torch.compile` or the Thunder compiler to optimize model performance during training. This transformation is particularly useful for enhancing the efficiency of models with complex architectures or when deploying models in production environments.

Parameters
----------

.. py:attribute:: config

    **Type:** `JitConfig`

    The configuration object specifying how JIT compilation should be applied. It includes options to select the compiler (`torch.compile` or Thunder), module selectors to target specific parts of the model, and additional parameters for profiling and compiler behavior.

Usage Example
-------------

Below is an example of how to integrate `JitTransform` into a PyTorch Lightning `Trainer`:

.. code-block:: python

    from nemo.lightning.pytorch.callbacks import JitTransform
    from nemo.lightning.pytorch.callbacks import JitConfig
    from pytorch_lightning import Trainer

    # Define the JIT configuration
    jit_config = JitConfig(
        use_torch=True,
        torch_kwargs={'mode': 'default'},
        module_selector='*.layer*'
    )

    # Initialize the JitTransform callback
    jit_callback = JitTransform(config=jit_config)

    # Initialize the Trainer with the JitTransform callback
    trainer = Trainer(callbacks=[jit_callback])

    # Proceed with training as usual
    trainer.fit(model)

Attributes
----------

.. py:attribute:: config

    The `JitConfig` instance containing the JIT compilation settings.

Methods
-------

.. py:method:: on_train_epoch_start(trainer, pl_module)

    Called at the start of each training epoch. This method applies JIT compilation to the specified modules within the model based on the provided configuration.

    **Parameters:**

    - `trainer` (`pl.Trainer`): The PyTorch Lightning trainer instance.
    - `pl_module` (`pl.LightningModule`): The PyTorch Lightning module being trained.

Detailed Description
--------------------

The `JitTransform` callback applies JIT compilation at the beginning of each training epoch. It identifies the target modules within the model based on the `module_selector` patterns provided in the `JitConfig`. Depending on the configuration, it either uses `torch.compile` or Thunder to compile the selected modules, enhancing their execution performance.

The callback ensures that compilation occurs only once by setting a `_compiled` flag on the `pl_module`. This prevents redundant compilation in subsequent epochs. Additionally, it supports profiling through Thunder's profiler if enabled in the configuration.

Constraints
-----------

- **Mutually Exclusive Options:** The `use_torch` and `use_thunder` options in `JitConfig` are mutually exclusive. Enabling both simultaneously will raise an assertion error.
- **Module Selection:** The `module_selector` supports simple wildcard patterns. Complex selectors may require extending the matching logic.

See Also
--------

- :class:`JitConfig`
- `PyTorch Lightning Callbacks <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_
- `torch.compile <https://pytorch.org/docs/stable/generated/torch.compile.html>`_
