Adapter Components
==================

Adapters can be considered as any set of parameters that are added to a pre-existing module/model. In our case, we currently support the standard adapter in literature, more advanced adapter modules are being researched and can potentially be supported by NeMo.

An adapter module can be any pytorch module, but it must follow certain straightforward requirements -

1) The model accepts an input of some input dimension, and its output must match this dimension.
2) Ideally, the module is initialized such that the output of the adapter when initialized is such that it does not modify the original input. This allows the model to produce the same output results, even when additional parameters have been added.

According to Junxian et al :cite:`adapters-components-Junxian2021unified`, we can consider an adapter being represented as three components -

1) Functional form - the trainable parameters that will modify the input
2) Insertion form - Where the adapter outputs are integrated with the original input. The input to the adapters can be the last output of the layer, the input to some attention layer, or even the original input to the module itself (before even the modules forward pass).
3) Composition function - How the adapters outputs are integrated with the inputs. It can be as simple as residual addition connection, or concatenation, or point-wise multiplication etc.

Functional Form - Adapter Networks
==================================

Adapter modules represent the functional form of the adapter. We discuss an example of a most commonly used adapter module found in literature, titled the ``LinearAdapter`` (or Houlsby Adapter) :cite:`adapters-components-houlsby2019adapter`.

.. note::

    All adapter modules must extend :class:`~nemo.collections.common.parts.adapter_modules.AdapterModuleUtil` and should ideally have an equivalent DataClass config for easy instantiation !


.. autoclass:: nemo.collections.common.parts.adapter_modules.AdapterModuleUtil
    :show-inheritance:
    :members:
    :member-order: bysource
    :noindex:

-----

.. autoclass:: nemo.collections.common.parts.adapter_modules.LinearAdapter
    :show-inheritance:
    :members:
    :member-order: bysource
    :noindex:


Insertion Form - Module Adapters
--------------------------------

Adapter modules can be integrated into many different locations of a given module. For example, it is possible to have an adapter that affects only the outputs of the final layer in each module. We can also have a ``Parallel Adapter`` :cite:`adapters-components-Junxian2021unified` that operates at the input of the module itself, in parallel to the forward pass of the module. Yet another insertion location is inside the Multi Head Attention Layers.

On top of this, while adapters are commonly used only in the layers containing the most parameters (say the Encoder of a network), some models can support adapters in multiple locations (Encoder-Decoder architecture for Language Models, Machine Translation, or even Encoder-Decoder-Joint for ASR with Transducer Loss). As such, NeMo utilizes the concept of ``Module Adapters``.

``Module Adapters`` are very simply defined when adding an adapter - by specifying the module that the adapter should be inserted into.

.. code-block:: python

    # Get the list of supported modules / locations in a adapter compatible Model
    print(model.adapter_module_names)  # assume ['', 'encoder', 'decoder']

    # When calling add_adapter, specify the module name in the left of the colon symbol, and the adapter name afterwords.
    # The adapter is then directed to the decoder module instead of the default / encoder module.
    model.add_adapter("decoder:first_adapter", cfg=...)

You might note that ``model.adapter_module_names`` can sometimes return ``''`` as one of the supported module names - this refers to the "default module". Generally we try to provide the default as the most commonly used adapter in literature - for example, Encoder adapters in NLP/NMT/ASR.

Composition Function - Adapter Strategies
-----------------------------------------

Finally, we discuss how to compose the input and output of adapter modules. In order to generalize this step, we construct ``Adapter Strategies``.
A strategy is any class (not torch.nn.Module!) that extends :class:`~nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy`, and provides a ``forward()`` method that accepts a specific signature of the inputs and produces an output tensor which combines the input and output with some specific method.

We discuss a simple residual additional connection strategy below - that accepts an input to the adapter and an adapters output and simply adds them together. It also supports ``stochastic_depth`` which enables adapters to be dynamically switched off during training, making training more robust.

.. autoclass:: nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy
    :show-inheritance:
    :members:
    :member-order: bysource
    :undoc-members: adapter_module_names
    :noindex:

-----

.. autoclass:: nemo.core.classes.mixins.adapter_mixin_strategies.ResidualAddAdapterStrategy
    :show-inheritance:
    :members:
    :member-order: bysource
    :undoc-members: adapter_module_names
    :noindex:

-----


References
----------

.. bibliography:: ./adapter_bib.bib
    :style: plain
    :keyprefix: adapters-components-
