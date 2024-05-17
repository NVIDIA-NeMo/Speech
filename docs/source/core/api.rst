
NeMo Core APIs
==============

Base class for all NeMo models
------------------------------

.. autoclass:: nemo.core.ModelPT
    :show-inheritance:
    :members:
    :member-order: bysource
    :undoc-members: cfg, num_weights
    :exclude-members: set_eff_save, use_eff_save, teardown

Base Neural Module class
------------------------

.. autoclass:: nemo.core.NeuralModule
    :show-inheritance:
    :members:
    :member-order: bysource

Base Mixin classes
------------------

.. autoclass:: nemo.core.Typing
    :show-inheritance:
    :members:
    :member-order: bysource
    :private-members:
    :exclude-members: _abc_impl
    :noindex:

-----

.. autoclass:: nemo.core.Serialization
    :show-inheritance:
    :members:
    :member-order: bysource
    :noindex:

-----

.. autoclass:: nemo.core.FileIO
    :show-inheritance:
    :members:
    :member-order: bysource
    :noindex:


Base Connector classes
----------------------

.. autoclass:: nemo.core.connectors.save_restore_connector.SaveRestoreConnector
    :show-inheritance:
    :members:
    :member-order: bysource


Base Mixin Classes
------------------

.. autoclass:: nemo.core.classes.mixins.access_mixins.AccessMixin
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO
    :show-inheritance:
    :members:
    :member-order: bysource


Neural Type checking
--------------------

.. autoclass:: nemo.core.classes.common.typecheck
    :show-inheritance:
    :members:
    :member-order: bysource

    .. automethod:: __call__

Neural Type classes
-------------------

.. autoclass:: nemo.core.neural_types.NeuralType
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: nemo.core.neural_types.axes.AxisType
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: nemo.core.neural_types.elements.ElementType
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: nemo.core.neural_types.comparison.NeuralTypeComparisonResult
    :show-inheritance:
    :members:
    :member-order: bysource

Experiment manager
------------------

.. autoclass:: nemo.utils.exp_manager.exp_manager
    :show-inheritance:
    :members:
    :member-order: bysource

.. autoclass:: nemo.utils.exp_manager.ExpManagerConfig
    :show-inheritance:
    :members:
    :member-order: bysource


Exportable
----------

.. autoclass:: nemo.core.classes.exportable.Exportable
    :show-inheritance:
    :members:
    :member-order: bysource

