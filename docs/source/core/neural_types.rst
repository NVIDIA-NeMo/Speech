
Neural Types
============

Motivation
----------

Neural Types describe the semantics, axis order, and dimensions of a tensor. The purpose of this type system is to catch semantic and
dimensionality errors during model creation and facilitate module re-use.

.. image:: whyntypes.gif
  :width: 900
  :alt: Neural Types Motivation

``NeuralType`` class
--------------------

Neural Types perform semantic checks for modules and models inputs/outputs. They contain information about:

    - Semantics of what is stored in the tensors. For example, logits, logprobs, audiosignal, embeddings, etc.
    - Axes layout, semantic and (optionally) dimensionality. For example: ``[Batch, Time, Channel]``

Types are implemented in ``nemo.core.neural_types.NeuralType`` class. When you instantiate an instance of this class, you
are expected to include both *axes* information and *element type* information.

.. autoclass:: nemo.core.neural_types.NeuralType
    :noindex:

Type Comparison Results
-----------------------

When comparing two neural types, the following comparison results are generated.

.. autoclass:: nemo.core.neural_types.NeuralTypeComparisonResult
    :noindex:

Examples
--------

Long vs short notation
~~~~~~~~~~~~~~~~~~~~~~

NeMo's ``NeuralType`` class allows you to express axis semantics information in long and short form. Consider these two equivalent types. Both encoder 3 dimensional tensors and both contain elements of type ``AcousticEncodedRepresentation`` (this type is a typical output of ASR encoders).

.. code-block:: python

    long_version = NeuralType(
            axes=(AxisType(AxisKind.Batch, None), AxisType(AxisKind.Dimension, None), AxisType(AxisKind.Time, None)),
            elements_type=AcousticEncodedRepresentation(),
        )
    short_version = NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())
    assert long_version.compare(short_version) == NeuralTypeComparisonResult.SAME

Transpose same
~~~~~~~~~~~~~~

Often it is useful to know if a simple transposition will solve type incompatibility. This is the case if the comparison result of two types equals ``nemo.core.neural_types.NeuralTypeComparisonResult.TRANSPOSE_SAME``.

.. code-block:: python

    type1 = NeuralType(axes=('B', 'T', 'C'))
    type2 = NeuralType(axes=('T', 'B', 'C'))
    assert type1.compare(type2) == NeuralTypeComparisonResult.TRANSPOSE_SAME
    assert type2.compare(type1) == NeuralTypeComparisonResult.TRANSPOSE_SAME

Note that in this example, we dropped ``elements_type`` argument of ``NeuralType`` constructor. If not supplied, the element type is ``VoidType``.

``VoidType`` for elements
~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes it is useful to express that elements' types don't matter but axes layout do. ``VoidType`` for elements can be used to express this.

.. note:: ``VoidType`` is compatible with every other elements' type but not the other way around. See the following code snippet below for details.

.. code-block:: python

        btc_spctr = NeuralType(('B', 'T', 'C'), SpectrogramType())
        btc_spct_bad = NeuralType(('B', 'T'), SpectrogramType())
        # Note the VoidType for elements here
        btc_void = NeuralType(('B', 'T', 'C'), VoidType())

        # This is true because VoidType is compatible with every other element type (SpectrogramType in this case)
        # And axes layout between btc_void and btc_spctr is the same
        assert btc_void.compare(btc_spctr) == NeuralTypeComparisonResult.SAME
        # These two types are incompatible because even though VoidType is used for elements on one side,
        # the axes layout is different
        assert btc_void.compare(btc_spct_bad) == NeuralTypeComparisonResult.INCOMPATIBLE
        # Note that even though VoidType is compatible with every other type, other types are not compatible with VoidType!
        # It is one-way compatibility
        assert btc_spctr.compare(btc_void) == NeuralTypeComparisonResult.INCOMPATIBLE

Element type inheritance
~~~~~~~~~~~~~~~~~~~~~~~~

Neural types in NeMo support Python inheritance between element types. Consider an example where you want to develop a Neural Module which performs data augmentation for all kinds of spectrograms.
In ASR, two types of spectrograms are frequently used: mel and mfcc. To express this, we will create 3 classes to express
element's types: ``SpectrogramType``, ``MelSpectrogramType(SpectrogramType)``, ``MFCCSpectrogramType(SpectrogramType)``.

.. code-block:: python

        input = NeuralType(('B', 'D', 'T'), SpectrogramType())
        out1 = NeuralType(('B', 'D', 'T'), MelSpectrogramType())
        out2 = NeuralType(('B', 'D', 'T'), MFCCSpectrogramType())

        # MelSpectrogram and MFCCSpectrogram are not interchangeable.
        assert out1.compare(out2) == NeuralTypeComparisonResult.INCOMPATIBLE
        assert out2.compare(out1) == NeuralTypeComparisonResult.INCOMPATIBLE
        # Type comparison detects that MFCC/MelSpectrogramType is a kind of SpectrogramType and can be accepted.
        assert input.compare(out1) == NeuralTypeComparisonResult.GREATER
        assert input.compare(out2) == NeuralTypeComparisonResult.GREATER

Custom element types
~~~~~~~~~~~~~~~~~~~~

It is possible to create user-defined element types to express the semantics of elements in your tensors. To do so, the user will need to inherit and implement abstract methods of the ``nemo.core.neural_types.elements.ElementType`` class

.. autoclass:: nemo.core.neural_types.elements.ElementType
    :noindex:

Note that element types can be parametrized. Consider this example where it distinguishes between audio sampled at 8Khz and 16Khz.

.. code-block:: python

    audio16K = NeuralType(axes=('B', 'T'), elements_type=AudioSignal(16000))
    audio8K = NeuralType(axes=('B', 'T'), elements_type=AudioSignal(8000))

    assert audio8K.compare(audio16K) == NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
    assert audio16K.compare(audio8K) == NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS

Enforcing dimensions
~~~~~~~~~~~~~~~~~~~~

In addition to specifying tensor layout and elements' semantics, neural types also allow you to enforce tensor dimensions.
The user will have to use long notations to specify dimensions. Short notations only allows you to specify axes semantics and assumes
arbitrary dimensions.

.. code-block:: python

        type1 = NeuralType(
        (AxisType(AxisKind.Batch, 64), AxisType(AxisKind.Time, 10), AxisType(AxisKind.Dimension, 128)),
        SpectrogramType(),
        )
        type2 = NeuralType(('B', 'T', 'C'), SpectrogramType())

        # type2 will accept elements of type1 because their axes semantics match and type2 does not care about dimensions
        assert type2.compare(type1), NeuralTypeComparisonResult.SAME
        # type1 will not accept elements of type2 because it need dimensions to match strictly.
        assert type1.compare(type2), NeuralTypeComparisonResult.DIM_INCOMPATIBLE

Generic Axis kind
~~~~~~~~~~~~~~~~~

Sometimes (especially in the case of loss modules) it is useful to be able to specify a "generic" axis kind which will make it
compatible with any other kind of axis. This is easy to express with Neural Types by using ``nemo.core.neural_types.axes.AxisKind.Any`` for axes.

.. code-block:: python

        type1 = NeuralType(('B', 'Any', 'Any'), SpectrogramType())
        type2 = NeuralType(('B', 'T', 'C'), SpectrogramType())
        type3 = NeuralType(('B', 'C', 'T'), SpectrogramType())

        # type1 will accept elements of type2 and type3 because it only cares about element kind (SpectrogramType)
        # number of axes (3) and that first one corresponds to batch
        assert type1.compare(type2) == NeuralTypeComparisonResult.SAME
        assert type1.compare(type3) == NeuralTypeComparisonResult.INCOMPATIBLE

Container types
~~~~~~~~~~~~~~~

The NeMo-type system understands Python containers (lists). If your module returns a nested list of typed tensors, the way to express it is by
using Python list notation and Neural Types together when defining your input/output types.

The example below shows how to express that your module returns single output ("out") which is list of lists of two dimensional tensors of shape ``[batch, dimension]`` containing logits.

.. code-block:: python

    @property
    def output_types(self):
        return {
            "out": [[NeuralType(('B', 'D'), LogitsType())]],
        }
