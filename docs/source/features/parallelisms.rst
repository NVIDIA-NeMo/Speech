.. _parallelisms:

Parallelisms
------------

NeMo Megatron supports 5 types of parallelisms (which can be mixed together arbitrarily):

Distributed Data Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Distributed Data Parallelism (DDP) creates idential copies of the model across multiple GPUs.

.. image:: ../nlp/nemo_megatron/images/ddp.gif
    :align: center
    :width: 800px
    :alt: Distributed Data Parallel


Tensor Parallelism
^^^^^^^^^^^^^^^^^^
With Tensor Paralellism (TP) a tensor is split into non-overlapping pieces and
different parts are distributed and processed on separate GPUs.

.. image:: ../nlp/nemo_megatron/images/tp.gif
    :align: center
    :width: 800px
    :alt: Tensor Parallel

Pipeline Parallelism
^^^^^^^^^^^^^^^^^^^^
With Pipeline Paralellism (PP) consecutive layer chunks are assigned to different GPUs.

.. image:: ../nlp/nemo_megatron/images/pp.gif
    :align: center
    :width: 800px
    :alt: Pipeline Parallel

Sequence Parallelism
^^^^^^^^^^^^^^^^^^^^

.. image:: ../nlp/nemo_megatron/images/sp.gif
    :align: center
    :width: 800px
    :alt: Sequence Parallel

Expert Parallelism
^^^^^^^^^^^^^^^^^^
Expert Paralellim (EP) distributes experts across GPUs.


.. image:: ../nlp/nemo_megatron/images/ep.png
    :align: center
    :width: 800px
    :alt: Expert Parallelism

Parallelism nomenclature
^^^^^^^^^^^^^^^^^^^^^^^^

When reading and modifying NeMo Megatron code you will encounter the following terms.

.. image:: ../nlp/nemo_megatron/images/pnom.gif
    :align: center
    :width: 800px
    :alt: Parallelism nomenclature
