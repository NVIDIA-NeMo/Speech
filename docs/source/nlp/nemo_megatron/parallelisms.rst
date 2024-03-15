.. _parallelisms:

Parallelisms
------------

NeMo Megatron supports 5 types of parallelisms (which can be mixed together arbitraritly):

Distributed Data parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Distributed Data parallelism (DDP) creates idential copies of the model across multiple GPUs.

.. image:: images/ddp.gif
    :align: center
    :width: 800px
    :alt: Distributed Data Parallel


Tensor Parallelism
^^^^^^^^^^^^^^^^^^
With Tensor Paralellism (TP) a tensor is split into non-overlapping pieces and
different parts are distributed and processed on separate GPUs.

.. image:: images/tp.gif
    :align: center
    :width: 800px
    :alt: Tensor Parallel

Pipeline Parallelism
^^^^^^^^^^^^^^^^^^^^
With Pipeline Paralellism (PP) consecutive layer chunks are assigned to different GPUs.

.. image:: images/pp.gif
    :align: center
    :width: 800px
    :alt: Pipeline Parallel

Sequence Parallelism
^^^^^^^^^^^^^^^^^^^^

.. image:: images/sp.gif
    :align: center
    :width: 800px
    :alt: Sequence Parallel

Expert Parallelism
^^^^^^^^^^^^^^^^^^
Expert Paralellim (EP) distributes experts across GPUs.


.. image:: images/ep.png
    :align: center
    :width: 800px
    :alt: Expert Parallelism

Parallelism nomenclature
^^^^^^^^^^^^^^^^^^^^^^^^

When reading and modifying NeMo Megatron code you will encounter the following terms.

.. image:: images/pnom.gif
    :align: center
    :width: 800px
    :alt: Parallelism nomenclature
