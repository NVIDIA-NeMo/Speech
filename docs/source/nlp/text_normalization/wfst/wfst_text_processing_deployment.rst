.. _wfst_deployment:

Deploy to Production with C++ backend
=====================================

.. warning::

    TN/ITN transitioned from `NVIDIA/NeMo <https://github.com/NVIDIA/NeMo>`_ repository to a standalone `NVIDIA/NeMo-text-processing <https://github.com/NVIDIA/NeMo-text-processing>`_ repository. All updates and discussions/issues should go to the new repository.

NeMo-text-processing provides tools to deploy :doc:`TN and ITN <wfst_text_normalization>` for production :cite:`textprocessing-deployment-zhang2021nemo`.
It uses `Sparrowhawk <https://github.com/google/sparrowhawk>`_ :cite:`textprocessing-deployment-sparrowhawk` -- an open-source C++ framework by Google.
The grammars written with NeMo-text-processing can be exported into an `OpenFST <https://www.openfst.org/>`_ Archive File (FAR) and dropped into Sparrowhawk.

    .. image:: ./images/deployment_pipeline.png
        :align: center
        :alt: Deployment pipeline
        :scale: 50%


Requirements
------------

* :doc:`nemo_text_processing <wfst_text_normalization>` package
* `Docker <https://www.docker.com/>`_
* `NeMo-text-processing source code <https://github.com/NVIDIA/NeMo-text-processing>`_


.. _wfst_deployment_quick_start:

Quick Start
-----------

Examples how to run: 

.. code-block:: bash

    # export English TN grammars and return prompt inside docker container  
    cd NeMo-text-processing/tools/text_processing_deployment
    bash export_grammars.sh --GRAMMARS=tn_grammars --LANGUAGE=en --INPUT_CASE=cased

    # export English ITN grammars and return prompt inside docker container  
    cd NeMo-text-processing/tools/text_processing_deployment
    bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE=en


Arguments:
^^^^^^^^^^
* ``GRAMMARS`` - ``tn_grammars`` or ``itn_grammars`` to export either TN or ITN grammars.
* ``LANGUAGE`` - `en` for English. Click :doc:`here <wfst_text_normalization>` for full list of languages.
* ``INPUT_CASE`` - ``cased`` or ``lower_cased`` (ITN has no differentiation between these two, only used for TN).
* ``MODE`` - By default ``export`` which returns prompt inside the docker. If ``--MODE=test`` runs NeMo-text-processing pytests inside container.
* ``OVERWRITE_CACHE`` - Whether to re-export grammars or load from cache. By default ``True``. 
* ``FORCE_REBUILD`` - Whether to rebuild docker image in cased of updated dependencies. By default ``False``.

Detailed pipeline
-----------------

`export_grammars.sh` runs the following steps in sequence:

Go to script folder:

.. code-block:: bash

    cd NeMo-text-processing/tools/text_processing_deployment

1. Grammars written in Python are exported to `OpenFST <https://www.openfst.org/>`_ archive files (FAR). Specifically, grammars `ClassifyFst` and `VerbalizeFst` from :doc:`nemo_text_processing <wfst_text_normalization>` are exported and saved to `./LANGUAGE/classify/tokenize_and_classify.far` and `./LANGUAGE/verbalize/verbalize.far` respectively.

.. code-block:: bash

    python pynini_export.py <--output_dir .> <--grammars GRAMMARS> <--input_case INPUT_CASE> <--language LANGUAGE>

.. warning::

    TN and ITN grammars are saved to the same file by default.

2. Docker image is built with dependencies, including `Thrax <https://www.openfst.org/twiki/bin/view/GRM/Thrax>`_ and `Sparrowhawk <https://github.com/google/sparrowhawk>`_.

.. code-block:: bash

    bash docker/build.sh

3. Plugs in grammars into production backend by mounting grammar directory `LANGUAGE/classify/` and `LANGUAGE/verbalize/` inside docker. Returns docker prompt.

.. code-block:: bash

    # launch container with the exported grammars
    bash docker/launch.sh

4. Runs system in docker container.

.. code-block:: bash

    echo "ITN result: two dollars fifty. TN result: $2.50" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto

This returns "ITN result: $2.50. TN result: two dollars fifty cents"

See :doc:`WFST Resources <wfst_resources>` for more details.

References
----------

.. bibliography:: ../tn_itn_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-DEPLOYMENT
    :keyprefix: textprocessing-deployment-
