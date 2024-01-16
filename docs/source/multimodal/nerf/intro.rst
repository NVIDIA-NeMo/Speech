NeRF
====
NeMO NeRF is a collection of models and tools for training 3D and 4D models.

The library is designed with a modular approach, enabling developers to explore and find the most suitable solutions for their requirements,
and allowing researchers to accelerate their experimentation process.


Supported Models
-----------------
NeMo NeRF currently supports the following models:

+----------------------------------------+------------+
| Model                                  | Categories |
+========================================+============+
| `DreamFusion <./dreamfusion.html>`_    | text to 3D |
+----------------------------------------+------------+


Spotlight Models
-----------------

DreamFusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The `DreamFusion <https://dreamfusion3d.github.io/>`_ model utilizing pre-trained 2D text-to-image diffusion models to create detailed 3D objects from textual descriptions.
This approach overcomes the limitations of traditional 3D synthesis, which typically requires extensive labeled 3D data and sophisticated denoising architectures.
At the core of DreamFusion is the optimization of a Neural Radiance Field (NeRF), a parametric model for rendering 3D scenes.
The optimization process is driven by a loss function based on probability density distillation, which enables the 2D diffusion model to act as an effective prior.
DreamFusion is capable of producing 3D models that are not only accurate representations of the input text but also offer versatility in terms of rendering from any viewpoint,
relighting under diverse lighting conditions, and integration into various 3D environments. Importantly, this method achieves these results without the need for
specific 3D training data or modifications to the existing image diffusion model.

- Model Structure:
    - Text-to-image model: a pretrained text-to-image diffusion model is used to generate a 2D image from a given text.
    - NeRF: a neural radiance field (NeRF) that can generate novel views of complex 3D scenes, based on a partial set of 2D images.
    - Renderer: A volume rendering layer is used to render the NeRF model from a given viewpoint.


For more information, see additional sections in the NeRF docs on the left-hand-side menu or in the list below:

.. toctree::
   :maxdepth: 1

   datasets
   configs
   dreamfusion

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS
    :keyprefix: mm-models-
