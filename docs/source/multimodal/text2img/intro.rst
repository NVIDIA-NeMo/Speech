Text to Image Models
====================


Supported Models
-----------------
NeMo Multimodal currently supports the following models:

+----------------------------------------+------------+
| Model                                  | Categories |
+========================================+============+
| `Stable Diffusion <./sd.html>`_        | Foundation |
+----------------------------------------+------------+
| `Imagen <./imagen.html>`_              | Foundation |
+----------------------------------------+------------+
| `DreamBooth <./dreambooth.html>`_      | Finetune   |
+----------------------------------------+------------+
| `ControlNet <./controlnet.html>`_      | Finetune   |
+----------------------------------------+------------+
| `instructPix2Pix <./insp2p.html>`_     | Finetune   |
+----------------------------------------+------------+


Text2Img Foundation Models
--------------------------
Text-to-image models are a fascinating category of artificial intelligence models that aim to generate realistic images from textual descriptions. The mainstream text-2-image models can be broadly grouped into:

#. **Diffusion Based Models**: these models leverage diffusion processes to
   generate images from text and may operate in the latent space (Stable Diffusion :cite:`mm-models-rombach2022highresolution`) or directly in the pixel space (Imagen :cite:`mm-models-saharia2022photorealistic`). These models typically use probabilistic models to model the generation process.
   They consider the sequential diffusion of information, which helps them generate images in a more coherent and controlled manner.
   This approach is known for producing high-quality and diverse images while incorporating textual descriptions.

#. **Autoregressive Based Models**: like Parti :cite:`mm-models-yu2022scaling`
   and Make-A-Scene :cite:`mm-models-gafni2022makeascene`, generate images one pixel or region at a time.
   These models take in the text description and gradually build the image pixel by pixel or element by element in
   an autoregressive manner. While this approach can produce detailed images, it can be computationally expensive
   and may not scale well for high-resolution images.


#. **Masked Token Prediction Models**: including MUSE :cite:`mm-models-chang2023muse`, employ masked token prediction-based architectures.
   These models learn to map text and image inputs into a shared embedding space.
   They use a masked token prediction task during pretraining, allowing them to understand the
   relationships between text and images. Given a text prompt, they can retrieve or generate images
   that align with the content and context of the text description.


Each of these approaches has its strengths and weaknesses, making them suitable for different use cases and scenarios.
Diffusion-based models excel in generating diverse and high-quality images, autoregressive models offer fine-grained control,
and masked token prediction-based models are strong at understanding and aligning text and images.
The choice of model depends on the specific requirements of the text-to-image generation task at hand.


Approaches to Customize/Extend Text2Img Models
----------------------------------------------

Customizing and extending Text2Img models can be essential to tailor these foundation models to
specific applications or creative tasks. Some popular approaches to customize and extend text2img models include:


#. **Text-Based Image Editing**: such as instructPix2Pix :cite:`mm-models-insp2p`, involves manipulating or modifying generated images based on
   textual descriptions. To customize text2img models for this purpose, one can employ post-processing techniques to
   alter the generated images.

#. **Injecting New Concepts**: including DreamBooth :cite:`mm-models-ruiz2023dreambooth`, can introduce new concepts into text2img models. This is typically done by
   adapting foundation models with additional data for finetuning.

#. **Adding Conditionings to Guide Image Generation**: like ControlNet :cite:`mm-models-zhang2023adding`, allows for greater control and specificity in the generated images.
   These conditionings can be based on various factors including specific attributes mentioned in the text (such as colors, sizes, or object properties),
   spatial information, style and mood.

Customizing and extending Text2Img models based on these approaches empowers users to have more control over the generated content,
make images more contextually relevant, and adapt the models to a wide array of creative and practical tasks,
from art creation to content personalization.

.. note::
    NeMo Megatron has an Enterprise edition which proffers tools for data preprocessing, hyperparameter tuning, containers, scripts for various clouds, and more. With the Enterprise edition, you also garner deployment tools. Apply for `early access here <https://developer.nvidia.com/nemo-megatron-early-access>`_ .


For more information, see additional sections in the MM Text2Img docs on the left-hand-side menu or in the list below:

.. toctree::
   :maxdepth: 1

   datasets
   configs
   checkpoint
   sd
   imagen
   dreambooth
   controlnet

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS
    :keyprefix: mm-models-