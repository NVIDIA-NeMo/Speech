Vision-Language Foundation
==========================

Humans naturally process information using multiple senses like sight and sound. Similarly, multi-modal learning aims to create models that handle different types of data, such as images, text, and audio. There's a growing trend in models that combine vision and language, like OpenAI's CLIP. These models excel in tasks like aligning image and text features, image captioning and visual question-answering. Their ability to generalize without specific training offers many practical uses.

Supported Models
-----------------
NeMo Multimodal currently supports the following models:

+-----------------------------------+----------+-------------+------+-------------------------+------------------+
| Model                             | Training | Fine-Tuning | PEFT | Evaluation              | Inference        |
+===================================+==========+=============+======+=========================+==================+
| `CLIP <./clip.html>`_             | ✓        | -           | -    | zero-shot imagenet      | similarity score |
+-----------------------------------+----------+-------------+------+-------------------------+------------------+

Spotlight Models
-----------------

Vision-Language models are at the forefront of multimodal learning, showcasing impressive abilities in tasks that require a combination of visual and textual comprehension. Let's take a quick look at some key models driving progress in this field:

#. **Contrastive Learning Based Models**: At the forefront is CLIP :cite:`mm-models-radford2021clip`, which harnesses contrastive learning to jointly fine-tune a text and image encoder, facilitating a gamut of downstream tasks. CLIP's success has spurred further research, leading to models like ALIGN :cite:`mm-models-saharia2022photorealistic` and DeCLIP :cite:`mm-models-li2021declip`.

#. **Holistic Foundation Models**: FLAVA :cite:`mm-models-singh2022flava` aspires to craft a universal model adept at vision, language, and multimodal tasks. Through a unified architecture, it vies to excel across a spectrum of tasks, embodying the essence of a true foundation model.

#. **Bootstrapping Techniques**: BLIP :cite:`mm-models-blip2` employs a pioneering framework that shines in both understanding-based and generation-based vision-language tasks. By bootstrapping captions from noisy web data, it exhibits remarkable generalization across a plethora of vision-language challenges.

Anatomy of Vision-Language Models
----------------------------------

At their core, vision-language models fundamentally consist of three main parts:

1. **Image Encoder:** Extracts features from images.
2. **Text Encoder:** Extracts features from textual data.
3. **Fusion Strategy:** Merges the information gleaned from both encoders.

These models have undergone a significant transformation. Earlier models used manually designed image descriptors and pre-trained word vectors. Nowadays, models primarily utilize transformer architectures for both image and text encoding, learning features together or separately. The pre-training objectives of these models are carefully designed to suit a wide range of tasks.

Contrastive Learning: Bridging Vision and Language
---------------------------------------------------

Contrastive learning has burgeoned as a pivotal pre-training objective, especially for vision-language models. Models like CLIP, CLOOB, ALIGN, and DeCLIP have harnessed contrastive learning to bridge the chasm between vision and language. They accomplish this by jointly learning a text encoder and an image encoder using a contrastive loss, typically on extensive datasets encompassing {image, caption} pairs.

The quintessence of contrastive learning is to map images and texts to a shared feature realm. Here, the distance between the embeddings of congruent image-text pairs is minimized, while it's maximized for incongruent pairs. For instance, CLIP employs the cosine distance between text and image embeddings, while models like ALIGN and DeCLIP have crafted their own distance metrics to cater to the intricacies of their datasets.

CLIP and Beyond
---------------

The CLIP (Contrastive Language-Image Pre-training) model  has notably served as a linchpin for various models and applications within the realms of deep learning and computer vision, and also within the NeMo toolkit. Below is an elucidation on how the CLIP model extends its influence into other models and domains:

1. **Use Cases in Vision Tasks:**
   * **Classification:** CLIP can be harnessed for classification tasks, accepting arbitrary text labels for zero-shot classification on video frames or images.
   * **Semantic Image Search:** Constructing a semantic image search engine with CLIP showcases its capability to generate embeddings for semantic content analysis and similarity search.

2. **Image Similarity and Clustering:**
   * In a practical scenario, CLIP's embeddings were leveraged for an image similarity search engine, showcasing its effectiveness in generating useful representations for visual similarity scenarios, even without being specifically trained for such tasks.

3. **Foundation for Multimodal Language Models:**
   * Large language models with visual capabilities, such as LLaVA, Flamingo, Kosmos-1, and Kosmos-2, have leaned on CLIP's architecture. In these models, images are encoded using a visual encoder derived from CLIP.

4. **Foundation Diffusion Models:**
   * Models like Stable Diffusion and Imagen have tapped into the prowess of the text encoder from CLIP to condition their processes based on text prompts. This integration exemplifies the adaptability and influence of the CLIP encoder in the broader AI landscape, especially in the domain of diffusion models.

.. note::
    NeMo Megatron has an Enterprise edition which proffers tools for data preprocessing, hyperparameter tuning, containers, scripts for various clouds, and more. With the Enterprise edition, you also garner deployment tools. Apply for `early access here <https://developer.nvidia.com/nemo-megatron-early-access>`_ .


.. toctree::
   :maxdepth: 1

   datasets
   configs
   checkpoint
   clip

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS
    :keyprefix: mm-models-