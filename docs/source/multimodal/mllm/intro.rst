Multimodal Language Models
==========================

The endeavor to extend Language Models (LLMs) into multimodal domains by integrating additional structures like visual encoders has become a focal point of recent research, especially given its potential to significantly lower the cost compared to training multimodal universal models from scratch.

The advent of GPT-4 has spurred a plethora of developments including notable models like LLaVA, Mini-GPT4, and Flamingo. These models, despite minor differences, share similar structural and training strategies.

Supported Models
-----------------
NeMo Multimodal currently supports the following models:

+-----------------------------------+----------+-------------+------+-------------------------+------------------+
| Model                             | Training | Fine-Tuning | PEFT | Evaluation              | Inference        |
+===================================+==========+=============+======+=========================+==================+
| `NeVA (LLaVA) <./neva.html>`_     | ✓        | ✓           | -    | -                       | ✓                |
+-----------------------------------+----------+-------------+------+-------------------------+------------------+
| Kosmos-2                          | WIP      | WIP         | -    | -                       | WIP              |
+-----------------------------------+----------+-------------+------+-------------------------+------------------+

Spotlight Models
-----------------

LLaVA: Visual Instruction Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LLaVA :cite:`mm-models-llava` focuses on creating a dataset for visual instruction tuning to enhance LLMs' ability to comprehend diverse instructions and provide detailed responses. NeMo's implementation of LLaVA is called NeVA.

- Model Structure:
    - Visual Encoder: Utilizes CLIP’s ViT-L/14.
    - Text Decoder: Employs LLaMA.
    - Connection: A simple linear mapping layer connects the visual encoder's output to the text decoder's word embedding space (v1.0 version).

- Training:
    1. Cross-modal Pre-training: Utilizes 595k image-text data from CC3M, training only the linear mapping layer while keeping the visual encoder and text decoder frozen.
    2. Instruction Fine-tuning: Custom-built 158k multimodal instruction dataset employed for fine-tuning targeting multimodal chatbot scenarios, with a variant targeting the Science QA dataset.

Flamingo: A Visual Language Model for Few-Shot Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Flamingo :cite:`mm-models-flamingo` addresses inconsistent visual feature map sizes by generating fixed-length feature sequences, enhancing visual relevance generation.

- Model Structure:
    - Resampler: Utilizes a Perceiver Resampler for generating fixed-length feature sequences.
    - Attention: Adds cross-attention layers before each LLM layer to enhance visual relevance generation.

- Training:
    - Dataset: Utilizes data from various datasets like M3W, ALIGN, LTIP, and VTP emphasizing multimodal in-context learning.

Kosmos-1: Language Is Not All You Need: Aligning Perception with Language Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Kosmos-1 :cite:`mm-models-kosmos1` by Microsoft is a Multimodal Large Language Model (MLLM) aimed at melding language, perception, action, and world modeling.

- Model Structure:
    - Core Backbone: Transformer-Based Causal Language Model.
    - Architecture: Utilizes MAGNETO, a nuanced Transformer variant.
    - Position Encoding: Employs XPOS relative position encoding for long-context modeling.
    - Resampler: Employs Flamingo's Perceiver Resampler

- Training:
    - Dataset: Encompasses web-scale multimodal corpora including monomodal, cross-modal paired, and interleaved multimodal data.
    - Objective: Focused on next-token prediction to maximize log-likelihood of tokens within examples.

BLIP-2: Bootstrapping Language-Image Pre-training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BLIP-2 :cite:`mm-models-blip2` adopts a two-phase training strategy focusing on learning key visual information and adapting visual encoding structure to LLMs.

- Model Structure:
    - Visual Encoder: Combines a pre-trained image encoder with a Querying Transformer (Q-Former).
    - Bridging: The Q-Former acts as the bridge between the image encoder and the Large Language Model (LLM).

- Training:
    1. Phase 1: Focuses on tasks like Image-Text Contrastive Learning, Image-grounded Text Generation, and Image-Text Matching.
    2. Phase 2: Aims at adapting the visual encoding structure's output to LLMs with language modeling as the training task.

Mini-GPT4: Enhancing Vision-Language Understanding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mini-GPT4 :cite:`mm-models-minigpt4` emphasizes the importance of multimodal instruction data for model performance in multimodal open-ended scenarios.

- Model Structure:
    - Visual Encoder: Employs BLIP2’s ViT and Q-Former.
    - Text Decoder: Uses Vicuna (a fine-tuned version of LLaMA).
    - Connection: A linear mapping layer projects visual features into text representation space.

- Training:
    1. Cross-modal Learning: Focuses on learning the relationship between vision and language using data from CC+SBU+LAION datasets.
    2. Fine-tuning: Utilizes a multimodal fine-tuning dataset built using ChatGPT to enhance text descriptions generated in phase 1.

.. note::
    NeMo Megatron has an Enterprise edition which proffers tools for data preprocessing, hyperparameter tuning, containers, scripts for various clouds, and more. With the Enterprise edition, you also garner deployment tools. Apply for `early access here <https://developer.nvidia.com/nemo-megatron-early-access>`_ .

For more information, see additional sections in the NeMo multimodal language model docs on the left-hand-side menu or in the list below:

.. toctree::
   :maxdepth: 1

   datasets
   configs
   checkpoint
   neva

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS
    :keyprefix: mm-models-