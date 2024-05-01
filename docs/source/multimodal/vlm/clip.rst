CLIP
====

Model Introduction
-------------------

Contrastive Language-Image Pre-training (CLIP) :cite:`mm-models-clip-radford2021learning` offers an efficient method for learning image representations using natural language supervision. The essence of CLIP is to train both an image encoder and a text encoder from scratch. The model aims to predict the correct pairings of a batch of (image, text) training examples by jointly training these encoders. During pre-training, CLIP is designed to predict which images and texts form a semantically coherent pair by maximizing the similarity between the correct (image, text) pairs while minimizing the similarity between incorrect pairs. This contrastive learning approach ensures that CLIP learns meaningful and contextually rich representations of both visual and textual data.

NeMo's implementation of the CLIP model leverages its parallel transformer implementation, specifically the `nemo.collections.nlp.modules.common.megatron.transformer.ParallelTransformer`, to enable model parallelism support in both the text encoder and vision model. This design choice ensures efficient scaling and utilization of resources during training. Additionally, some of the model design and loss implementations in NeMo's CLIP are inspired by the open-source [open_clip](https://github.com/mlfoundations/open_clip) repository.

    .. image:: images/clip_arch.png
        :align: center
        :alt: CLIP model
        :scale: 30%

CLIP models in NeMo can be instantiated using the :class:`~nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models.MegatronCLIPModel` class.

Text Encoder
^^^^^^^^^^^^^^^

CLIP uses a transformer-based text encoder to encode text features. The text input is tokenized and embedded. Positional embeddings are added to these token embeddings, and this combined representation is then passed through several transformer layers. The output from the last transformer layer corresponding to the first token is used as the text representation. In NeMo, the CLIP text encoder can be instantiated using the :class:`~nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models.CLIPTextTransformer` class.

Vision Model
^^^^^^^^^^^^

CLIP's vision model is based on the Vision Transformer (ViT) architecture. The image is first divided into fixed-size patches (e.g., 16x16 pixels). These patches are linearly embedded into a flat vector, which is then used as input to the transformer. The output of the transformer is then pooled to produce a single image representation. In NeMo, the CLIP vision model can be instantiated using the :class:`~nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models.CLIPVisionTransformer` class.


+-------+------------+----------------------+------------------------+-----------------+------------------+-----------+----------------+-------------+-----------------+------------------+------------+
| Model | Image size | Image Model size (M) | Hidden size (FFN size) | Attention heads | Number of layers | Patch dim | Model size (M) | Hidden size | Attention heads | Number of layers | Output dim |
|       |            | (Vision)             | (Vision)               | (Vision)        | (Vision)         | (Vision)  | (Text)         | (Text)      | (Text)          | (Text)           |            |
+=======+============+======================+========================+=================+==================+===========+================+=============+=================+==================+============+
| B/32  | 224        | 87.85                | 768                    | 12              | 12               | 16        | 63.43          | 512         | 8               | 12               | 512        |
+-------+------------+----------------------+------------------------+-----------------+------------------+-----------+----------------+-------------+-----------------+------------------+------------+
| B/16  | 224        | 86.19                | 768                    | 12              | 12               | 32        | 91.16          | 512         | 8               | 12               | 512        |
+-------+------------+----------------------+------------------------+-----------------+------------------+-----------+----------------+-------------+-----------------+------------------+------------+
| L/14  | 224        | 303.97               | 1024                   | 16              | 24               | 14        | 123.65         | 768         | 12              | 12               | 768        |
+-------+------------+----------------------+------------------------+-----------------+------------------+-----------+----------------+-------------+-----------------+------------------+------------+
| H/14  | 224        | 638.08               | 1280                   | 20              | 32               | 14        | 354.03         | 1024        | 16              | 24               | 1024       |
+-------+------------+----------------------+------------------------+-----------------+------------------+-----------+----------------+-------------+-----------------+------------------+------------+
| g/14  | 224        | 1012.65              | 1408 (6144)            | 22              | 40               | 14        | 354.03         | 1024        | 16              | 24               | 1024       |
+-------+------------+----------------------+------------------------+-----------------+------------------+-----------+----------------+-------------+-----------------+------------------+------------+
| G/14  | 224        | 1840                 | 1664 (8192)            | 16              | 48               | 14        | 590            | 1280        | 20              | 32               | 1280       |
+-------+------------+----------------------+------------------------+-----------------+------------------+-----------+----------------+-------------+-----------------+------------------+------------+
| e/14  | 224        | 2200                 | 1792 (15360)           | 28              | 56               | 14        | 660            | 1280        | 20              | 36               | 1280       |
+-------+------------+----------------------+------------------------+-----------------+------------------+-----------+----------------+-------------+-----------------+------------------+------------+



Model Configuration
-------------------

General Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  model:
    output_dim: 512
    local_loss: False
    gather_with_grad: True

- ``output_dim``: Represents the dimensionality of the output embeddings for both the text and vision models.
- ``local_loss``: If set to `True`, the loss is calculated with local features at a global level, avoiding the need to realize the full global matrix. This can be beneficial for memory efficiency, especially when training on multiple devices.
- ``gather_with_grad``: Enables full distributed gradient for feature gathering. Disabling this (setting to `False`) may cause convergence issues.

Vision Model Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  vision:
    patch_dim: 16
    img_h: 224
    img_w: 224
    image_mean: null
    image_std: null
    num_channels: 3
    drop_patch_rate: 0.0
    drop_path_rate: 0.0
    global_average_pool: False

    output_dim: ${model.output_dim}
    class_token_length: 8
    encoder_seq_length: 196
    num_layers: 12
    hidden_size: 768
    ffn_hidden_size: 3072
    num_attention_heads: 12
    hidden_dropout: 0.
    attention_dropout: 0.

- ``patch_dim``: Size of the patches the image is divided into.
- ``img_h`` and ``img_w``: Height and width of the input images.
- ``image_mean`` and ``image_std``: Mean and standard deviation values for image normalization.
- ``num_channels``: Number of channels in the input image (e.g., 3 for RGB images).
- ``drop_patch_rate`` and ``drop_path_rate``: Dropout rates for patches and paths respectively.
- ``global_average_pool``: If set to `True`, applies global average pooling to the output.
- ``class_token_length``: Length of the extra classification tokens.
- ``encoder_seq_length``: Sequence length for the vision encoder.
- ``num_layers``, ``hidden_size``, ``ffn_hidden_size``, ``num_attention_heads``: Parameters defining the architecture of the vision transformer. The ``ffn_hidden_size`` is typically 4 times the ``hidden_size``.
- ``hidden_dropout`` and ``attention_dropout``: Dropout probabilities for the hidden state and attention in the transformer respectively.

Text Model Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  text:
    output_dim: ${model.output_dim}
    encoder_seq_length: 77
    num_layers: 12
    hidden_size: 512
    ffn_hidden_size: 2048
    num_attention_heads: 8
    hidden_dropout: 0.
    attention_dropout: 0.

- ``output_dim``: Dimensionality of the output embeddings for the text model.
- ``encoder_seq_length``: Sequence length for the text encoder.
- ``num_layers``, ``hidden_size``, ``ffn_hidden_size``, ``num_attention_heads``: Parameters defining the architecture of the text transformer. The ``ffn_hidden_size`` is typically 4 times the ``hidden_size``.
- ``hidden_dropout`` and ``attention_dropout``: Dropout probabilities for the hidden state and attention in the transformer respectively.

Optimizations
^^^^^^^^^^^^^^

+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Feature                  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | To Enable                                                                                                                                                                                                        |
+==========================+=========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================+==================================================================================================================================================================================================================+
| Data parallelism         | Dataset is read concurrently across multiple GPUs or nodes, allowing for faster data loading and processing.                                                                                                                                                                                                                                                                                                                                                                                            | Automatically when training on multi GPUs/nodes                                                                                                                                                                  |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Tensor parallelism       | Each tensor is split up into multiple chunks, allowing for horizontal parallelism across GPUs. This technique, known as TensorParallel (TP), distributes the model's tensors across multiple GPUs. During processing, each shard gets processed separately and in parallel on different GPUs, and the results are synced at the end of the step. This approach is inspired by NVIDIA's Megatron implementation. [Reference](https://github.com/NVIDIA/Megatron-LM#distributed-pretraining)              | ``model.tensor_model_parallel_size={parallel_size}``                                                                                                                                                             |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Activation Checkpointing | To reduce memory usage, activations of certain layers are cleared and recomputed during a backward pass. This technique is particularly useful for training large models that wouldn't fit in GPU memory using traditional methods.                                                                                                                                                                                                                                                                     | ``model.vision.activations_checkpoint_granularity=full``, ``model.vision.activations_checkpoint_method=block``, ``model.vision.activations_checkpoint_num_layers={num_layers_to_check}`` (Same for ``model.llm``)|
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Bfloat16 Training        | Training is conducted in Bfloat16 precision, which offers a balance between the higher precision of FP32 and the memory savings and speed of FP16.                                                                                                                                                                                                                                                                                                                                                      | ``trainer.precision=bf16``                                                                                                                                                                                       |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| BF16 O2                  | Enables O2-level automatic mixed precision, optimizing Bfloat16 precision for better performance.                                                                                                                                                                                                                                                                                                                                                                                                       | ``model.megatron_amp_O2=True``                                                                                                                                                                                   |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Distributed Optimizer    | The optimization process is distributed across multiple GPUs, reducing memory requirements. This technique distributes the optimizer state across data parallel ranks, rather than replicating it, offering significant memory savings. This approach is inspired by the ZeRO optimization described in the paper "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" and implemented in NVIDIA's Megatron. [Reference](https://github.com/NVIDIA/Megatron-LM#distributed-optimizer) | ``model.optim.name="distributed_fused_adam"``                                                                                                                                                                    |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Flash Attention V2       | FlashAttention is a fast and memory-efficient algorithm to compute exact attention. It speeds up model training and reduces memory requirement by being IO-aware. This approach is particularly useful for large-scale models and is detailed further in the repository linked. [Reference](https://github.com/Dao-AILab/flash-attention)                                                                                                                                                               | ``model.vision.use_flash_attention=True``, ``model.llm.use_flash_attention=True``                                                                                                                                |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Model Training
-------------------
Refer to https://laion.ai/blog/large-openclip/#results for community training recipe.

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS-CLIP
    :keyprefix: mm-models-clip-
