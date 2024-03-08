Vision Models
=============

NeMo has implemented foundational vision models, establishing a solid base for further exploration into multimodal applications. These foundational vision models can be leveraged in a variety of multimodal applications including multimodal language models and text to image generation tasks, among others. These foundation models not only lay the functional groundwork but also play a crucial role in achieving state-of-the-art performance on NVIDIA GPUs through our custom optimizations.

Supported Models
-----------------
NeMo's vision foundation currently supports the following models:

+----------------------------------+----------+-------------+------+----------------------+------------------+
| Model                            | Training | Fine-Tuning | PEFT | Evaluation           | Inference        |
+==================================+==========+=============+======+======================+==================+
| Vision Transformer (ViT)         | ✓        | ✓           | ✗    | imagenet zero-shot   | ✗                |
+----------------------------------+----------+-------------+------+----------------------+------------------+
| AutoencoderKL (VAE with KL loss) | ✗        | ✗           | ✗    | ✗                    | To be added      |
+----------------------------------+----------+-------------+------+----------------------+------------------+

Spotlight Models
-----------------

1. **Vision Transformer (ViT)**:
   Vision Transformer (ViT) :cite:`vision-models-vit` stands as a compelling alternative to the traditionally employed Convolutional Neural Networks (CNNs) for image classification tasks. Unlike CNNs that work on the entire image, ViT divides an image into fixed-size patches, linearly embeds them into 1D vectors, and adds positional embeddings. These vectors are then fed into a Transformer encoder to capture both local and global features of the image. This model has shown to outperform CNNs in terms of computational efficiency and accuracy by a significant margin, making it a powerful tool for image-related tasks.

2. **AutoencoderKL (Variational Autoencoder with KL loss**:
   The AutoencoderKL model is a Variational Autoencoder (VAE) equipped with KL loss, introduced in the paper Auto-Encoding Variational Bayes by Diederik P. Kingma and Max Welling :cite:`vision-models-kingma2022autoencoding`. This model is adept at encoding images into latent representations and decoding these representations back into images. The KL divergence term in the loss function serves to align the distribution of the encoder output as closely as possible to a standard multivariate normal distribution, facilitating the exploration of the latent space. The continuous nature of the Variational Autoencoder's latent space enables random sampling and interpolation, which are crucial for tasks like image reconstruction and generation.

.. note::
    NeMo Megatron has an Enterprise edition which contains tools for data preprocessing, hyperparameter tuning, container, scripts for various clouds and more. With Enterprise edition you also get deployment tools. Apply for `early access here <https://developer.nvidia.com/nemo-megatron-early-access>`_ .


.. toctree::
   :maxdepth: 1

   datasets
   configs
   checkpoint
   vit

References
----------

.. bibliography:: ./vision_all.bib
    :style: plain
    :labelprefix: VISION-MODELS
    :keyprefix: vision-models-