Stable Diffusion
================

This section gives a brief overview of the stable diffusion model in NeMo Framework.

Model Introduction
--------------------

Stable Diffusion stands out as an advanced text-to-image diffusion model, trained using a massive dataset of image,text pairs. Its core capability is to refine and enhance images by eliminating noise, resulting in clear output visuals. When presented with an image named z0, the model systematically injects noise. With each iteration marked by "t", the image, now termed zt, becomes increasingly distorted. As the value of "t" climbs, the image edges closer to resembling complete noise. Moreover, when provided with specific details like a text prompt or the time step "t", the model can accurately determine the extent of noise introduced to zt.

Stable diffusion has three main components: A U-Net, an image encoder(Variational Autoencoder, VAE) and a text-encoder(CLIP).


- U-Net: The Unet processes the noisy latents (x) to predict the noise, utilizing a conditional model which also incorporates the timestep (t) and text embedding for guidance.

- VAE: The VAE model, equipped with both an encoder and decoder, engages in image compression during latent diffusion training. In a standard Stable Diffusion training stage, for instance, an input image is condensed from 512x512x3 dimensions to 64x64x4. This compression results in decreased memory and computational requirements when compared to pixel-space diffusion models. Subsequently, during inference, the decoder reverses this process by transforming denoised latent representations back into their original, tangible image forms.

- Text-encoder: The text-encoder, typically a simple transformer like CLIP, converts input prompts into embeddings, which guides the U-Net's denoising process. These embeddings help train the U-Net to handle noisy latents effectively.

.. _sd-config-section:

Model Configuration
--------------------


In this section, we explain how to configure the size and initialization of the VAE, U-Net, and text encoder components of the Stable Diffusion model.

Variational Auto Encoder
^^^^^^^^^^^^^^^^^^^^^^^^^

The VAE configuration is defined under **first_stage_config**.

.. code-block:: yaml

    first_stage_config:
        _target_: nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder.AutoencoderKL
        from_pretrained: /path/to/vae.bin
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256  #Never used
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

The VAE weights are fixed during training, and it's essential to pass a pretraiend checkpoint to ``first_stage_config.from_pretrained`` for initialization. The VAE architecture is shared for Stable diffusion v1 and v2 series. The scaling factor of VAE is ``2**(len(ch_mult - 1))``, which is 8 in this case. Thus the output image shape will be ``(H//8, W//8, 4)``.


U-Net
^^^^^^

The U-Net configuration is defined under **unet_config**.

.. code-block:: yaml

    unet_config:
        _target_: nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel.UNetModel
        from_pretrained: /path/to/pretrain.ckpt
        from_NeMo: True #Must be specified when from pretrained is not None, False means loading unet from HF ckpt
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions:
        - 4
        - 2
        - 1
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 4
        - 4
        num_head_channels: 64
        use_spatial_transformer: true
        use_linear_in_transformer: true
        transformer_depth: 1
        context_dim: 1024
        use_checkpoint: False
        legacy: False
        use_flash_attention: True

- If ``from_pretrained`` is not specified, the U-Net initializes with random weights. To fine-tune, you can provide a pretrained U-Net checkpoint, either from an intermediate NeMo checkpoint (set ``from_NeMo=True``) or from other platforms like Huggingface (set ``from_NeMo=False``).

- U-Net size
    + ``num_res_blocks``: Defines the count of resnet blocks at every level.
    + ``model_channels`` and ``channel_mult``: Set the tensor dimensions for each level.

- Attention blocks
    + ``attention_resolution``: Integrates attention blocks after the resnet block of every level.
    + ``use_spatial_transformer``: Specifies the type of attention block employed.
    + ``use_linear_in_transformer``: Chooses between a linear layer and convolution layer for in/out projections.
    + ``transformer_depth``: Dictates the count of ``basic_transformer_block`` in each ``spatial_transformer_block``.

- ``context_dim``: Must be adjusted to match the text encoder's output dimension.

Text Encoder
^^^^^^^^^^^^
The text encoder configuration is defined under **cond_stage_config**.

To use the NeMo implementation of the CLIP model in stable diffusion, one can use the following cond_stage_config:

.. code-block:: yaml

      cond_stage_config:
        _target_: nemo.collections.multimodal.modules.stable_diffusion.encoders.modules.FrozenMegatronCLIPEmbedder
        restore_from_path: /path/to/nemo_clip.nemo
        device: cuda
        freeze: True
        layer: "penultimate"

- ``restore_from_path``: Must be provided to use NeMo CLIP models, all CLIP config-related information is already embeded in ``.nemo`` checkpoint file.

- ``layer``: Specifies which layer's output will be used as text encoder output.

Alternatively, one can also use the Huggingface implementation of the CLIP model using the config below

.. code-block:: yaml

    cond_stage_config:
        _target_: nemo.collections.multimodal.modules.stable_diffusion.encoders.modules.FrozenOpenCLIPEmbedder
        arch: ViT-H-14
        version: laion2b_s32b_b79k
        device: cuda
        max_length: 77
        freeze: True
        layer: "penultimate"

- ``arch`` and ``version``: Determines which CLIP model to load.


Optimization related configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------+-----------------------------------------------------------------------------------------------------------+-------------------------------------------------+
| Feature                  | Description                                                                                               | To Enable                                       |
+==========================+===========================================================================================================+=================================================+
| Data parallelism         | Dataset read concurrently                                                                                 | default when training on multi GPUs/nodes       |
+--------------------------+-----------------------------------------------------------------------------------------------------------+-------------------------------------------------+
| Activation Checkpointing | Reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass | ``model.unet_config.use_checkpoint=True``       |
+--------------------------+-----------------------------------------------------------------------------------------------------------+-------------------------------------------------+
| Bfloat16 Training        | Training in Bfloat16 precision                                                                            | ``trainer.precision=bf16``                      |
+--------------------------+-----------------------------------------------------------------------------------------------------------+-------------------------------------------------+
| Flash Attention          | Fast and Memory-Efficient Exact Attention with IO-Awareness                                               | ``model.unet_config.use_flash_attention=True``  |
+--------------------------+-----------------------------------------------------------------------------------------------------------+-------------------------------------------------+
| Channels Last            | Ordering NCHW tensors in memory preserving dimensions ordering.                                           | ``model.channels_last=True``                    |
+--------------------------+-----------------------------------------------------------------------------------------------------------+-------------------------------------------------+
| Inductor                 | TorchInductor compiler                                                                                    | ``model.inductor=True``                         |
+--------------------------+-----------------------------------------------------------------------------------------------------------+-------------------------------------------------+

Training with precached latents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since the VAE and text encoder remain frozed during training, you can pre-calculate the image and caption latents offline, enhancing training throughput. To create a pre-cached dataset, see :doc:`Multimodal Dataset <./datasets>`. For training using this dataset, configure ``model.data`` section properly and set ``model.first_stage_key=image_encoded`` along with ``model.cond_stage_key=captions_encoded``.

Reference
-----------

