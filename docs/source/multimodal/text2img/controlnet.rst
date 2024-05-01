ControlNet
===================

Model Introduction
--------------------

ControlNet :cite:`mm-models-cn-controlnetgithub` is a neural network structure to control diffusion models by adding extra conditions.
It copies the weights of neural network blocks into a "locked" copy and a "trainable" copy. The "trainable" one learns your condition. The "locked" one preserves your model. In this way, the ControlNet can reuse the SD encoder as a deep, strong, robust, and powerful backbone to learn diverse controls.
NeMo Multimodal provides a training pipeline and example implementation for generating images based on segmentation maps. Users have the flexibility to explore other implementations using their own control input dataset and recipe.

.. image:: ./images/controlnet-structure.png
   :alt: ControlNet structure on stable diffusion (See :cite:`mm-models-cn-controlnetgithub`)


ControlNet Dataset
^^^^^^^^^^^^^^^^^^^^

ControlNet employs the WebDataset format for data ingestion. (See :doc:`Datasets<./datasets>`) Beyond the essential image-text pairs saved in tarfiles with matching names but distinct extensions (like 000001.jpg and 000001.txt), ControlNet also requires control input within the tarfiles, identifiable by their specific extension. By default, the control input should be stored as 000001.png for correct loading and identification in NeMo's implementation.

Model Configuration
--------------------

Even though the original copy of Stable Diffusion weights is locked, proper configuration settings toghether with a compatible pre-trained checkpoint are required for initialization. See :ref:`sd-config-section` for more details about ``unet_config``, ``first_stage_config`` and ``cond_stage_config``.

Contol Stage Config
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    control_stage_config:
        _target_: nemo.collections.multimodal.models.controlnet.controlnet.ControlNet
        params:
          from_pretrained_unet: /ckpts/v1-5-pruned.ckpt
          from_NeMo: False
          image_size: 32 # unused
          in_channels: 4
          hint_channels: 3
          model_channels: 320
          attention_resolutions: [ 4, 2, 1 ]
          num_res_blocks: 2
          channel_mult: [ 1, 2, 4, 4 ]
          num_heads: 8
          use_spatial_transformer: True
          use_linear_in_transformer: False
          transformer_depth: 1
          context_dim: 768
          use_checkpoint: False
          legacy: False
          use_flash_attention: True

- ``from_pretrained_unet``: Same logic as ``unet_config.from_pretrained``, adjust the from_NeMo based on the checkpoint's source, whether it's from Huggingface or NeMo.


- ``control_stage_config``: Outlines the architecture for the trainable copy of U-Net. It's essential that all parameters align with the U-Net checkpoint specified in this section.

- ``hint_channels``: Represents the channels of input controls, which is 3 in the mentioned example due to the RGB image input having a shape of (H, W, 3).

ControlNet Training Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    model:
        control_key: hint
        only_mid_control: False
        sd_locked: True
        ...


- ``contorl_key``: Identifier of the control input, ``.png`` files will be converted to dictionary for dataloaders with their keys being ``hint``.

- ``only_mid_control``: When set to True, during training, only the output from the middle block of the trainable copy will be incorporated into the locked copy.

- ``sd_locked``: Whether to lock the original stable diffusion weights during training.


Optimization related configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------+-----------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| Feature                  | Description                                                                                               | To Enable                                                                                                  |
+==========================+===========================================================================================================+============================================================================================================+
| Data parallelism         | Dataset read concurrently                                                                                 | Automatically when training on multi GPUs/nodes                                                            |
+--------------------------+-----------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| Activation Checkpointing | Reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass | ``model.unet_config.use_checkpoint=True``                                                                  |
+--------------------------+-----------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| Bfloat16 Training        | Training in Bfloat16 precision                                                                            | ``trainer.precision=bf16``                                                                                 |
+--------------------------+-----------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| Flash Attention          | Fast and Memory-Efficient Exact Attention with IO-Awareness                                               | ``model.unet_config.use_flash_attention=True`` &&  ``model.control_stage_config.use_flash_attention=True`` |
+--------------------------+-----------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| Channels Last            | Ordering NCHW tensors in memory preserving dimensions ordering.                                           | ``model.channels_last=True``                                                                               |
+--------------------------+-----------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| Inductor                 | TorchInductor compiler                                                                                    | ``model.inductor=True``                                                                                    |
+--------------------------+-----------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+




Reference
-----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS-CN
    :keyprefix: mm-models-cn-
