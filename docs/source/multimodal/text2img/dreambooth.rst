DreamBooth
===================


Model Introduction
--------------------

DreamBooth :cite:`mm-models-db-dreamboothpaper` is a fine-tuning technique and a solution to personalize large diffusion models like Stable Diffusion, which are powerful but lack the
ability to mimic subjects of a given reference set. With DreamBooth, you only need a few images of a specific subject to
fine-tune a pretrained text-to-image model, so that it learns to bind a unique identifier with a special subject. This
unique identifier can then be used to synthesize fully-novel photorealistic images of the subject contextualized in
different scenes.

NeMo's Dreambooth is built upon the Stable Diffusion framework. While its architecture mirrors Stable Diffusion (refer to :ref:`sd-config-section`), the distinction lies in its training process, specifically when utilizing a different dataset and incorporating the prior preservation loss when necessary.

- Prior Preservation Loss

    When finetuning large pretrained language models on specific tasks or text-to-image diffusion models on a small dataset, problems like language drift and decreased output variety often arise. The concept of the prior preservation loss is straightforward: it guides the model using its self-generated samples and incorporates the discrepancy between the model-predicted noise on these samples. The influence of this loss component can be adjusted using model.prior_loss_weight.

.. code-block:: python

    model_pred, model_pred_prior = torch.chunk(model_output, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)
    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
    prior_loss = torch.nn.functional.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
    loss = loss + prior_loss * self.prior_loss_weight


- Training Dataset

    NeMo's Dreambooth model dataset is different from other NeMo multimodal models in that it doesn't necessitate data stored in the webdataset format. You can find a sample dataset at :cite:`mm-models-db-dreamboothdataset`. For each object you aim to integrate into the model, just place its images (typically 3-5) in a folder and specify its path in ``model.data.instance_dir``. When training with the prior preservation loss, store images produced by the original model in a distinct folder and reference its path in ``model.data.regularization_dir``. This process is automated in NeMo's DreamBooth implementation.

Model Configuration
--------------------

Pleaser refer to :ref:`sd-config-section` for how to configure Stable Diffusion. Here we show DreamBooth-specific configurations.

Prior Preservation Loss
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  model:
    with_prior_preservation: False
    prior_loss_weight: 0.5
    train_text_encoder: False
    restore_from_path: /ckpts/nemo-v1-5-188000-ema.nemo #This ckpt is only used to generate regularization images, thus .nemo ckpt is needed

    data:
      instance_dir: /datasets/instance_dir
      instance_prompt: a photo of a sks dog
      regularization_dir: /datasets/nemo_dogs
      regularization_prompt: a photo of a dog
      num_reg_images: 10
      num_images_per_prompt: 4
      resolution: 512
      center_crop: True


- ``train_text_encoder``: Dictates if the text encoder should be finetuned alongside the U-Net.

- ``with_prior_preservation``: Depending on its setting, this influences how the model behaves with respect to the regularization data. If set to ``False``, both ``model.prior_loss_weight`` and ``model.restore_from_path`` will be disregarded. If set to ``True``, the actions will differ based on the number of images present in ``model.data.regularization_dir``:

  #. If the count is fewer than ``model.data.num_reg_images``:

     + ``model.restore_from_path`` should be provided with a `.nemo` checkpoint, allowing the inference pipeline to produce regularization images.
     + ``model.data.num_images_per_prompt`` is analogous to the inference batch size and indicates the number of images generated in one pass, restricted by GPU capabilities.
     + ``model.regularization_prompt`` determines the text prompt for the inference pipeline to generate images. It's generally a variant of ``model.data.instance_prompt`` minus the unique token.
     + Once all above parameters are satisfied, the inference pipeline will run until the required image count is achieved in the regularization directory.

  #. If the count matches or exceeds ``model.data.num_reg_images``

     + Training will proceed without calling inference pipeline, and the parameters mentioned above will be ignored.

Optimization related configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------+-----------------------------------------------------------------------------------------------------------+-------------------------------------------------+
| Feature                  | Description                                                                                               | To Enable                                       |
+==========================+===========================================================================================================+=================================================+
| Data parallelism         | Dataset read concurrently                                                                                 | Automatically when training on multi GPUs/nodes |
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


Training with Cached Latents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    model:
        use_cached_latents: True

        data:
            num_workers: 4
            instance_dir: /datasets/instance_dir
            instance_prompt: a photo of a sks dog
            regularization_dir: /datasets/nemo_dogs
            regularization_prompt: a photo of a dog
            cached_instance_dir: #/datasets/instance_dir_cached
            cached_reg_dir: #/datasets/nemo_dogs_cached


- ``use_cached_latents``: Determines whether to train using online encoding or pre-cached latents.

- ``cached_instance_dir``:

  + If ``use_cached_latents`` is enabled and these directories with latents in `.pt` format are specified, training will utilize the latents rather than the original images.
  + If a cached directory isn't provided or the number of latent files doesn't match the original image count, the Variational Auto Encoder will compute the image latents before training, and the results will be saved on the disk.

- ``cached_reg_dir``:
  + The logic is consistent with above, contingent on the model.with_prior_preservation setting.





Reference
-----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS-DB
    :keyprefix: mm-models-db-
