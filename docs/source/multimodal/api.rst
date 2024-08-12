NeMo Multimodal API
===================

Model Classes
-------------

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_base_model.MegatronBaseModel
    :show-inheritance:
    :no-members:
    :members: __init__, configure_optimizers
    :noindex:


.. autoclass:: nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm.MegatronLatentDiffusion
    :show-inheritance:
    :no-members:
    :members: __init__, training_step, validation_step, setup, build_train_valid_test_datasets


.. autoclass:: nemo.collections.multimodal.models.text_to_image.dreambooth.dreambooth.MegatronDreamBooth
    :show-inheritance:
    :no-members:
    :members: __init__, training_step, validation_step, setup, build_train_valid_test_datasets


.. autoclass:: nemo.collections.multimodal.models.text_to_image.controlnet.controlnet.MegatronControlNet
    :show-inheritance:
    :no-members:
    :members: __init__, training_step, validation_step, setup, build_train_valid_test_datasets

.. autoclass:: nemo.collections.multimodal.models.text_to_image.imagen.imagen.MegatronImagen
    :show-inheritance:
    :no-members:
    :members: __init__, training_step, validation_step, setup, build_train_valid_test_datasets



Modules
-------

.. autoclass:: nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel.UNetModel
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.modules.imagen.diffusionmodules.nets.UNetModel
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.modules.imagen.diffusionmodules.nets.EfficientUNetModel
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder.AutoencoderKL
    :show-inheritance:
    :no-members:
    :members: __init__, encode, decode

.. autoclass:: nemo.collections.multimodal.modules.stable_diffusion.encoders.modules.FrozenMegatronCLIPEmbedder
    :show-inheritance:
    :no-members:
    :members: __init__, forward

.. autoclass:: nemo.collections.multimodal.modules.imagen.encoder.t5encoder.T5Encoder
    :show-inheritance:
    :no-members:
    :members: __init__, encode


.. autoclass:: nemo.collections.multimodal.models.text_to_image.controlnet.controlnet.ControlledUnetModel
    :show-inheritance:
    :no-members:
    :members: forward

Datasets
---------

.. autoclass:: nemo.collections.multimodal.data.common.webdataset.WebDatasetCommon
    :show-inheritance:

.. autoclass:: nemo.collections.multimodal.data.dreambooth.dreambooth_dataset.DreamBoothDataset
    :show-inheritance:

