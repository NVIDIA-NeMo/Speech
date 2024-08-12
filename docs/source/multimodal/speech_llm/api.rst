SpeechLLM API
=============

Model Classes
-------------

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_base_model.MegatronBaseModel
    :show-inheritance:
    :no-members:
    :members: __init__, configure_optimizers
    :noindex:


.. autoclass:: nemo.collections.multimodal.speech_llm.models.modular_models.ModularAudioGPTModel
    :show-inheritance:
    :no-members:
    :members: __init__, training_step, validation_step, setup, build_train_valid_test_datasets


.. autoclass:: nemo.collections.multimodal.speech_llm.models.modular_models.CrossAttendModularAudioGPTModel
    :show-inheritance:
    :no-members:
    :members: __init__, training_step, validation_step, setup, build_train_valid_test_datasets


.. autoclass:: nemo.collections.multimodal.speech_llm.models.modular_t5_models.ModularizedAudioT5Model
    :show-inheritance:
    :no-members:
    :members: __init__, training_step, validation_step, setup, build_train_valid_test_datasets


.. autoclass:: nemo.collections.multimodal.speech_llm.models.modular_t5_models.DecoderTextPromptModularizedAudioT5Model
    :show-inheritance:
    :no-members:
    :members: __init__, training_step, validation_step, setup, build_train_valid_test_datasets



Modules
-------

.. autoclass:: nemo.collections.multimodal.speech_llm.modules.perception_modules.AudioPerceptionModule
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.speech_llm.modules.perception_modules.MultiAudioPerceptionModule
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.speech_llm.modules.TransformerCrossAttention
    :show-inheritance:
    :no-members:


Dataset Classes
---------------
.. autoclass:: nemo.collections.multimodal.speech_llm.data.audio_text_dataset.AudioTextDataset
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.speech_llm.data.audio_text_dataset.TarredAudioTextDataset
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.speech_llm.data.audio_text_dataset.get_tarred_audio_text_dataset_from_config
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.speech_llm.data.audio_text_dataset.get_audio_text_dataset_from_config
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.speech_llm.data.lhotse_dataset.LhotseAudioQuestionAnswerDataset
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.speech_llm.data.build_dataset.build_speechllm_dataset
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.multimodal.speech_llm.data.build_dataset.build_speechllm_dataloader
    :show-inheritance:
    :no-members:





