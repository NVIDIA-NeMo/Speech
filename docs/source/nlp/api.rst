NeMo Large language Model API
=============================

Pretraining Model Classes
-------------------------

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_base_model.MegatronBaseModel
    :show-inheritance: 
    :no-members:
    :members: __init__, configure_optimizers

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel
    :show-inheritance:
    :no-members:
    :members: generate, training_step, validation_step, build_train_valid_test_datasets, setup, on_save_checkpoint, on_load_checkpoint

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_bert_model.MegatronBertModel
    :show-inheritance: 
    :no-members:
    :members: training_step, validation_step, build_train_valid_test_datasets, build_LDDL_data, setup, on_save_checkpoint, on_load_checkpoint

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_bart_model.MegatronBARTModel
    :show-inheritance: 
    :no-members:
    :members: training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_retrieval_model.MegatronRetrievalModel
    :show-inheritance: 
    :no-members:
    :members: generate, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model
    :show-inheritance: 
    :no-members:
    :members: complete, encode, decode, add_special_tokens_to_tokenizer, training_step, validation_step, build_train_valid_test_datasets, setup

Customization Model Classes
---------------------------

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model.MegatronGPTSFTModel
    :show-inheritance: 
    :no-members:
    :members: generate, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_adapter_model.MegatronGPTAdapterLearningModel
    :show-inheritance: 
    :no-members:
    :members: __init__, state_dict, generate, training_step, validation_step, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_adapter_model.MegatronGPTInfusedAdapterModel
    :show-inheritance: 
    :no-members:
    :members: __init__, state_dict, generate, training_step, validation_step, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model.MegatronGPTPromptLearningModel
    :show-inheritance: 
    :no-members:
    :members: build_virtual_prompt_dataset, generate, training_step, validation_step, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model.MegatronT5AdapterLearningModel
    :show-inheritance: 
    :no-members:
    :members: _add_adapters_to_component, __init__, state_dict, training_step, validation_step, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model.MegatronT5InfusedAdapterModel
    :show-inheritance: 
    :no-members:
    :members: _add_adapters_to_component, __init__, state_dict, training_step, validation_step, setup

Modules
-------

.. autoclass:: nemo.collections.nlp.modules.common.megatron.module.MegatronModule
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.modules.common.megatron.module.Float16Module
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron.gpt_model.GPTModel
    :show-inheritance: 
    :no-members:
    :members: forward

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron.bert.bert_model.NeMoBertModel
    :show-inheritance: 
    :no-members:
    :members: forward

.. autoclass:: nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder.MegatronTokenLevelEncoderDecoderModule
    :show-inheritance: 
    :no-members:
    :members: forward

.. autoclass:: nemo.collections.nlp.modules.common.megatron.retrieval_token_level_encoder_decoder.MegatronRetrievalTokenLevelEncoderDecoderModule
    :show-inheritance: 
    :no-members:
    :members: forward


Datasets
--------

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset.BlendableDataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset.GPTDataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset.MockGPTDataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.bert_dataset.BertDataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.base_prompt_learning_dataset.BasePromptLearningDataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset.GPTSFTDataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset.GPTSFTChatDataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.retro_dataset.RETRODataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.t5_dataset.T5Dataset
    :show-inheritance: 
    :exclude-members: MAX_SEQ_LENGTH_DELTA

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.t5_prompt_learning_dataset.T5PromptLearningDataset
    :show-inheritance: 

.. autoclass:: nemo.collections.nlp.data.language_modeling.megatron.ul2_dataset.UL2Dataset
    :show-inheritance: 


Adapter Mixin Class
-------------------------

.. autoclass:: nemo.collections.nlp.parts.mixins.nlp_adapter_mixins.NLPAdapterModelMixin
    :show-inheritance:
    :members: add_adapter, load_adapters, merge_cfg_with, merge_inference_cfg
    :exclude-members: first_stage_of_pipeline, tie_weights, get_peft_state_dict, state_dict, sharded_state_dict, load_state_dict, on_load_checkpoint
    :member-order: bysource


Exportable Model Classes
-------------------------

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTExportableModel
    :show-inheritance:

.. toctree::
   :maxdepth: 1

   megatron_onnx_export