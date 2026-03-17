NeMo Speaker Diarization API
=============================


Model Classes
-------------

.. autoclass:: nemo.collections.asr.models.ClusteringDiarizer
    :show-inheritance:
    :members:  

.. autoclass:: nemo.collections.asr.models.SortformerEncLabelModel
    :show-inheritance:
    :members: list_available_models, setup_training_data, setup_validation_data, setup_test_data, process_signal, forward, forward_infer, frontend_encoder, diarize, training_step, validation_step, multi_validation_epoch_end, _get_aux_train_evaluations, _get_aux_validation_evaluations, _init_loss_weights, _init_eval_metrics, _reset_train_metrics, _reset_valid_metrics, _setup_diarize_dataloader, _diarize_forward, _diarize_output_processing, test_batch, _get_aux_test_batch_evaluations, on_validation_epoch_end

Mixins
------

.. autoclass:: nemo.collections.asr.parts.mixins.DiarizationMixin
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.mixins.diarization.SpkDiarizationMixin
    :show-inheritance:
    :members: diarize, diarize_generator, _diarize_on_begin, _diarize_input_processing, _diarize_input_manifest_processing, _setup_diarize_dataloader, _diarize_forward, _diarize_output_processing, _diarize_on_end, _input_audio_to_rttm_processing, get_value_from_diarization_config
    