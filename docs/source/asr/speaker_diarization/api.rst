NeMo Speaker Diarization API
=============================


Model Classes
-------------

.. autoclass:: nemo.collections.asr.models.ClusteringDiarizer
    :show-inheritance:
    :members:  

.. autoclass:: nemo.collections.asr.models.EncDecDiarLabelModel
    :show-inheritance:
    :members: add_speaker_model_config, _init_segmentation_info, _init_speaker_model, setup_training_data, setup_validation_data, setup_test_data, get_ms_emb_seq, get_cluster_avg_embs_model, get_ms_mel_feat, forward, forward_infer, training_step, validation_step, compute_accuracies

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
    