Checkpoints
===========

There are two main ways to load pretrained checkpoints in NeMo as introduced in the :doc:`ASR checkpoints <../results>` section.
In speaker diarization, the diarizer loads checkpoints that are passed through the config file. 


End-to-end Speaker Diarization Models
=====================================

Sortformer Diarizer Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the following command to train a Sortformer diarizer model.
  
.. code-block:: bash  
  
  # Feed the config for Sortformer diarizer model training
  python ${NEMO_ROOT}/examples/speaker_tasks/diarization/neural_diarizer/sortformer_diar_train.py --config-path='../conf/neural_diarizer' \  
    --config-name='sortformer_diarizer_hybrid_loss_4spk-v1.yaml' \   
    trainer.devices=1 \  
    model.train_ds.manifest_filepath="<train_manifest_path>" \  
    model.validation_ds.manifest_filepath="<dev_manifest_path>" \   
    exp_manager.name='sample_train' \   
    exp_manager.exp_dir=./sortformer_diar_train


Sortformer Diarizer Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the following command to run inference on a Sortformer diarizer model.

.. code-block:: bash  

  # Config for post-processing  
  PP_YAML1=${NEMO_ROOT}/examples/speaker_tasks/diarization/conf/post_processing/sortformer_diar_4spk-v1_dihard3-dev.yaml  
  PP_YAML2=${NEMO_ROOT}/examples/speaker_tasks/diarization/conf/post_processing/sortformer_diar_4spk-v1_callhome-part1.yaml   
  python ${NEMO_ROOT}/examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py \  
    batch_size=1 \  
    model_path=/path/to/diar_sortformer_4spk_v1.nemo \  
    postprocessing_yaml=${PP_YAML2} \  
    dataset_manifest=/path/to/diarization_manifest.json  

HuggingFace Pretrained Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ASR collection has checkpoints of several models trained on various datasets for a variety of tasks.
These checkpoints are obtainable via NGC `NeMo Automatic Speech Recognition collection <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`__.
The model cards on NGC contain more information about each of the checkpoints available.

In general, you can load models with model name in the following format, 

.. code-block:: bash

  pip install -U "huggingface_hub[cli]"
  huggingface-cli login

.. code-block:: python

  from nemo.collections.asr.models import SortformerEncLabelModel
  diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")

where the model name is the value under "Model Name" entry in the tables below.

End-to-end Speaker Diarization Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: data/e2e_diar_models.csv
   :align: left
   :widths: 30, 30, 40
   :header-rows: 1



Models for Cascaded Speaker Diarization Pipeline
================================================

Loading Local Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^

Load VAD models

.. code-block:: bash

  pretrained_vad_model='/path/to/vad_multilingual_marblenet.nemo' # local .nemo or pretrained vad model name
  ...
  # pass with hydra config
  config.diarizer.vad.model_path=pretrained_vad_model


Load speaker embedding models

.. code-block:: bash

  pretrained_speaker_model='/path/to/titanet-l.nemo' # local .nemo or pretrained speaker embedding model name
  ...
  # pass with hydra config
  config.diarizer.speaker_embeddings.model_path=pretrained_speaker_model

Load neural diarizer models

.. code-block:: bash

  pretrained_neural_diarizer_model='/path/to/diarizer_msdd_telephonic.nemo' # local .nemo or pretrained neural diarizer model name
  ...
  # pass with hydra config
  config.diarizer.msdd_model.model_path=pretrained_neural_diarizer_model


NeMo will automatically save checkpoints of a model you are training in a `.nemo` format.
You can also manually save your models at any point using :code:`model.save_to(<checkpoint_path>.nemo)`.


Inference
^^^^^^^^^

.. note::
  For details and deep understanding, please refer to ``<NeMo_root>/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb``.

Check out :doc:`Datasets <./datasets>` for preparing audio files and optional label files.

Run and evaluate speaker diarizer with below command:

.. code-block:: bash

  # Have a look at the instruction inside the script and pass the arguments you might need. 
  python <NeMo_root>/examples/speaker_tasks/diarization/offline_diarization.py 


NGC Pretrained Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ASR collection has checkpoints of several models trained on various datasets for a variety of tasks.
These checkpoints are obtainable via NGC `NeMo Automatic Speech Recognition collection <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`__.
The model cards on NGC contain more information about each of the checkpoints available.

In general, you can load models with model name in the following format, 

.. code-block:: python

  pretrained_vad_model='vad_multilingual_marblenet'
  pretrained_speaker_model='titanet_large'
  pretrained_neural_diarizer_model='diar_msdd_telephonic'
  ...
  config.diarizer.vad.model_path=retrained_vad_model \
  config.diarizer.speaker_embeddings.model_path=pretrained_speaker_model \
  config.diarizer.msdd_model.model_path=pretrained_neural_diarizer_model

where the model name is the value under "Model Name" entry in the tables below.

Models for Speaker Diarization Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: data/diarization_results.csv
   :align: left
   :widths: 30, 30, 40
   :header-rows: 1
