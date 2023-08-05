NeMo TTS Collection API
=======================

Model Classes
-------------
Mel-Spectrogram Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nemo.collections.tts.models.FastPitchModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.tts.models.MixerTTSModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.tts.models.RadTTSModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.tts.models.Tacotron2Model
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.tts.models.SpectrogramEnhancerModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Speech-to-Text Aligner Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nemo.collections.tts.models.AlignerModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Two-Stage Models
~~~~~~~~~~~~~~~~~
.. autoclass:: nemo.collections.tts.models.TwoStagesModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Vocoders
~~~~~~~~
.. autoclass:: nemo.collections.tts.models.GriffinLimModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.tts.models.HifiGanModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.tts.models.UnivNetModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.tts.models.WaveGlowModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Base Classes
----------------

The classes below are the base of the TTS pipeline.
To read more about them, see the `Base Classes <./intro.html#Base Classes>`__ section of the intro page.

.. autoclass:: nemo.collections.tts.models.base.MelToSpec
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.models.base.SpectrogramGenerator
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.models.base.Vocoder
    :show-inheritance:
    :members:


Dataset Processing Classes
--------------------------
.. autoclass:: nemo.collections.tts.data.dataset.MixerTTSXDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.data.dataset.TTSDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.data.dataset.VocoderDataset
    :show-inheritance:
    :members:
