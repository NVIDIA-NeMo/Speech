Datasets
========

Use this page to prepare training or evaluation data and to choose between the NeMo and Lhotse loaders. The Audio
collection accepts NeMo JSON manifests, Lhotse CutSet manifests, and Lhotse Shar datasets. Training, validation, and
test data are configured under ``model.train_ds``, ``model.validation_ds``, and ``model.test_ds``. The examples below
show how the two loaders represent the same paired data.

Signal Roles
------------

The signal names describe their role in a training example:

* The **input** is the observed signal presented to the model, such as noisy, reverberant, or mixed audio.
* The **target** is the desired model output used to compute the training loss. Depending on the task, it can be clean
  speech, an anechoic signal, or the source to extract.
* A **reference** is an additional observation that helps identify or estimate the target. Examples include a target
  speaker enrollment utterance, a playback signal for echo cancellation, or a correlated signal from another sensor.
  It is not another name for the degraded input.
* An **embedding** is a precomputed conditioning vector used in place of reference audio.

Configurations for paired data consume input and target signals; the self-supervised pretraining configuration consumes
input only. Use the reference and embedding dataset classes when a custom model needs one of those additional inputs.

Choose a Data Loader
--------------------

The NeMo and Lhotse loaders represent the same paired example differently. With a NeMo JSON manifest, the dataset
configuration maps manifest keys to the input and target roles:

.. code-block:: yaml

   model:
     train_ds:
       manifest_filepath: /path/to/train.json
       input_key: input_filepath
       target_key: target_filepath

With a Lhotse CutSet, the cut's main recording is always the input and its ``target_recording`` custom field is the
target. The configuration selects the CutSet rather than naming JSON fields:

.. code-block:: yaml

   model:
     train_ds:
       use_lhotse: true
       cuts_path: /path/to/train_cuts.jsonl

A Lhotse Shar dataset uses the same signal roles but is selected with ``shar_path``:

.. code-block:: yaml

   model:
     train_ds:
       use_lhotse: true
       shar_path: /path/to/train_shar

For the paired CutSet and Lhotse Shar datasets used by Audio configurations, store the target in
``target_recording`` instead of setting ``input_key`` and ``target_key``. The snippets show only how each source is
selected and how Audio signal roles are assigned. See :doc:`Lhotse Dataloading </dataloaders>` for batching,
truncation, sharding, and other shared loader settings. Configure validation and test data with fields for the same
loader.

NeMo JSON Manifests
-------------------

A NeMo manifest is a JSON Lines file with one utterance per line. Paths may be absolute or relative to the manifest.
The key names are configurable, so ``noisy_filepath`` and ``clean_filepath`` can be used in place of the generic
``input_filepath`` and ``target_filepath`` names below.

Input and Target Audio
~~~~~~~~~~~~~~~~~~~~~~

Most enhancement and restoration tasks use paired input and target recordings:

.. code-block:: json

   {"input_filepath": "audio/noisy.wav", "target_filepath": "audio/clean.wav", "duration": 3.147}

Use :class:`~nemo.collections.audio.data.audio_to_audio.AudioToTargetDataset` for this layout. A value may also be a
list of synchronized mono files; the files are combined as channels of one recording. The optional ``offset`` field
defaults to zero. It is the fixed start time when ``random_offset`` is false and the earliest possible start time when
``random_offset`` is true. The loader uses it for synchronized signals and for a reference loaded independently.

Input and target recordings must be time-aligned and provide the same usable duration. Training uses the input length
when encoding the target and computing the loss.

Reference Audio
~~~~~~~~~~~~~~~

Models for extraction, echo cancellation, or sensor fusion can use an additional reference recording:

.. code-block:: json

   {"input_filepath": "audio/mixture.wav", "target_filepath": "audio/target.wav", "reference_filepath": "audio/enrollment.wav", "duration": 3.147}

Use :class:`~nemo.collections.audio.data.audio_to_audio.AudioToTargetWithReferenceDataset`. Set
``reference_is_synchronized: false`` for an independently sampled reference, such as an enrollment utterance, and use
``reference_duration`` when a fixed reference segment is required.

Embedding Vectors
~~~~~~~~~~~~~~~~~

A model can use a precomputed conditioning vector in place of reference audio:

.. code-block:: json

   {"input_filepath": "audio/mixture.wav", "target_filepath": "audio/target.wav", "embedding_filepath": "embeddings/target.npy", "duration": 3.147}

Use :class:`~nemo.collections.audio.data.audio_to_audio.AudioToTargetWithEmbeddingDataset`. Embeddings must be stored
as NumPy ``.npy`` arrays.

Duration, Channels, and Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``audio_duration`` requests synchronized input and target segments of a fixed length. If a synchronized recording is
shorter, the loader shortens the example to the shortest recording. Set ``min_duration`` to at least
``audio_duration`` and ensure the target files are not shorter when fixed-length batches are required. With
``random_offset: true``, the segment start is randomized whenever the dataset is sampled. ``min_duration`` and
``max_duration`` filter complete utterances, while ``max_utts`` limits the number loaded.

The ``input_channel_selector``, ``target_channel_selector``, and ``reference_channel_selector`` fields select channels
from multichannel files. A selector may be a channel index, a list of indices, or ``average``. When no selector is
provided, all channels are loaded.

``normalization_signal`` can be ``input_signal``, ``target_signal``, or ``reference_signal`` for datasets that provide
that signal. The selected signal determines a scale that is applied consistently to every loaded signal in the
example, including a non-synchronized reference signal.

Representing Audio Examples with Lhotse
----------------------------------------

Set ``use_lhotse: true`` to use :class:`~nemo.collections.audio.data.audio_to_audio_lhotse.LhotseAudioToTargetDataset`.
The input signal is the cut's recording. Optional custom fields provide the remaining inputs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Custom field
     - Meaning
   * - ``target_recording``
     - Target audio synchronized with the input cut.
   * - ``reference_recording``
     - Optional reference audio synchronized with the input cut.
   * - ``embedding_vector``
     - Optional conditioning array.

CutSet and Shar storage, ``input_cfg``, sampling, weighting, and augmentation are shared with other NeMo collections.
See :doc:`Lhotse Dataloading </dataloaders>` for those options and the complete ``LhotseDataLoadingConfig`` reference.

Convert a NeMo manifest to a Lhotse CutSet with:

.. code-block:: bash

   python scripts/audio_to_audio/convert_nemo_to_lhotse.py \
       /path/to/nemo_manifest.json \
       /path/to/lhotse_manifest.jsonl \
       --input_key input_filepath \
       --target_key target_filepath

If the source manifest uses relative paths, write the CutSet in a directory where those paths still resolve. Use
``--force_absolute_paths`` before exporting the CutSet to a Lhotse Shar dataset. The Audio
``save_augmented.py`` workflow instead requires relative recording paths. Run ``--help`` to see the available
conversion options.
