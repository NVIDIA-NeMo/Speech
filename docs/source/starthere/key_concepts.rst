.. _key-concepts:

Key Concepts in Speech AI
=========================

This page introduces the fundamental concepts you'll encounter when working with speech models in NeMo. No prior NeMo experience is required — we start from the basics of audio and work up to how NeMo structures its models.

Audio Basics
------------

**Sampling Rate**
   Audio is a continuous signal, but computers work with discrete samples. The *sampling rate* (measured in Hz) is how many samples per second are captured. Most speech models expect **16,000 Hz** (16 kHz) audio. If your audio has a different rate, you'll need to resample it.

**Mono vs. Stereo**
   Speech models typically expect **mono** (single-channel) audio. If you have stereo audio, convert it first:

   .. code-block:: bash

      ffmpeg -i input.mp3 -ac 1 -ar 16000 -y output.wav

**Mel-Spectrogram**
   A mel-spectrogram is a visual representation of audio that shows how energy is distributed across frequency bands (on the mel scale) over time. Many speech models convert raw audio into mel-spectrograms as a preprocessing step before feeding data to a neural network.


Speech AI Tasks
---------------

NeMo supports several speech AI tasks, each solving a different problem:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Task
     - What it does
     - Example use case
   * - **ASR** (Automatic Speech Recognition)
     - Converts spoken audio to text
     - Transcribing meetings, voice assistants
   * - **TTS** (Text-to-Speech)
     - Generates natural speech from text
     - Audiobooks, voice interfaces
   * - **Speaker Diarization**
     - Determines "who spoke when"
     - Meeting transcription with speaker labels
   * - **Speaker Recognition**
     - Identifies or verifies a speaker's identity
     - Voice authentication, speaker search
   * - **Speech Enhancement**
     - Improves audio quality (removes noise)
     - Preprocessing noisy recordings
   * - **SpeechLM**
     - Augments LLMs with audio understanding
     - Audio-aware agents, speech translation, reasoning about audio


Encoder Architectures
---------------------

The *encoder* converts audio features into a sequence of high-level representations:

**FastConformer**
   NeMo's default encoder. A faster variant of Conformer that combines self-attention (for global context) with depthwise convolutions (for local patterns). ~2.4x faster than standard Conformer.

**Conformer**
   The original architecture from `Gulati et al. (2020) <https://arxiv.org/abs/2005.08100>`_ that combines transformers and convolutions. FastConformer is recommended for new projects.


How NeMo Models Work
---------------------

Every NeMo model wraps these components into a single, cohesive unit:

.. raw:: html

   <div style="margin: 24px 0; overflow-x: auto;">
   <svg viewBox="0 0 820 130" xmlns="http://www.w3.org/2000/svg" style="max-width:820px; width:100%; height:auto; font-family:'NVIDIA Sans',sans-serif;">
     <defs>
       <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#76b900"/></marker>
     </defs>
     <!-- Preprocessor -->
     <rect x="0" y="20" width="140" height="70" rx="8" fill="#76b900" opacity="0.15" stroke="#76b900" stroke-width="2"/>
     <text x="70" y="48" text-anchor="middle" font-weight="700" font-size="13" fill="#333">Preprocessor</text>
     <text x="70" y="66" text-anchor="middle" font-size="10" fill="#555">Audio &#8594; Mel-spectrogram</text>
     <!-- Arrow 1 -->
     <line x1="140" y1="55" x2="170" y2="55" stroke="#76b900" stroke-width="2" marker-end="url(#arrow)"/>
     <!-- Encoder -->
     <rect x="170" y="20" width="140" height="70" rx="8" fill="#76b900" opacity="0.15" stroke="#76b900" stroke-width="2"/>
     <text x="240" y="48" text-anchor="middle" font-weight="700" font-size="13" fill="#333">Encoder</text>
     <text x="240" y="66" text-anchor="middle" font-size="10" fill="#555">Features &#8594; Hidden repr.</text>
     <!-- Arrow 2 -->
     <line x1="310" y1="55" x2="340" y2="55" stroke="#76b900" stroke-width="2" marker-end="url(#arrow)"/>
     <!-- Decoder -->
     <rect x="340" y="20" width="140" height="70" rx="8" fill="#76b900" opacity="0.15" stroke="#76b900" stroke-width="2"/>
     <text x="410" y="48" text-anchor="middle" font-weight="700" font-size="13" fill="#333">Decoder</text>
     <text x="410" y="66" text-anchor="middle" font-size="10" fill="#555">Hidden repr. &#8594; Output</text>
     <!-- Arrow 3 -->
     <line x1="480" y1="55" x2="510" y2="55" stroke="#76b900" stroke-width="2" marker-end="url(#arrow)"/>
     <!-- Loss -->
     <rect x="510" y="20" width="140" height="70" rx="8" fill="#76b900" opacity="0.15" stroke="#76b900" stroke-width="2"/>
     <text x="580" y="48" text-anchor="middle" font-weight="700" font-size="13" fill="#333">Loss Function</text>
     <text x="580" y="66" text-anchor="middle" font-size="10" fill="#555">Measures quality</text>
     <!-- Arrow 4 -->
     <line x1="650" y1="55" x2="680" y2="55" stroke="#76b900" stroke-width="2" marker-end="url(#arrow)"/>
     <!-- Optimizer -->
     <rect x="680" y="20" width="140" height="70" rx="8" fill="#76b900" opacity="0.15" stroke="#76b900" stroke-width="2"/>
     <text x="750" y="48" text-anchor="middle" font-weight="700" font-size="13" fill="#333">Optimizer</text>
     <text x="750" y="66" text-anchor="middle" font-size="10" fill="#555">Updates weights</text>
     <!-- Feedback arrow from Optimizer back to Encoder -->
     <path d="M750,90 L750,120 L240,120 L240,96" fill="none" stroke="#76b900" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arrow)"/>
     <text x="495" y="116" text-anchor="middle" font-size="9" fill="#76b900" font-style="italic">backpropagation</text>
   </svg>
   </div>

NeMo models are PyTorch modules that also integrate with `PyTorch Lightning <https://lightning.ai/>`__ for training and `Hydra <https://hydra.cc/>`__ + `OmegaConf <https://omegaconf.readthedocs.io/>`__ for configuration.


Configuration with YAML
------------------------

NeMo experiments are configured with YAML files. A typical config has three main sections:

.. code-block:: yaml

   model:
     # Model architecture, data, loss, optimizer
     encoder:
       _target_: nemo.collections.asr.modules.ConformerEncoder
       feat_in: 80
       n_layers: 17
       ...
     train_ds:
       manifest_filepath: /path/to/train_manifest.json
       batch_size: 32
     optim:
       name: adamw
       lr: 0.001

   trainer:
     # PyTorch Lightning trainer settings
     devices: 4
     accelerator: gpu
     max_steps: 100000
     precision: bf16-mixed

   exp_manager:
     # Experiment logging and checkpointing
     exp_dir: /path/to/experiments
     name: my_asr_experiment

You can override any value from the command line:

.. code-block:: bash

   python train_script.py \
       model.optim.lr=0.0005 \
       model.train_ds.manifest_filepath=/data/train.json \
       trainer.devices=8


Manifest Files
--------------

NeMo uses **manifest files** (JSONL format) to describe datasets. Each line is one training example:

.. code-block:: json

   {"audio_filepath": "/data/audio/001.wav", "text": "hello world", "duration": 2.5}
   {"audio_filepath": "/data/audio/002.wav", "text": "how are you", "duration": 1.8}

Key fields:

- ``audio_filepath`` — path to the audio file
- ``text`` — the transcript (for ASR) or input text (for TTS)
- ``duration`` — audio duration in seconds

See :doc:`../asr/datasets` for details on preparing datasets.


Model Checkpoints
-----------------

NeMo models are saved as ``.nemo`` files — tar archives containing model weights, configuration, and tokenizer files. You can load models in two ways:

.. code-block:: python

   # From a pretrained checkpoint (downloads from HuggingFace/NGC)
   model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

   # From a local .nemo file
   model = nemo_asr.models.ASRModel.restore_from("path/to/model.nemo")

See :doc:`../checkpoints/intro` for more details on checkpoint formats.

