NVIDIA NeMo Speech Developer Docs
=================================

`NVIDIA NeMo Speech <https://github.com/NVIDIA/NeMo>`_ is an open-source toolkit for speech, audio, and multimodal language model research, with a clear path from experimentation to production deployment.

.. raw:: html

   <style>
   .task-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin: 24px 0; }
   .task-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; text-decoration: none !important; color: inherit !important; transition: box-shadow 0.2s; }
   .task-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
   .task-card h3 { margin-top: 0; }
   .task-card p { color: #555; font-size: 0.95em; }
   </style>

   <div class="task-card-grid">
     <a class="task-card" href="asr/intro.html">
       <h3>🎙️ Transcribe Speech (ASR)</h3>
       <p>Convert audio to text with state-of-the-art accuracy. Supports 14+ languages, streaming, and timestamps.</p>
       <strong>Quick Start →</strong>
     </a>
     <a class="task-card" href="tts/intro.html">
       <h3>🔊 Synthesize Speech (TTS)</h3>
       <p>Generate natural human speech from text. Multi-language, multi-speaker, with controllable prosody.</p>
       <strong>Quick Start →</strong>
     </a>
     <a class="task-card" href="asr/speaker_diarization/intro.html">
       <h3>👥 Identify Speakers</h3>
       <p>Determine "who spoke when" in multi-speaker audio. Speaker diarization, recognition, and verification.</p>
       <strong>Quick Start →</strong>
     </a>
     <a class="task-card" href="speechlm2/intro.html">
       <h3>🧠 Speech Language Models</h3>
       <p>Audio-aware LLMs that understand and generate speech. Speech-to-text, speech-to-speech, and more.</p>
       <strong>Quick Start →</strong>
     </a>
     <a class="task-card" href="audio/intro.html">
       <h3>🎧 Process Audio</h3>
       <p>Enhance, restore, and separate audio signals. Improve audio quality for downstream tasks.</p>
       <strong>Quick Start →</strong>
     </a>
     <a class="task-card" href="tools/intro.html">
       <h3>🛠️ Speech AI Tools</h3>
       <p>Forced alignment, data exploration, CTC segmentation, and evaluation utilities for speech workflows.</p>
       <strong>Explore Tools →</strong>
     </a>
   </div>


What is NeMo?
--------------

`NVIDIA NeMo <https://github.com/NVIDIA/NeMo>`_ is an open-source toolkit for building, customizing, and deploying speech, audio, and multimodal language models. It provides:

- **Pretrained models** — production-ready checkpoints on `NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`__ and `HuggingFace Hub <https://huggingface.co/nvidia>`__
- **Modular architecture** — neural modules you can mix, match, and extend
- **Scalable training** — multi-GPU/multi-node via PyTorch Lightning with mixed-precision support
- **Simple configuration** — YAML-based experiment configs with `Hydra <https://hydra.cc/>`__

Get started in 30 seconds:

.. code-block:: bash

   pip install nemo_toolkit[asr,tts]

.. code-block:: python

   import nemo.collections.asr as nemo_asr
   model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
   print(model.transcribe(["audio.wav"])[0].text)


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: starthere

   starthere/install
   starthere/ten_minutes
   starthere/key_concepts
   starthere/choosing_a_model
   starthere/tutorials


.. toctree::
   :maxdepth: 1
   :caption: Training
   :name: Training

   features/parallelisms
   features/mixed_precision

.. toctree::
   :maxdepth: 1
   :caption: Model Checkpoints
   :name: Checkpoints

   checkpoints/intro

.. toctree::
   :maxdepth: 1
   :caption: APIs
   :name: APIs
   :titlesonly:

   apis

.. toctree::
   :maxdepth: 1
   :caption: Collections
   :name: Collections
   :titlesonly:

   collections

.. toctree::
   :maxdepth: 1
   :caption: Speech AI Tools
   :name: Speech AI Tools
   :titlesonly:

   tools/intro
