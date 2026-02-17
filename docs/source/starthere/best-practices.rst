.. _best-practices:

Why NeMo?
=========

NeMo simplifies Speech AI development through its modular approach, providing neural modules -- logical blocks of AI applications with typed inputs and outputs -- that enable seamless model construction. This accelerates development, improves accuracy on domain-specific data, and promotes modularity, flexibility, and reusability within AI workflows.

Automatic Speech Recognition (ASR)
-----------------------------------

NeMo provides state-of-the-art ASR models for a wide range of applications:

- **Parakeet** family of models, including the leaderboard-topping Parakeet-TDT, built on the FastConformer encoder architecture
- **Canary** multi-lingual ASR with support for translation and code-switching
- **FastConformer** encoder with CTC, RNNT, and TDT decoder variants
- GPU-accelerated decoding algorithms for real-time and batch transcription
- Multi-language support, including English, Mandarin, German, French, Spanish, and more

Text-to-Speech (TTS)
---------------------

NeMo offers production-ready speech synthesis:

- **MagpieTTS** for high-quality, multi-lingual speech generation
- **FastPitch** spectrogram generator for fast, controllable synthesis
- **HiFi-GAN** neural vocoder for high-fidelity audio waveform generation
- Multi-language and multi-speaker support

Speaker Tasks
-------------

NeMo includes models for speaker-related tasks:

- **Sortformer** for streaming speaker diarization -- determining "who spoke when" in multi-speaker audio
- Speaker recognition and verification models
- Speaker embedding extraction

Speech Language Models (SpeechLM2)
-----------------------------------

NeMo's SpeechLM2 collection enables speech-aware language models:

- **SALM** (Speech-Augmented Language Models), powering models like Canary-Qwen 2.5B
- **Duplex Speech-to-Speech** for real-time conversational AI
- Integration with HuggingFace Transformers for backbone LLMs

Audio Processing
----------------

NeMo provides tools for audio signal processing:

- Speech enhancement for improving audio quality
- Source separation for isolating individual speakers or sounds

Training and Tools
------------------

NeMo provides a comprehensive set of training utilities and tools:

- Multi-GPU and multi-node training via PyTorch Lightning
- Mixed precision training (FP16, BF16) for faster training with lower memory usage
- **NeMo Forced Aligner** for aligning audio with transcripts at word and segment level
- **Speech Data Explorer** for interactive exploration and analysis of ASR/TTS datasets
- **CTC Segmentation** for creating training data from long audio files with transcripts

Resources
---------

- `How to Build Domain Specific Automatic Speech Recognition Models on GPUs <https://developer.nvidia.com/blog/how-to-build-domain-specific-automatic-speech-recognition-models-on-gpus/>`_
- `Develop Smaller Speech Recognition Models with NVIDIA's NeMo Framework <https://developer.nvidia.com/blog/develop-smaller-speech-recognition-models-with-nvidias-nemo-framework/>`_
- `Neural Modules for Fast Development of Speech and Language Models <https://developer.nvidia.com/blog/neural-modules-for-speech-language-models/>`_
