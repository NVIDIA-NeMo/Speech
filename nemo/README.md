NeMo (**Ne**ural **Mo**dules) is a toolkit for creating AI applications built around **neural modules**, conceptual blocks of neural networks that take *typed* inputs and produce *typed* outputs.

## **collections/**
* **ASR** - Collection of modules and models for building speech recognition networks.
* **TTS** - Collection of modules and models for building speech synthesis networks.
* **Audio** - Collection of modules and models for building audio processing networks.
* **SpeechLM2** - Collection of modules and models for building multimodal LLM.

## **core/**
Provides fundamental APIs and utilities for NeMo modules, including:
- **Classes** - Base classes for datasets, models, and losses.
- **Config** - Configuration management utilities.
- **Neural Types** - Typed inputs/outputs for module interaction.
- **Optim** - Optimizers and learning rate schedulers.

## **lightning/**
Integration with PyTorch Lightning for training and distributed execution:
- **Strategies & Plugins** - Custom Lightning strategies.
- **Fabric** - Lightweight wrapper for model training.
- **Checkpointing & Logging** - Utilities for managing model states.

## **utils/**
General utilities for debugging, distributed training, logging, and model management:
- **callbacks/** - Hooks for training processes.
- **loggers/** - Logging utilities for different backends.
- **debugging & profiling** - Performance monitoring tools.
