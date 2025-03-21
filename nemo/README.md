NeMo (**Ne**ural **Mo**dules) is a toolkit for creating AI applications built around **neural modules**, conceptual blocks of neural networks that take *typed* inputs and produce *typed* outputs.

## **collections/**
### **NeMo 2.0 Collections**
* **LLM** - A collection of data modules, models, configurations, and recipes for building training and parameter-efficient fine-tuning (PEFT) pipelines, including decoder-only models like those in the Llama, Gemma, and Mamba families.
* **VLM** - A collection of data modules, models, configurations, and recipes for training and PEFT pipelines in vision-language models.
* **Diffusion** - A collection of data modules, models, configurations, and recipes for training and PEFT pipelines in diffusion models.
* **SpeechLM** - A collection of data modules, models, configurations, and recipes for training and PEFT pipelines in speech-lm models.

### **NeMo 1.0 Collections**
* **ASR** - Collection of modules and models for building speech recognition networks.
* **TTS** - Collection of modules and models for building speech synthesis networks.
* **NLP** - Collection of modules and models for building NLP networks.
* **Vision** - Collection of modules and models for building computer vision networks.
* **Multimodal** - Collection of modules and models for building multimodal networks.
* **Audio** - Collection of modules and models for building audio processing networks.

## **core/**
Provides fundamental APIs and utilities for NeMo modules, including:
- **Classes** - Base classes for datasets, models, and losses.
- **Config** - Configuration management utilities.
- **Neural Types** - Typed inputs/outputs for module interaction.
- **Optim** - Optimizers and learning rate schedulers.

## **deploy/**
Contains utilities for deploying NeMo models using different frameworks:
- **deploy_base.py** - Base deployment classes.
- **deploy_pytriton.py** - Triton inference server deployment.
- **multimodal/** - Deployment utilities for multimodal models.
- **nlp/** - NLP-specific deployment modules.
- **service/** - REST API service utilities.

## **export/**
Contains tools for exporting NeMo models to various formats:
- **ONNX & TensorRT exporters** - Export models for optimized inference.
- **Quantization utilities** - Model compression techniques.
- **VLLM** - Exporting models for vLLM serving.

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

