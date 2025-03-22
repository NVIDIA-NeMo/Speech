# **Checkpoint Conversion Scripts**

Please note that the conversion scripts in this folder (except the [convert_nemo1_to_nemo2.py](https://github.com/NVIDIA/NeMo/blob/main/scripts/checkpoint_converters/convert_nemo1_to_nemo2.py) script) are a part of [the NeMo 1.0 collection](https://docs.nvidia.com/nemo-framework/user-guide/24.07/overview.html) that is no longer actively developed. Please move on to using [the NeMo 2.0 collection](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html) for the latest NeMo Framework features.

For importing HuggingFace models into NeMo, use the ``import_ckpt`` API, which allows you to load pre-trained HuggingFace model checkpoints into the NeMo Framework for further fine-tuning or inference. Please check examples in [the NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/search.html?q=import_ckpt) on how to use this functionality.


**Migrating Checkpoints to NeMo 2.0**

The [convert_nemo1_to_nemo2.py](https://github.com/NVIDIA/NeMo/blob/main/scripts/checkpoint_converters/convert_nemo1_to_nemo2.py) script can be used to convert NeMo 1.0 checkpoints into NeMo 2.0 checkpoint format. For detailed usage instructions or examples, please refer to the [NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemo-2.0/migration/checkpointing.html#convert-nemo-1-0-checkpoint-to-nemo-2-0-checkpoint).
