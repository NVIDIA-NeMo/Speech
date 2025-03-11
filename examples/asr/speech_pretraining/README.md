# Speech Self-Supervised Learning

This directory contains example scripts to self-supervised speech models. 

There are two main types of supported self-supervised learning methods:
- [Wav2vec-BERT](https://arxiv.org/abs/2108.06209): `speech_pre_training.py`
- [NEST](https://arxiv.org/abs/2408.13106): `masked_token_pred_pretrain.py`
    - For downstream tasks that use NEST as multi-layer feature extractor, please refer to `./downstream/speech_classification_mfa_train.py`
    - For extracting multi-layer features from NEST, please refer to `<NEMO ROOT>/scripts/ssl/extract_features.py`
    - For using NEST as weight initialization for downstream tasks, please refer to the usage of [maybe_init_from_pretrained_checkpoint](https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/modelPT.py#L1242).


For their corresponding usage, please refer to the example yaml config:
- Wav2vec-BERT: `examples/asr/conf/ssl/fastconformer/fast-conformer.yaml`
- NEST: `examples/asr/conf/ssl/nest/nest_fast-conformer.yaml`


The dataset format follows that of ASR models, but no groundtruth transcriptions are needed. For example, the jsonl file specified in `manifest_filepath` should look like:
```
{"audio_filepath": "path/to/audio1.wav", "duration": 10.0, "text": ""}
{"audio_filepath": "path/to/audio2.wav", "duration": 5.0, "text": ""}
```

Please refer to the [ASR dataset documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#preparing-custom-asr-data%60) for more details.


For most efficient data loading, please refer to 
- [lhotse dataloading](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#lhotse-dataloading) 
- [pre-compute bucket durations](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#pre-computing-bucket-duration-bins)
- [optimizing GPU memory usage](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#pushing-gpu-utilization-to-the-limits-with-bucketing-and-oomptimizer)
