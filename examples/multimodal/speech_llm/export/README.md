## Setup
In this part, we are going to export SALM model into TRTLLM.
First, let's download the [SALM nemo model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speechllm_fc_llama2_7b/) from NVIDIA ngc.

```bash
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/speechllm_fc_llama2_7b/1.23.1/files?redirect=true&path=speechllm_fc_llama2_7b.nemo' -O speechllm_fc_llama2_7b.nemo
```

Then, we need to extract the different parts of SALM.
```bash
output=$PWD/output
python3 extract_salm_weights.py --model_file_path=speechllm_fc_llama2_7b.nemo --output_dir=$output
```
It takes a while to run the above command.

Under the `output` dir, you'll see:
```
output
    |___speechllm_fc_llama2_7b_lora.nemo
    |___speechllm_fc_llama2_7b_perception
    |         |____model_config.yaml
    |         |____model_weights.ckpt
    |___speechllm_fc_llama2_7b_llm.nemo
    |___ xxx.tokenizer.model
```

After we get the lora nemo model and llm nemo model, we can merge the lora part into the llm by:
```bash
python /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py \
    trainer.accelerator=gpu \
    tensor_model_parallel_size=1 \
    pipeline_model_parallel_size=1 \
    gpt_model_file=output/speechllm_fc_llama2_7b_llm.nemo \
    lora_model_path=output/speechllm_fc_llama2_7b_lora.nemo \
    merged_model_path=speechllm_fc_llama2_7b_llm_merged.nemo
```

Now we are able to export the engine by:
```bash
python3 export_salm.py \
    model.perception_model_path=output/speechllm_fc_llama2_7b_perception \
    model.llm_model_path=output/speechllm_fc_llama2_7b_llm_merged.nemo
```

You should be able to get the generated engines under `./salm` folder. To run the engines, you may run:
```python
from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

output_dir = "/ws/salm" # the engine directory
trt_llm_exporter = TensorRTMMExporter(model_dir=output_dir, load_model=True, modality='audio')
input_text = "Q: what's the transcription of the audio? A:"
input_media = '/ws/data/test_audio.wav'
print(trt_llm_exporter.forward(input_text, input_media))

```

## Deploy
If you want to generate the engines and deploy them with Triton Inference Server, you may also run:

```bash
python3 NeMo/scripts/deploy/multimodal/deploy_triton.py \
        --modality="audio" \
        --visual_checkpoint=NeMo/examples/multimodal/speech_llm/export/output/speechllm_fc_llama2_7b_perception \
        --llm_checkpoint=NeMo/examples/multimodal/speech_llm/export/output/speechllm_fc_llama2_7b_llm_merged.nemo \
        --llm_model_type="llama" \
        --model_type="salm" \
        --triton_model_name="salm" \
        --max_input_len=4096 \
        --max_output_len=256 \
        --max_multimodal_len=3072 \
        --triton_model_repository=/tmp/trt_model_dir/
```

And on client side, you may run:
```bash
python3 NeMo/scripts/deploy/multimodal/query.py \
        --model_name="salm" \
        --model_type="salm" \
        --input_text="Q: what's the transcription of the audio? A:" \
        --input_media=/ws/data/test_audio.wav
```

For more details, please check `NeMo/scripts/deploy/multimodal/deploy_triton.py` and ` NeMo/scripts/deploy/multimodal/query.py`.