## NIMification of `.nemo` checkpoint
This guide walks through converting a .nemo checkpoint to Hugging Face (HF) format and deploying it as a NIM using NVIDIA containers.

### 1. Set up
```
git lfs install

mkdir -p nim_deployment/hf_input_model
mkdir -p nim_deployment/hf_output_model
cd nim_deployment/hf_input_model

# Clone base Llama3.1-70B checkpoint from Hugging Face
git clone https://huggingface.co/meta-llama/Llama-3.1-70B
```

### 2. Convert checkpoint from `.nemo` to HF format
```
docker pull nvcr.io/nvidia/nemo:25.04.rc1

docker run --gpus all -it \
  -v /home/nim_deployment:/workspace/nim_deployment \
  nvcr.io/nvidia/nemo:25.04.rc1
```

### Inside the container: 
```
cd /opt/NeMo/scripts/checkpoint_converters/

# Replace megatron_gpt_peft_none_tuning.nemo with your checkpoint name 

python convert_llama_nemo_to_hf.py \
  --input_name_or_path /workspace/nim_deployment/megatron_gpt_peft_none_tuning.nemo \
  --output_path /workspace/nim_deployment/hf_output_model/pytorch_model.bin \
  --hf_input_path /workspace/nim_deployment/hf_input_model/Llama-3.1-70B \
  --hf_output_path /workspace/nim_deployment/hf_output_model \
  --input_tokenizer /workspace/nim_deployment/hf_input_model/Llama-3.1-70B \
  --cpu-only
```

### Copy tokenizer files from original hf model directory to converted Hugging Face directory 

```
cp /workspace/nim_deployment/hf_input_model/Llama-3.1-70B/tokenizer.json \
   /workspace/nim_deployment/hf_output_model/

cp /workspace/nim_deployment/hf_input_model/Llama-3.1-70B/tokenizer_config.json \
   /workspace/nim_deployment/hf_output_model/
```

### 3. Build Engine & deploying the model as a NIM

```
docker pull nvcr.io/nim/meta/llama-3.1-70b-instruct:1.3.3

# Set environment variables
export CONTAINER_NAME=llama-3.1-70b-instruct
export IMG_NAME="nvcr.io/nim/meta/llama-3.1-70b-instruct:1.3.3"
export LOCAL_NIM_CACHE=/home/nim_deployment/downloaded-nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+w "$LOCAL_NIM_CACHE"

export CUSTOM_WEIGHTS=/home/nim_deployment/hf_output_model/
```

### Run the container:
```
docker run -it --rm --name=$CONTAINER_NAME --gpus all \
  -e NIM_FT_MODEL=$CUSTOM_WEIGHTS \
  -e NIM_SERVED_MODEL_NAME="megatron_gpt_peft_none_tuning_nemo" \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_CUSTOM_MODEL_NAME=custom_1 \
  -v $CUSTOM_WEIGHTS:$CUSTOM_WEIGHTS \
  -v /home/nim_deployment:/workspace/nim_deployment \
  -p 8000:8000 \
  -u $(id -u) $IMG_NAME
```

#### Expected Output: 
```
INFO 2025-04-03 01:20:29.195 api_server.py:718] Custom model custom_1 successfully cached.
INFO 2025-04-03 01:20:29.246 server.py:82] Started server process [130]
INFO 2025-04-03 01:20:29.246 on.py:48] Waiting for application startup.
INFO 2025-04-03 01:20:29.255 on.py:62] Application startup complete.
INFO 2025-04-03 01:20:29.257 server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 4. Send a curl request
```
curl -X 'POST' \
 'http://0.0.0.0:8000/v1/chat/completions' \
 -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
  "model": "megatron_gpt_peft_none_tuning_nemo",
  "messages": [
   {
    "role":"user",
    "content":"Hello! How are you?"
   },
   {
    "role":"assistant",
    "content":"Hi! I am quite well, how can I help you today?"
   },
   {
    "role":"user",
    "content":"Can you write me a song?"
   }
  ],
  "top_p": 1,
  "n": 1,
  "max_tokens": 15,
  "stream": true,
  "frequency_penalty": 1.0,
  "stop": ["hello"]
 }'
 ```
### Error Handling
If the above request runs into the following error:

```{"object":"error","message":"As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if the tokenizer does not define one.","type":"BadRequestError","param":null,"code":400}```

Use this instead (with chat_template):
```
# streaming can be set to true or false in the curl command

curl -X 'POST' 'http://0.0.0.0:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "megatron_gpt_peft_none_tuning_nemo",
    "messages": [
      {
        "role": "user",
        "content": "Can you write me a song?"
      }
    ],
    "chat_template": "llama-3",
    "top_p": 1,
    "n": 1,
    "max_tokens": 15,
    "stream": false,
    "frequency_penalty": 1.0,
    "stop": ["hello"]
  }' | python3 -m json.tool
```

#### Example Output: 
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   711  100   389  100   322    332    275  0:00:01  0:00:01 --:--:--   608
{
    "id": "chat-7fff9af934414525b02282ca923cbc70",
    "object": "chat.completion",
    "created": 1743645285,
    "model": "megatron_gpt_peft_none_tuning_nemo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "(L7)..Modified) M (k) =C+,y( x"
            },
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 19,
        "completion_tokens": 15
    },
    "prompt_logprobs": null
}
```