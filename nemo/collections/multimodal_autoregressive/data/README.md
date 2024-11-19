## MULTIMODAL AUTOREGRESSIVE GENERATION    
This is an example of how to do autoregressive generation for multiple modalities using discrete tokenizer. This example will cover vision understanding (i.e Image to captions). However this can easily be extended to image generation , or to other modalities depending on how you preprocess the data. 


1. Vision Understanding using EMU3 Tokenizer
2. Image generation using Cosmos Tokenizer

### 1. Vision Understanding using EMU3 Tokenizer

#### Download and Extract data 
Download the [COYO700M dataset](https://github.com/kakaobrain/coyo-dataset)

Once downloaded extract the data using tar utilities. 


#### Preprocess data
In the preprocessing script we will do the following 
1. Convert images to discrete vision tokens using [EMU3 Tokenizer](https://github.com/baaivision/Emu3) 
2. Create input data of the format <BOS>You are a helpful assistant. USER: <IMAGE_PROMPT_STRING>Please describe the image. ASSISTANT: <CAPTION><EOS>
3. We will then store it as an indexed dataset. (i.e .bin and .idx files)

Run the preprocessing script as follows : 
```
NUM_GPUS=2
IMAGES_DIR=/path/to/images
CAPTIONS_DIR=/path/to/captions
OUTPUT_PREFIX=/path/to/bin/idx/file


# Make sure you have tiktoken==0.6.0 installed
torchrun --nproc-per-node $NUM_GPUS nemo/collections/multimodal_autoregressive/data/coyo700m/pypreprocess_coyo.py --input_image_dir $IMAGES_DIR --input_captions_dir /$CAPTIONS_DIR --output_prefix $OUTPUT_PREFIX
```

*NOTE* : The images should be of type .jpg, and each image file should have a caption file of type .pkl with the same name as image file. 

#### Train model
Follow usual nemo instructions to train any autoregressive model. 
1. Make sure you have tiktoken (pip install tiktoken==0.6.0)
2. For tokenizer use this : 
```
  tokenizer:
    library: huggingface
    type: BAAI/Emu3-Gen
    model: null
    delimiter: null
    vocab_file: null
    merge_file: null
    sentencepiece_legacy: false
    trust_remote_code: true
```

#### Inference 
To run inference edit the [inference config file](examples/multimodal_autoregressive/conf/megatron_mm_ar_inference_vision_understanding.yaml)
*NOTE* Make sure you have a .nemo file (checkpoint). If you just have a regular megatron checkpoint you  have to do a conversion as shown in [this doc](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt/checkpointconversion.html?highlight=convert)

Run inference as follows

```
torchrun --nproc-per-node 2 examples/multimodal_autoregressive/megatron_mm_autoregressive_eval_vision_understanding.py
```


### 2. Image generation using Cosmos Tokenizer

#### Preprocess data
In the preprocessing script for image generation we will do the following 
1. Download pokemon image captions dataset from hugging face
2. Convert images to discrete vision tokens using [Cosmos Tokenizer](../../../../nemo/collections/common/video_tokenizers/cosmos_tokenizer.py) 
3. Create input data of the format <BOS>You are a helpful assistant. Draw a picture for the caption given by the user. USER: <CAPTION>. ASSISTANT: <IMAGE_PROMPT_STRING><EOS>
4. We will then store it as an indexed dataset. (i.e .bin and .idx files)

Run the preprocessing script as follows : 
```
# Make sure you have tiktoken == 0.6.0 installed

MULTIMODAL_TOKENIZER_PATH=/path/to/nemo/collections/multimodal_autoregressive/tokenizer
OUTPUT_PREFIX=/path/to/bin/idx/file

python nemo/collections/multimodal_autoregressive/data/preprocess_pokemon_blip_cosmos_tokenizer.py --output_prefix $OUTPUT_PREFIX --multimodal_tokenizer_path $MULTIMODAL_TOKENIZER_PATH
```

#### Train model
Follow usual nemo instructions to train any autoregressive model. 
1. Make sure you have tiktoken (pip install tiktoken==0.6.0)
2. For tokenizer use this : 
```
  tokenizer:
    library: huggingface
    type: /path/to/nemo/collections/multimodal_autoregressive/tokenizer
    model: null
    delimiter: null
    vocab_file: null
    merge_file: null
    sentencepiece_legacy: false
    trust_remote_code: true
```

#### Inference 
To run inference edit the [inference config file](examples/multimodal_autoregressive/conf/megatron_mm_ar_inference_image_generation.yaml)
*NOTE* Make sure you have a .nemo file (checkpoint). If you just have a regular megatron checkpoint you  have to do a conversion as shown in [this doc](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt/checkpointconversion.html?highlight=convert)

Run inference as follows

```
torchrun --nproc-per-node 2 examples/multimodal_autoregressive/megatron_mm_autoregressive_eval_image_generation.py
```