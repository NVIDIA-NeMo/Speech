# NeMo Voice Agent

A fully open-source NVIDIA NeMo Voice Agent example demonstrating a simple way to combine NVIDIA NeMo STT/TTS service and HuggingFace LLM together into a conversational agent. Everything is open-source and deployed locally so you can have your own voice agent. Feel free to explore the code and see how different speech technologies can be integrated with LLMs to create a seamless conversation experience. 

As of now, we only support English input and output, but more languages will be supported in the future.


## ✨ Key Features

- Open-source, local deployment, and flexible customization.
- Allow users to talk to most LLMs from HuggingFace with configurable prompts. 
- Streaming speech recognition with low latency.
- Low latency TTS for fast audio response generation.
- Speaker diarization up to 4 speakers in different user turns.
- WebSocket server for easy deployment.


## 💡 Upcoming Next
- Joint ASR and EOU detection in the same model.
- Accuracy and robustness ASR model improvements.
- Better TTS with more natural voice (e.g., [Magpie-TTS](https://build.nvidia.com/nvidia/magpie-tts-multilingual)).
- Combine ASR and speaker diarization model to handle overlapping speech.


## Latest Updates
- 2025-10-10: Added support for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS model.
- 2025-10-03: Add support for serving LLM with vLLM and auto-switch between vLLM and HuggingFace, add [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) as default LLM.
- 2025-09-05: First release of NeMo Voice Agent.


## 🚀 Quick Start

### Hardware requirements

- A computer with at least one GPU. At least 21GB VRAM is recommended for using 9B LLMs, and 13GB VRAM for 4B LLMs.
- A microphone connected to the computer.
- A speaker connected to the computer.

### Install dependencies

First, install or update the npm and node.js to the latest version, for example:

```bash
sudo apt-get update
sudo apt-get install -y npm nodejs
```

or:

```bash
curl -fsSL https://fnm.vercel.app/install | bash
. ~/.bashrc
fnm use --install-if-missing 20
```

Second, create a new conda environment with the dependencies:

```bash
conda env create -f environment.yaml
```

Then you can activate the environment via `conda activate nemo-voice`.

Alternatively, you can install the dependencies manually in an existing environment via:
```bash
pip install -r requirements.txt
```
The incompatibility errors from pip can be ignored.

### Configure the server

If you want to just try the default server config, you can skip this step.

Edit the `server/server_configs/default.yaml` file to configure the server as needed, for example:
- Changing the LLM and system prompt you want to use in `llm.model` and `llm.system_prompt`, by either putting a local path to a text file or the whole prompt string. See `server/example_prompts/` for examples to start with. 
- Configure the LLM parameters, such as temperature, max tokens, etc. You may also need to change the HuggingFace or vLLM server parameters, depending on the LLM you are using. Please refer to the LLM's model page for details on the recommended parameters.
- If you know whether you want to use vLLM or HuggingFace, you can set `llm.type` to `vllm` or `hf` to force using vLLM or HuggingFace, respectively. Otherwise, it will automatically switch between the two based on the model's support. Please also remember to update the parameters of the chosen backend as well, by referring to the LLM's model page.
- Distribute different components to different GPUs if you have more than one.
- Adjust VAD parameters for sensitivity and end-of-turn detection timeout.

**If you want to access the server from a different machine, you need to change the `baseUrl` in `client/src/app.ts` to the actual ip address of the server machine.**



### Start the server

Open a terminal and run the server via:

```bash
NEMO_PATH=???  # Use your local NeMo path with the latest main branch from: https://github.com/NVIDIA-NeMo/NeMo
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH
# export HF_TOKEN="hf_..."  # Use your own HuggingFace API token if needed, as some models may require.
# export HF_HUB_CACHE="/path/to/your/huggingface/cache"  # change where HF cache is stored if you don't want to use the default cache
# export SERVER_CONFIG_PATH="/path/to/your/server/config.yaml"  # change to the server config you want to use, otherwise it will use the default config in `server/server_configs/default.yaml`
python ./server/server.py
```

### Launch the client
In another terminal on the server machine, start the client via:

```bash
cd client
npm install
npm run dev
```

There should be a message in terminal showing the address and port of the client.

### Connect to the client via browser

Open the client via browser: `http://[YOUR MACHINE IP ADDRESS]:5173/` (or whatever address and port is shown in the terminal where the client was launched). 

You can mute/unmute your microphone via the "Mute" button, and reset the LLM context history and speaker cache by clicking the "Reset" button. 

**If using chrome browser, you need to add `http://[YOUR MACHINE IP ADDRESS]:5173/` to the allow list via `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.**

If you want to use a different port for client connection, you can modify `client/vite.config.js` to change the `port` variable.

## 📑 Supported Models

### 🤖 LLM

Most LLMs from HuggingFace are supported. A few examples are:
- [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) (default)
    - Please use `server/server_configs/llm_configs/nemotron-nano-v2.yaml` as the server config.
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
    - Please use `server/server_configs/llm_configs/qwen2.5-7B.yaml` as the server config.
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
    - Please use `server/server_configs/llm_configs/qwen3-8B.yaml` as the server config.
    - Please use `server/server_configs/llm_configs/qwen3-8B_think.yaml` if you want to enable thinking mode.
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
    - Please use `server/server_configs/llm_configs/llama3.1-8B-instruct.yaml` as the server config.
    - Note that you need to get access to the model first, and specify `export HF_TOKEN="hf_..."` when launching the server.
- [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) 
- [nvidia/Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct)


Please refer to the homepage of each model to configure the model parameters:
- If `llm.type=hf`, please set `llm.generation_kwargs` and `llm.apply_chat_template_kwargs` in the server config as needed.
- If `llm.type=vllm`, please set `llm.vllm_server_params` and `llm.vllm_generation_params`in the server config as needed.
- If `llm.type=auto`, the server will first try to use vLLM, and if it fails, it will try to use HuggingFace. In this case, you need to make sure parameters for both backends are set properly.

You can change the `llm.system_prompt` in `server/server_configs/default.yaml` to configure the behavior of the LLM, by either putting a local path to a text file or the whole prompt string. See `server/example_prompts/` for examples to start with.

#### Thinking/reasoning Mode for LLMs

A lot of LLMs support thinking/reasoning mode, which is useful for complex tasks, but it will create a long latency for the final answer. By default, we turn off the thinking/reasoning mode for all models for best latency.

Different models may have different ways to support thinking/reasoning mode, please refer to the model's homepage for details on their thinking/reasoning mode support. Meanwhile, in many cases, they support enabling thinking/reasoning can be achieved by adding `/think` or `/no_think` to the end of the system prompt, and the thinking/reasoning content is wrapped by the tokens `["<think>", "</think>"]`. Some models may also support enabling thinking/reasoning by setting `llm.apply_chat_template_kwargs.enable_thinking=true/false` in the server config when `llm.type=hf`.

If thinking/reasoning mode is enabled (e.g., in `server/server_configs/qwen3-8B_think.yaml`), the voice agnet server will print out the thinking/reasoning content so that you can see the process of the LLM thinking and still have a smooth conversation experience. The thinking/reasoning content will not go through the TTS process, so you will only hear the final answer, and this is achieved by specifying the pair of thinking tokens `tts.think_tokens=["<think>", "</think>"]` in the server config.

For vLLM server, if you specify `--reasoning_parser` in `vllm_server_params`, the thinking/reasoning content will be filtered out and does not show up in the output.

### 🎤 ASR 

We use [cache-aware streaming FastConformer](https://arxiv.org/abs/2312.17279) to transcribe the user's speech into text. While new models will be released soon, we use the existing English models for now:
- [stt_en_fastconformer_hybrid_large_streaming_80ms](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_80ms)  (default)
- [nvidia/stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)

### 💬 Speaker Diarization

Speaker diarization aims to distinguish different speakers in the input speech audio. We use [streaming Sortformer](http://arxiv.org/abs/2507.18446) to detect the speaker for each user turn. 

As of now, we only support detecting 1 speaker per user turn, but different turns come from different speakers, with a maximum of 4 speakers in the whole conversation. 

Currently supported models are:
 - [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) (default)


Please note that in some circumstances, the diarization model might not work well in noisy environments, or it may confuse the speakers. In this case, you can disable the diarization by setting `diar.enabled` to `false` in `server/server_configs/default.yaml`.

### 🔉 TTS

Here are the supported TTS models:
- [FastPitch-HiFiGAN](https://huggingface.co/nvidia/tts_en_fastpitch) is an NVIDIA-NeMo TTS model. It only supports English output. 
    - Please use `server/server_configs/tts_configs/nemo_fastpitch-hifigan.yaml` as the server config.

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a lightweight TTS model. This model is the default speech generation backend.
    - Please use `server/server_configs/tts_configs/kokoro_82M.yaml` as the server config.

We will support more TTS models in the future.


### Turn-taking

As the new turn-taking prediction model is not yet released, we use the VAD-based turn-taking prediction for now. You can set the `vad.stop_secs` to the desired value in `server/server_configs/default.yaml` to control the amount of silence needed to indicate the end of a user's turn.

Additionally, the voice agent support ignoring back-channel phrases while the bot is talking, which it means phrases such as "uh-huh", "yeah", "okay"  will not interrupt the bot while it's talking. To control the backchannel phrases to be used, you can set the `turn_taking.backchannel_phrases` in the server config to the desired list of phrases or a file path to a yaml file containing the list of phrases. By default, it will use the phrases in `server/backchannel_phrases.yaml`. Setting it to `null` will disable detecting backchannel phrases, and that the VAD will interrupt the bot immediately when the user starts speaking.


## 📝 Notes & FAQ
- Only one connection to the server is supported at a time, a new connection will disconnect the previous one, but the context will be preserved.
- If directly loading from HuggingFace and got I/O erros, you can set `llm.model=<local_path>`, where the model is downloaded using a command like `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir <local_path>`. Same for TTS models.
- The current ASR and diarization models are not noise-robust, you might need to use a noise-cancelling microphone or a quiet environment. But we will release better models soon.
- The diarization model works best with speakers that have much more different voices from each other, while it might not work well on some accents due to the limited training data.
- If you see errors like `SyntaxError: Unexpected reserved word` when running `npm run dev`, please update the Node.js version.
- If you see the error `Error connecting: Cannot read properties of undefined (reading 'enumerateDevices')`, it usually means the browser is not allowed to access the microphone. Please check the browser settings and add `http://[YOUR MACHINE IP ADDRESS]:5173/` to the allow list, e.g., via `chrome://flags/#unsafely-treat-insecure-origin-as-secure` for chrome browser.
- If you see something like `node:internal/errors:496` when running `npm run dev`, remove the `client/node_modules` folder and run `npm install` again, then run `npm run dev` again.



## ☁️ NVIDIA NIM Services

NVIDIA also provides a variety of [NIM](https://developer.nvidia.com/nim?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.nim%3Adesc%2Ctitle%3Aasc&hitsPerPage=12) services for better ASR, TTS and LLM performance with more efficient deployment on either cloud or local servers.

You can also modify the `server/bot_websocket_server.py` to use NVIDIA NIM services for better LLM,ASR and TTS performance, by refering to these Pipecat services:
- [NVIDIA NIM LLM Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nim/llm.py)
- [NVIDIA Riva ASR Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/riva/stt.py)
- [NVIDIA Riva TTS Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/riva/tts.py)
- Please refer to this [NVIDIA ACE Controller example](https://github.com/NVIDIA/ace-controller/blob/main/examples/speech-to-speech/bot.py#L63) for more details on how to use NVIDIA NIM services in the voice agent.

For details of available NVIDIA NIM services, please refer to:
- [NVIDIA NIM LLM Service](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html)
- [NVIDIA Riva ASR NIM Service](https://docs.nvidia.com/nim/riva/asr/latest/overview.html)
- [NVIDIA Riva TTS NIM Service](https://docs.nvidia.com/nim/riva/tts/latest/overview.html)


## Acknowledgments

- This example uses the [Pipecat](https://github.com/pipecat-ai/pipecat) orchestrator framework.



## Contributing

We welcome contributions to this project. Please feel free to submit a pull request or open an issue.

