#!/bin/bash
sudo apt-get update
sudo apt-get install -y npm nodejs

pip install torch torchvision torchaudio
pip install “nemo-toolkit[asr,tts]”
pip install openai websockets fastapi kokoro python_weather onnxruntime silero-vad
pip install pipecat-ai==0.0.98
pip install vllm