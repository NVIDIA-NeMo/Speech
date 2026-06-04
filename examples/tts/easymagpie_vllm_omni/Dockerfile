FROM nvcr.io/nvidia/tritonserver:26.02-py3

# 1. System dependency for git-based installs
RUN apt-get update && \
    apt-get install -y git sox libsox-fmt-all

# 2. upstream vllm
RUN pip install --no-cache-dir \
    "vllm==0.21.0" \
    "vllm_omni==0.21.0rc1"

# 3. TODO install NeMo/examples/tts/easymagpie_vllm_omni

# 4. Extra python requirements needed to compile the model
RUN pip install --no-cache-dir \
    onnxscript \
    librosa \
    sox \
    onnx-graphsurgeon \
    "tritonclient[grpc]"
RUN pip install --no-cache-dir --force-reinstall --no-deps "numpy==2.3.5"

WORKDIR /workspace
