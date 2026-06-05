FROM nvcr.io/nvidia/tritonserver:26.02-py3

# 1. System dependency for git-based installs
RUN apt-get update && \
    apt-get install -y git sox libsox-fmt-all

# 2. upstream vllm
RUN pip install --no-cache-dir \
    "vllm==0.21.0" \
    "vllm_omni==0.21.0rc1"

# 3. Install easymagpie vllm omni model definition
# TODO: replace this clone with the upstream NeMo repo once the code is merged
RUN git clone --depth 1 --branch easymp_vllm_omni \
        https://github.com/vklimkov-nvidia/NeMoDuplexRealtime.git \
        /tmp/NeMoDuplexRealtime && \
    pip install --no-cache-dir /tmp/NeMoDuplexRealtime/examples/tts/easymagpie_vllm_omni && \
    rm -rf /tmp/NeMoDuplexRealtime

# 4. Extra python requirements needed to compile the model
RUN pip install --no-cache-dir \
    onnxscript \
    librosa \
    sox \
    onnx-graphsurgeon \
    "tritonclient[grpc]"
RUN pip install --no-cache-dir --force-reinstall --no-deps "numpy==2.3.5"

# 5. Restrict vLLM plugin auto-discovery to the EasyMagpie plugin only.
ENV VLLM_PLUGINS=easymagpie_omni

WORKDIR /workspace
