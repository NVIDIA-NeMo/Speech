# Use the base image
FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10

# Install additional packages
RUN pip install \
    dash>=2.1.0 \
    dash_bootstrap_components>=1.0.3 \
    diff_match_patch \
    editdistance \
    jiwer \
    librosa>=0.9.1 \
    numpy \
    plotly \
    SoundFile \
    tqdm \
    Flask>=2.2.2 \
    cudf-cu11 --extra-index-url=https://pypi.nvidia.com 

# Uninstall wrong cupy 
RUN pip uninstall \
    cupy -y \
    cupy-cuda11x -y

# Install right cupy
RUN pip install \
    cupy-cuda11x

# Create a new directory
RUN mkdir /workspace

# Set the working directory
WORKDIR /workspace

# Copy the directory from your local machine to the container
COPY  data_explorer.py /workspace/speech_data_explorer/data_explorer.py
COPY  sde /workspace/speech_data_explorer/sde

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]
