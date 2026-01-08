# Fine-Tuning Embedding Models with NeMo  

This repository provides an **end-to-end Jupyter Notebook** for fine-tuning embedding models using NVIDIA NeMo. It guides you through dataset preparation, model import, fine-tuning, and experiment monitoring in a single notebook.  

---

## 📓 Notebook  

- **[`Embedding-Finetuning.ipynb`](./Embedding-Finetuning.ipynb)**  
  This notebook integrates all steps of the workflow:  
  1. **Dataset Preparation** – Download and preprocess the [AllNLI Triplet dataset](https://huggingface.co/datasets/sentence-transformers/all-nli).  
  2. **Model Import** – Convert pre-trained Hugging Face models into NeMo format.  
     - Example: `E5-Large-V2`, `LLaMA-3.2-1B`.  
  3. **Fine-Tuning** – Train the embedding models on the custom dataset.  
  4. **Monitoring** – Track experiments and results interactively.  

You can run all steps **inside the notebook** without using any separate `.py` scripts.  

---

## 🎯 Workflow Overview  

1. **Dataset Preparation**  
   - Automatically downloads and formats AllNLI Triplet dataset.  
   - Generates `allnli_triplet.json` with triplet entries for training.  

2. **Model Import**  
   - Imports Hugging Face models (e.g., `e5-large-v2`, `llama-3.2-1b`) into `.nemo` format.  

3. **Fine-Tuning**  
   - Fine-tunes embedding models using NeMo.  
   - Hyperparameters can be edited directly in the notebook cells.  
---

## 🛠️ Prerequisites  

**Hardware**  
- Single node with at least 1 GPU (recommended: A100/H100/H200).  

**Software**  
- [Enroot](https://github.com/NVIDIA/enroot) and NVIDIA Container Toolkit.  
- Optional: Docker.  
- NVIDIA NeMo Container (e.g., `nvcr.io/nvidia/nemo:25.04`).  
- Hugging Face token set as `$HUGGINGFACE_TOKEN`.  

---

## 🚀 Quickstart  

### Launch Jupyter Lab with Enroot  

```bash
enroot import "docker://$HUGGINGFACE_TOKEN@nvcr.io#nvidia/nemo:25.04"
enroot create -n nemo-25.04 "$PWD/nvidia+nemo+25.04.sqsh"

mkdir -p "$PWD/.jupyter_data" \
         "$PWD/.jupyter_runtime" \
         "$PWD/.jupyter_config" \
         "$PWD/.hf_cache" \
         "$PWD/.matplotlib" \
         "$PWD/.triton" \
         "$PWD/.cache" \
         "$PWD/enroot_data" \
         "$PWD/enroot_cache" \
         "$PWD/nemo_home" \
         "$PWD/nemo_cache" \
         "$PWD/nemo_run"

ENROOT_DATA_PATH="$PWD/enroot_data" \
ENROOT_CACHE_PATH="$PWD/enroot_cache" \
enroot start --root \
  --mount "$PWD:/host_pwd" \
  --env NVIDIA_VISIBLE_DEVICES=0,1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env JUPYTER_DATA_DIR=/host_pwd/.jupyter_data \
  --env JUPYTER_RUNTIME_DIR=/host_pwd/.jupyter_runtime \
  --env JUPYTER_CONFIG_DIR=/host_pwd/.jupyter_config \
  --env HF_HOME=/host_pwd/.hf_cache \
  --env MPLCONFIGDIR=/host_pwd/.matplotlib \
  --env TRITON_CACHE_DIR=/host_pwd/.triton \
  --env XDG_CACHE_HOME=/host_pwd/.cache \
  --env NEMO_HOME=/host_pwd/PythonNotebook/nemo_home \
  --env NEMO_MODELS_CACHE=/host_pwd/PythonNotebook/nemo_cache \
  --env NEMO_RUN_DIR=/host_pwd/PythonNotebook/nemo_run \
  nemo-25.04 \
  jupyter-lab --ip=0.0.0.0 --allow-root \
              --NotebookApp.token=“nemo” \
              --port=1234 \
              --notebook-dir=/host_pwd
````

Inside Jupyter Lab:

```bash
!nvidia-smi          # check GPU availability
!huggingface-cli login   # log in to Hugging Face
```

Then open **`Embedding-Finetuning.ipynb`** and run cells step by step.

---

### ⚙️ Optional: Docker

```bash
docker run \
  --gpus all \
  --shm-size=2g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v ${PWD}:/host_pwd \
  -w /host_pwd \
  nvcr.io/nvidia/nemo:25.04 bash
```

Start Jupyter Lab:

```bash
jupyter-lab --ip=0.0.0.0 --allow-root --NotebookApp.token="nemo" --port=1234 --notebook-dir=/host_pwd
```
